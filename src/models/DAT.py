# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:50:04 2020

@author: a
"""

from src.models.layers import LinkAttention
import torch
import torch.nn as nn
import numpy as np
from src.utils import pack_sequences, pack_pre_sequences, unpack_sequences, split_text, load_protvec, graph_pad
from torch.nn.utils.rnn import pack_padded_sequence ,pad_packed_sequence
import torch.nn.functional as F

device_ids = [0,1,2,3]

class DAT3(nn.Module):
    def __init__(self, embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout_rate, 
                     alpha, n_heads, graph_input_dim=78, rnn_layers=2, n_attentions=1, 
                     attn_type='dotproduct', vocab=26, smile_vocab = 63, is_pretrain = True,
                     is_drug_pretrain = False, n_extend=1):
        super(DAT3, self).__init__()
        
        # drugs
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.n_attentions = n_attentions
        self.n_heads = n_heads
        self.graph_head_fc1 = nn.Linear(graph_dim*n_heads, graph_dim)
        self.graph_head_fc2 = nn.Linear(graph_dim*n_heads, graph_dim)
        self.graph_out_fc = nn.Linear(graph_dim, hidden_dim)
        self.out_attentions1 = LinkAttention(hidden_dim, n_heads)
        
        #SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab+1, 256, padding_idx=smile_vocab)
        self.rnn_layers = 2
        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, rnn_dim)
        self.smiles_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                      , bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.smiles_out_fc = nn.Linear(rnn_dim*2, rnn_dim)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)
        
        #protein
        self.is_pretrain = is_pretrain
        if not is_pretrain:
            self.vocab = vocab
            self.embed = nn.Embedding(vocab+1, embedding_dim, padding_idx=vocab)
        self.rnn_layers = 2
        self.is_bidirectional = True
        self.sentence_input_fc = nn.Linear(embedding_dim, rnn_dim)
        self.encode_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                      , bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.rnn_out_fc = nn.Linear(rnn_dim*2, rnn_dim)
        self.sentence_head_fc = nn.Linear(rnn_dim*n_heads, rnn_dim)
        self.sentence_out_fc = nn.Linear(2*rnn_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)
           
        #link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim*3, 256*8)
        self.out_fc2 = nn.Linear(256*8, hidden_dim*2)
        self.out_fc3 = nn.Linear(hidden_dim*2, 1)
        self.layer_norm = nn.LayerNorm(rnn_dim*2)
        
        
        
        
    def forward(self, protein, smiles):
        #drugs
        batchsize = len(protein)
        
        smiles_lengths = np.array([len(x) for x in smiles])
        temp = (torch.zeros(batchsize, max(smiles_lengths))*63).long()
        for i in range(batchsize):
            temp[i,:len(smiles[i])] = smiles[i]
        smiles = temp.cuda()
        smiles = self.smiles_embed(smiles)
        smiles = self.smiles_input_fc(smiles)
        smiles_out, _ = self.smiles_rnn(smiles)
        smiles_mask = self.generate_masks(smiles_out, smiles_lengths, self.n_heads)
        smiles_cat, smile_attn = self.out_attentions3(smiles_out, smiles_mask)
        
        #proteins
        if self.is_pretrain:
            protein_lengths = np.array([x.shape[0] for x in protein])
            protein = graph_pad(protein, max(protein_lengths))
        
        h = self.sentence_input_fc(protein)
        sentence_out, _ = self.encode_rnn(h)
        sent_mask = self.generate_masks(sentence_out, protein_lengths, self.n_heads)
        sent_cat, sent_attn = self.out_attentions2(sentence_out, sent_mask)
        
        #drugs and proteins
        h = torch.cat((smiles_out, sentence_out), dim=1)
        out_masks = self.generate_out_masks(smiles_lengths, smiles_mask, sent_mask, protein_lengths, self.n_heads)
        out_cat, out_attn = self.out_attentions(h, out_masks)

        out = torch.cat([smiles_cat, sent_cat, out_cat], dim=1)

        d_block = self.dropout(self.relu(self.out_fc1(out)))
        out = self.dropout(self.relu(self.out_fc2(d_block)))    
        out = self.out_fc3(out).squeeze()
        

        return d_block, out
        
        
    
    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        for e_id, drug_len in enumerate(adj_sizes):
            out[e_id, drug_len : max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)
    
    def generate_out_masks(self, drug_sizes, adj, masks, source_lengths, n_heads):
        adj_size = adj.shape[2]
        sen_size = masks.shape[2]
        maxlen = adj_size + sen_size
        out = torch.ones(adj.shape[0], maxlen)
        for e_id in range(len(source_lengths)):
            src_len = source_lengths[e_id]
            drug_len = drug_sizes[e_id]
            out[e_id, drug_len : adj_size] = 0
            out[e_id, adj_size + src_len:] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1) 
        return out.cuda(device=adj.device)

        

        
