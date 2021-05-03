# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:06:26 2020

@author: a
"""

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        #self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, masks):
        query = self.query(x).transpose(1,2)
        value = x
        
        minus_inf = -9e15*torch.ones_like(query)
        e = torch.where(masks>0.5, query, minus_inf)# (B,heads,seq_len)
        a = self.softmax(e)
        
        #out = torch.matmul(a, value).view(query.shape[0], -1)
        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a
 
    
    
           
    
        
        