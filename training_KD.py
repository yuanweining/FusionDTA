import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid, ci, kd_loss
from src.models.DAT import DAT3



parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=256, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--embedding-dim', type=int, default=1280, help='dimension of embedding (default: 512)')
parser.add_argument('--rnn-dim', type=int, default=128, help='hidden unit/s of RNNs (default: 256)')
parser.add_argument('--hidden-dim', type=int, default=256, help='hidden units of FC layers (default: 256)')
parser.add_argument('--graph-dim', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')
parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')
parser.add_argument('--training-dataset-path', default='data/davis_train.csv', help='training dataset path: davis or kiba/ 5-fold or not')
parser.add_argument('--testing-dataset-path', default='data/davis_test.csv', help='training dataset path: davis or kiba/ 5-fold or not')


args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()

batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay

embedding_dim = args.embedding_dim
rnn_dim = args.rnn_dim
hidden_dim = args.hidden_dim
graph_dim = args.graph_dim

n_heads = args.n_heads
dropout = args.dropout
alpha = args.alpha

is_pretrain = args.pretrain

Alphabet = AminoAcid()
training_dataset_address = args.training_dataset_path
testing_dataset_address = args.testing_dataset_path

#processing training data
if is_pretrain:
    train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)

else:
    train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
    train_protein = [x.encode('utf-8').upper() for x in train_protein]
    train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
train_affinity = torch.from_numpy(np.array(train_affinity)).float()

dataset_train = DrugTargetDataset(train_drug, train_protein, train_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)
dataloader_train = torch.utils.data.DataLoader(dataset_train
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

#processing testing data
if is_pretrain:
    test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address, maxlen=1536)
else:
    test_drug, test_protein, test_affinity = getdata_from_csv(testing_dataset_address, maxlen=1024)
    test_protein = [x.encode('utf-8').upper() for x in test_protein]
    test_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in test_protein]
test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(test_drug, test_protein, test_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)
dataloader_test = torch.utils.data.DataLoader(dataset_test
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

#model with half parameter
model = DAT3(embedding_dim, int(rnn_dim/2), int(hidden_dim/2), int(graph_dim/2), dropout, alpha, int(n_heads/2), is_pretrain=is_pretrain)



if use_cuda:
    model.cuda()
    

teacher_model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)
teacher_model.load_state_dict(torch.load('saved_models/DAT_best_davis.pkl')['model'], strict=False)
teacher_model = teacher_model.cuda()
#optimizer
params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.Adam(params, lr=lr)
criterion = nn.MSELoss()


train_epoch_size = len(train_drug)
test_epoch_size = len(test_drug)

print('--- GAT-KD model --- \n')

best_ci = 0
best_mse = 100000
for epoch in range(epochs):
    
    #train
    model.train()
    b = 0
    total_loss = []
    total_ci = []
    for protein, smiles, affinity in dataloader_train:

        if use_cuda:

            protein = [p.cuda() for p in protein]
            smiles = [s.cuda() for s in smiles]
            affinity = affinity.cuda()
        
        guide, out = model(protein, smiles)
        hint, teacher_outputs = teacher_model(protein, smiles)
        hint, teacher_outputs = hint.detach(), teacher_outputs.detach()
        loss = kd_loss(guide, hint, out.unsqueeze(1), affinity.unsqueeze(1))
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        out = out.cpu()
        affinity = affinity.cpu()
        loss = loss.cpu().detach()
        
        c_index = ci(affinity.detach().numpy(),out.detach().numpy())
        #c_index = get_cindex(affinity.reshape(-1), out.reshape(-1))
        
        b = b + batch_size
        total_loss.append(loss)
        total_ci.append(c_index)
        
        print('# [{}/{}] training {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                    , epochs
                                                                    , b/train_epoch_size
                                                                    , loss 
                                                                    , c_index
                                                                    )
                     , end='\r')
    
        
    print('total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), np.mean(total_ci)))
        
    model.eval()
    b=0
    total_loss = []
    total_ci = []
    total_pred = torch.Tensor()
    total_label = torch.Tensor()
    with torch.no_grad():
        for protein, smiles, affinity in dataloader_test:
            
            if use_cuda:
                protein = [p.cuda() for p in protein]
                smiles = [s.cuda() for s in smiles]
                affinity = affinity.cuda()
            
            _, out = model(protein, smiles)
            
            loss = criterion(out, affinity)
            
            out = out.cpu()
            affinity = affinity.cpu()
            loss = loss.cpu().detach()
            c_index = ci(affinity.detach().numpy(),out.detach().numpy())
            #c_index = get_cindex(affinity.reshape(-1), out.reshape(-1))
            
            b = b + batch_size
            total_loss.append(loss)
            total_ci.append(c_index)
            total_pred = torch.cat((total_pred, out), 0)
            total_label = torch.cat((total_label, affinity), 0)
            
            print('# [{}/{}] testing {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                        , epochs
                                                                        , b/test_epoch_size
                                                                        , loss 
                                                                        , c_index
                                                                        )
            #, flush=True)
            , end='\r')
    all_ci = ci(total_label.detach().numpy().flatten(),total_pred.detach().numpy().flatten())
    print('total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), all_ci))
    save_path = 'saved_models/KD_best_davis.pkl'

    if all_ci > best_ci:
        best_ci = all_ci
        model.cpu()
        save_dict = {'model':model.state_dict(), 'optim':optim.state_dict(), 'ci':best_ci}
        torch.save(save_dict, save_path)
        if use_cuda:
            model.cuda()
            
            