import pandas as pd
import numpy as np
import torch
import torch.utils.data
from rdkit import Chem
import networkx as nx
from rdkit.Chem import MACCSkeys

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile, self_link=True):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature)
    
    features = torch.from_numpy(np.array(features, dtype=float))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    if self_link:
        for i in range(c_size):
            edges.append([i,i])
    g = nx.Graph(edges).to_directed()
    #adj = np.array(nx.adjacency_matrix(g).todense())
    adj = torch.from_numpy(np.array(nx.adjacency_matrix(g).todense()))
    return c_size, features, adj

def smile_to_label(smile):
    mol = Chem.MolFromSmiles(smile)
    fingerprints = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    label = fingerprints[1:]
    label = torch.from_numpy(np.array(list(label), dtype=float))
    return label




def getdata_from_csv(fname, maxlen=512):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    smiles, protein, affinity = select_seqlen(smiles, protein, affinity, maxlen=maxlen)
    return smiles, protein, affinity

def select_seqlen(smiles, protein, affinity, maxlen=512):    
    len_arr = []
    for i in protein:
        len_arr.append(len(i))
    while maxlen not in len_arr:
        maxlen = maxlen + 1
    sort_zip = list(zip(len_arr,smiles, protein, affinity))
    sort_zip.sort()
    len_arr[:], smiles[:], protein[:], affinity[:]= zip(*sort_zip)
    return [smiles[:len_arr.index(maxlen)],protein[:len_arr.index(maxlen)],affinity[:len_arr.index(maxlen)]]

def getsmiles_from_csv(fname):
    file1 = './data/'+fname+'_train.csv'
    file2 = './data/'+fname+'_test.csv'
    df = pd.read_csv(file1)
    smiles1 = list(df['compound_iso_smiles'])
    df = pd.read_csv(file2)
    smiles2 = list(df['compound_iso_smiles'])
    smiles = smiles1+smiles2
    return list(set(smiles))

'''
def add_designation(fname):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    designation = {}
    for i, s in enumerate(set(smiles)):
        designation[s] = i
    print(designation)
'''
        
    
#add_designation('../data/davis_train.csv')