import pandas as pd
import numpy as np
import torch

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





def get_cold_data_from_csv(fname, maxlen=512):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    pid = list(df['protein_id'])
    did = list(df['drug_id'])
    #smiles, protein, affinity, pid = select_seqlen(smiles, protein, affinity, pid, maxlen=maxlen)
    return smiles, protein, affinity, pid, did

def getdata_from_csv(fname, maxlen=512):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    pid = list(df['protein_id'])
    #smiles, protein, affinity, pid = select_seqlen(smiles, protein, affinity, pid, maxlen=maxlen)
    return smiles, protein, affinity, pid

def select_seqlen(smiles, protein, affinity, pid, maxlen=512):    
    len_arr = []
    for i in protein:
        len_arr.append(len(i))
    while maxlen not in len_arr:
        maxlen = maxlen + 1
    sort_zip = list(zip(len_arr,smiles, protein, affinity, pid))
    sort_zip.sort()
    len_arr[:], smiles[:], protein[:], affinity[:], pid[:]= zip(*sort_zip)
    return [smiles[:len_arr.index(maxlen)],protein[:len_arr.index(maxlen)],affinity[:len_arr.index(maxlen)],pid[:len_arr.index(maxlen)]]

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