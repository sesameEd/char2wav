#!/usr/bin/python3
import h5py
import torch
import numpy as np
from model import init_weights, shape_assert, Char2Voc
import argparse

parser = argparse.ArgumentParser('train Char2Voc model and pre-train character level embeddings')
parser.add_argument('--data', type=str, dafault='data/all_char.hdf5')
parser.add_argument('--mr', '--masking_rate', dest='masking_rate', type=float, default=0.15)
args = vars(parser.parse_args())
data_path = args['data']
mask_rate = args['masking_rate']

if __name__ == '__main__':
    with h5py.File(data_path, 'r') as f:
        char_dic = {k:torch.from_numpy(np.array(v)) for k,v in f.items()}
    masked_wid = torch.bernoulli(mask_rate * torch.ones(char_dic['bow_id'].shape))
    
