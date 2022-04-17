#!/usr/bin/env python3
import argparse
from char_extract import arr2sent
from train_char2voc import VariLenDataset, LenGroupedSampler, bipart_dataset
from glob import glob
import h5py
from itertools import repeat
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import Sampler, Subset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.utils.rnn as R
from torch.utils.tensorboard import SummaryWriter
from utils import write_binfile, MagPhaseLoss
from model import Char2Voc, var
from operator import itemgetter
from tqdm import tqdm

parser = argparse.ArgumentParser('train Char2Voc model and pre-train character level embeddings')
parser.add_argument('-c', '--char_data', type=str, default='data/all_char.hdf5')
parser.add_argument('-H', '--hid_size', type=int, default=256)
parser.add_argument('-E', '--epochs', type=int, default=5, help='default=5')
parser.add_argument('-B', '--batch_size', type=int, default=32, help='default=32')
parser.add_argument('-L', '--learning_rate', dest='learning_rate', type=float, default=4e-4, help='default=4e-4')
parser.add_argument('-D', '--dropout', type=float, default=0.5, help='default=0.5')
parser.add_argument('-T', '--test_size', type=int, default=5)
parser.add_argument('--ss', '--scheduled_sampling', action='store_true', dest='scheduled_sampling',
                    help='whether to use scheduled sampling for training, this will override tf_rate')
parser.add_argument('--tf', '--tf_rate', dest='tf_rate', type=float, default=1)
parser.add_argument('--frac', nargs='?', const=40, default=None, type=int)
parser.add_argument('--init', action='store_true')
args = vars(parser.parse_args())

char_path = args['char_data']
batch_size = args['batch_size']
epochs = args['epochs']
learning_rate = args['learning_rate']
val_rate = .05
test_size = args['test_size']
encoded_size = args['hid_size']
embedding_size = 32
mask_rate = .3
model_name = 'maskedLM_en{en}_b{bt}{init}{ss}_tf{tf}_L{lr}'.format(
    init=args['init'] * '_init',
    ss=args['scheduled_sampling'] * '_ss',
    tf=args['tf_rate'],
    en=encoded_size,
    lr=learning_rate,
    dp=args['dropout'],
    bt=args['batch_size'])
model_path = os.path.join('data', model_name + '.torch')
tb_dir = os.path.join('data/tensorboard', model_name)
tb = SummaryWriter(log_dir=tb_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mask(len_seq, m_rate, beginning_ids, end_ids):
    mask_ids = torch.bernoulli(m_rate * torch.ones(end_ids.shape)).nonzero()
    seq_mask = torch.zeros(len_seq, dtype=torch.uint8, device=device)
    for _i in mask_ids:
        seq_mask[beginning_ids[_i]:end_ids[_i]] = 1
    return seq_mask

def type_arr2dic(type_arr, begin_id=1):
    """takes a numpy array of character types, and enumerate into an id2char dict"""
    return dict(enumerate(type_arr.astype('str'), 1))

class PaddedSeqBatch:
    """expects: data == batches of (char_seq, (upper_case, )loss_mask, char_lens)"""
    def __init__(self, data):
        data_T = list(zip(*data))
        self.upper_in = True if len(data_T) == 4 else False
        padded_all = map(R.pad_sequence, data_T[:-1], repeat(True))
        if len(data_T) == 4:
            self.char, self.upper, self.l_mask = padded_all
        else:
            self.char, self.l_mask = padded_all
        in_place = var(torch.arange(self.char.shape[1]).unsqueeze(0))
        self.in_mask = in_place < torch.stack(data_T[1]).unsqueeze(1)
        
    def pin_memory(self):
        for d in (self.char, self.l_mask, self.in_mask):
            d = d.pin_memory()
        if self.upper_in:
            self.upper = self.upper.pin_memory()

def collate_wrapper(batch):
    return PaddedSeqBatch(batch)


if __name__ == "__main__":
    # load character array data
    with h5py.File(char_path, 'r') as f:
        char_dic = {k: np.array(v) for k, v in f.items()}
        char_ls = itemgetter('bow_id', 'eow_id', 'cat_encoded_seq',
                             'cat_is_upper', 'utt_id')(char_dic)
    n_type = char_dic['char_types'].shape[0]
    bow_id, eow_id, cat_char_seq, cat_upper_seq, utt_id = \
        [var(torch.from_numpy(_t.astype(int))) for _t in char_ls]
    loss_mask = get_mask(cat_char_seq.shape[0], mask_rate, bow_id, eow_id)
    assert all(cat_char_seq != 0), np.where(cat_char_seq == 0)
    lens_seq = (utt_id[1:] - utt_id[:-1])
    char_case, up_case, mask_case = map(torch.split, [cat_char_seq, cat_upper_seq, loss_mask],
                                        repeat(torch.unbind(lens_seq)))
    id2char, id2bnd = map(type_arr2dic, 
                          itemgetter('char_types', 'bndry_types')(char_dic))
    id2char[0] = '_'
    all_data = VariLenDataset(char_case, up_case, mask_case, lens_seq)
    used_size = len(char_case) if args['frac'] is None else args['frac']

    # load training, validation, and test data
    train_size = np.floor(used_size * (1 - val_rate)).astype(int)
    train_set, val_set = bipart_dataset(all_data, train_size)
    train_samp, val_samp = map(LenGroupedSampler,
                               [lens_seq[:train_size], lens_seq[train_size:used_size]])
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_samp, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            sampler=val_samp, collate_fn=collate_wrapper)
    test_ids = torch.randint(int(train_size), used_size, (test_size, ))
    test_set = Subset(all_data, test_ids)
    test_loader = DataLoader(test_set, collate_fn=collate_wrapper)
    for _i in test_loader:
        sent = arr2sent(_i.char, id2char, i2bnd_dic=id2bnd, up_indctr=_i.upper)
        print('{:05d}'.format(_i), sent)
        tb.add_text('text_{:05d}/ground_truth'.format(_i), sent, global_step=0)

    # initiate model
    masked_LM = Char2Voc(n_type, encoded_size, 2 * encoded_size, upper_in=True,
                         embedding_size=embedding_size, dropout=args['dropout'],
                         do_maskedLM=True).to(device)
    masked_LM.attention.const_shift = 1
    if args['init']:
        masked_LM.weight_init()
    loss_criterion = nn.CrossEntropyLoss()