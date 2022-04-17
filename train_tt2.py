#!/usr/bin/env python3
from globo import glob
import h5py
from itertools import repeat
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Sampler, Subset, DataLoader
import torch.nn as nn
import torch.nn.utils.rnn as R
from torch.utils.tensorboard import SummaryWriter
from model import Tacotron2
from utils import write_binfile, var, VariLenDataset  # , get_mask_from_lengths, MagPhaseLoss
from operator import itemgetter
import os
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse
from utils import parse_tacotron2_args
parent_parser = argparse.ArgumentParser(add_help=False)
parser = parse_tacotron2_args(parent_parser)
parser.add_argument('-c', '--char_data', type=str, default='data/all_char.hdf5')
parser.add_argument('-d', '--voc_data', type=str, default='data/all_vocoder.hdf5')
parser.add_argument('-v', '--voc_synth_dir', type=str, default='data/synth_voc/', 
                    help='default=\'data/synth_voc/\'')
parser.add_argument('-w', '--wav_dir', type=str, default='data/wavs_syn')

parser.add_argument('-E', '--epochs', type=int, default=5, help='default=5')
parser.add_argument('-B', '--batch_size', type=int, default=32, help='default=32')
parser.add_argument('-L', '--learning_rate', dest='learning_rate', 
                    type=float, default=4e-4, help='default=4e-4')
parser.add_argument('-D', '--dropout', type=float, 
                    default=0.5, help='default=0.5')
parser.add_argument('-T', '--test_size', type=int, default=5)
parser.add_argument('--ss', '--scheduled_sampling', action='store_true', 
                    dest='scheduled_sampling',
                    help='whether schedule sampling; will override tf_rate')
parser.add_argument('--tf', '--tf_rate', dest='tf_rate', type=float, default=1)
parser.add_argument('--frac', nargs='?', const=40, default=None, type=int)
parser.add_argument('--init', action='store_true')
parser.add_argument('--no_voice', action='store_true', default=False)
args = vars(parser.parse_args())

if not args['no_voice']:
    import soundfile as sf
char_path = args['char_data']
voc_path = args['voc_data']  # path to the training data, not to vocoder folder
wav_dir = args['wav_dir']
voc_dir = args['voc_synth_dir']
if not glob(voc_dir):
    os.mkdir(voc_dir)
batch_size = args['batch_size']
epochs = args['epochs']
learning_rate = args['learning_rate']
val_rate = .05
test_size = args['test_size']
model_name = 'char2voc_em{en}_b{bt}{init}{ss}_d{dp}_tf{tf}_L{lr}_mse'.format(
    init=args['init'] * '_init',
    ss=args['scheduled_sampling'] * '_ss',
    tf=args['tf_rate'],
    lr=learning_rate,
    dp=args['dropout'],
    bt=args['batch_size'])
model_path = os.path.join('data', model_name + '.torch')
tb_dir = os.path.join('data/tensorboard', model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_vocoder_seqs(h5_path):
    with h5py.File(h5_path, 'r') as f:
        voc_dic = {k: var(torch.from_numpy(np.array(v))) for k, v in f.items()}
    global voc_cat, voc_uid, voc_mean, voc_std, lens_voc, sampling_rate
    voc_cat, voc_uid, voc_mean, voc_std = itemgetter(
        'voc_scaled_cat', 'voc_utt_idx', 'voc_mean', 'voc_std')(voc_dic)
    lens_voc = voc_uid[1:] - voc_uid[:-1]
    sampling_rate = int(voc_dic.get('sampling_rate', 48000))
    voc_case = torch.split(voc_cat, torch.unbind(lens_voc), dim=0)
    return voc_case


def get_char_seqs(h5_path):
    with h5py.File(h5_path, 'r') as f:
        char_dic = {k: np.array(v) for k, v in f.items()}
        char_ls = itemgetter('bow_id', 'eow_id', 'cat_encoded_seq',
                             'cat_is_upper', 'utt_id')(char_dic)

    n_type = char_dic['char_types'].shape[0]
    print(n_type)
    bow_id, eow_id, cat_char_seq, cat_upper_seq, utt_id = \
        [var(torch.from_numpy(_t.astype(int))) for _t in char_ls]
    assert all(cat_char_seq != 0), "All character indices must be non-zero" # np.where(cat_char_seq == 0)
    lens_seq = (utt_id[1:] - utt_id[:-1])
    char_case, up_case = map(torch.split, [cat_char_seq, cat_upper_seq],
                             repeat(torch.unbind(lens_seq)))
