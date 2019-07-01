#!/usr/bin/env python3
import argparse
from glob import glob
import h5py
from itertools import repeat
import numpy as np
import torch
# import torch.optim as optim
# import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from libutils import shape_assert, write_binfile
from model import init_by_class, Char2Voc
from operator import itemgetter
import os
import soundfile as sf
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser('train Char2Voc model and pre-train character level embeddings')
parser.add_argument('-c', '--char_data', type=str, dafault='data/all_char.hdf5')
parser.add_argument('-d', '--voc_data', type=str, default='data/all_vocoder.hdf5')
parser.add_argument('-v', '--voc_synth_dir', type=str, default='data/synth_voc/', help='default=\'data/synth_voc/\'')
parser.add_argument('-w', '--wav_dir', type=str, default='data/wavs_syn')

parser.add_argument('-E', '--epochs', type=int, default=10, help='default=10')
parser.add_argument('-B', '--batch_size', type=int, default=32, help='default=32')
parser.add_argument('-L', '--learning_rate', dest='learning_rate', type=float, default=4e-4, help='default=4e-4')
parser.add_argument('-D', '--dropout', type=float, default=0.5, help='default=0.5')
parser.add_argument('-T', '--test_size', type=int, default=5)
parser.add_argument('--ss', '--scheduled_sampling', action='store_true', dest='scheduled_sampling',
                    help='whether to use scheduled sampling for training, this will override tf_rate')
parser.add_argument('--tf', '--tf_rate', dest='tf_rate', type=float, default=1)
parser.add_argument('--init', action='store_true')
args = vars(parser.parse_args())

char_path = args['char_data']
voc_path = args['voc_data']  # path to the training data, not to vocoder folder
wav_dir = args['wav_dir']
voc_dir = args['voc_synth_dir']
if not glob(voc_dir):
    os.mkdir(voc_dir)
batch_size = args['batch_size']
epochs = args['epochs']
learning_rate = args['learning_rate']
val_rate = .10
test_size = args['test_size']
model_name = 'srnn_r{init}{ss}_tf{tf}'.format(
    init=args['init'] * '_init',
    ss=args['scheduled_sampling'] * '_ss',
    tf=args['tf_rate'])
tb_dir = os.path.join('data/tensorboard', model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mask(len_seq, r_mask, beginning_ids, end_ids):
    mask_ids = torch.bernoulli(r_mask * torch.ones(end_ids.shape)).nonzero()
    seq_mask = torch.zeros(len_seq, dtype=torch.uint8, device=device)
    for _i in mask_ids:
        seq_mask[beginning_ids[_i]:end_ids[_i]] = 1
    return seq_mask

# def split_1d_cat_index(cat_seq, index):
#     shape_assert(cat_seq, (index[-1], ))
#     len_slices = index[1:] - index[:-1]
#     return torch.split(cat_seq, list(len_slices))
#     # return [cat_seq[index[_i]:index[_i+1]] for _i in range(len(index)-1)]

def synth_char2wav(char2voc_model, inp, i, vocoder_dic):
    voc_cat, voc_uid, voc_mean, voc_std = itemgetter(
        'voc_scaled_cat', 'voc_utt_idx', 'voc_mean', 'voc_std')(vocoder_dic)
    gt_vocoder = voc_cat[voc_uid[i]:voc_uid[i+1]]
    char2voc_model.eval()
    with torch.no_grad():
        out_vocoder = char2voc_model(
            torch.zeros(gt_vocoder.shape, device=device).unsqueeze_(1),
            inp.transpose(0, 1))[1].transpose(0, 1).squeeze()
    out_voiced = torch.bernoulli(out_vocoder[:, -2])
    out_magphase = out_vocoder[:, list(range(80))+[81]] * voc_std.to(device) + voc_mean.to(device)
    out_magphase[:, -1][(1 - out_voiced).byte()] = -1.0e+10
    out_split = torch.split(out_magphase, [60, 10, 10, 1], dim=1)
    for out, ft in zip(out_split, ['mag', 'real', 'imag', 'lf0']):
        write_binfile(out, os.path.join(voc_dir,
                                        'srnn_{:05d}.{}'.format(i, ft)))
    try:
        os.remove(os.path.join(wav_dir, 'srnn_{0:05d}.wav'.format(i)))
    except FileNotFoundError:
        pass
    subprocess.check_output('./voc_extract.py -m synth -o --no_batch ' +
        '-v {0} -w {1} -F srnn_{2:05d}'.format(voc_dir, wav_dir, i), shell=True)
    return sf.read(os.path.join(wav_dir, 'srnn_{0:05d}.wav'.format(i)))[0]


def synth_gt_wavs(i, vocoder_dic):
    voc_cat, voc_uid, voc_mean, voc_std = itemgetter(
        'voc_scaled_cat', 'voc_utt_idx', 'voc_mean', 'voc_std')(vocoder_dic)
    gt_vocoder = voc_cat[voc_uid[i]:voc_uid[i+1]]
    gt_magphase = gt_vocoder[:, list(range(80))+[81]] * voc_std + voc_mean
    assert ((gt_vocoder[:, -2] == 0) + (gt_vocoder[:, -2] == 1)).all(), \
            'wrong dimension for voicedness '
    gt_magphase[:, -1][(1 - gt_vocoder[:, -2]).byte()] = -1.0e+10
    gt_split = torch.split(gt_magphase, [60, 10, 10, 1], dim=1)
    for gt, ft in zip(gt_split, ['mag', 'real', 'imag', 'lf0']):
        write_binfile(gt, os.path.join(voc_dir,
                                       'ground_truth_{:05d}.{}'.format(i, ft)))
    os.system('./voc_extract.py -m synth -o --no_batch' +
              ' -v {0} -w {1} -F ground_truth_{2:05d}'.format(voc_dir, wav_dir, i))
    return sf.read(os.path.join(wav_dir, 'ground_truth_{0:05d}.wav'.format(i)))[0]


if __name__ == '__main__':
    # load character array data
    with h5py.File(char_path, 'r') as f:
        char_dic = {k: np.array(v) for k, v in f.items()}
        char_ls = itemgetter('bow_id', 'eow_id', 'cat_encoded_seq',
                             'cat_is_upper', 'utt_id')(char_dic)
    n_type = char_dic['char_types'].shape[0]
    bow_id, eow_id, cat_char_seq, cat_upper_seq, utt_id = \
        [torch.from_numpy(_t.astype(int)) for _t in char_ls]
    assert all(cat_char_seq.nonzero()), 'encoding needs to start from 1, not 0'  # unpadded encoded arr should not contain 0
    lens_seq = utt_id[1:] - utt_id[:-1]
    # loss_mask = get_mask(cat_char_seq.shape[0], mask_rate, bow_id, eow_id)
    # char_case, up_case, mask_case = map(torch.split,
    #     [cat_char_seq, cat_upper_seq, loss_mask], repeat(len(lens_seq)))
    char_case, up_case = map(torch.split, [cat_char_seq, cat_upper_seq], 
                             repeat(lens_seq))
    
    with h5py.File(voc_path, 'r') as f:
        voc_dic = {k: torch.from_numpy(np.array(v)) for k, v in f.items()}
    voc_utt_id, voc_cat = itemgetter('voc_utt_idx', 'voc_scaled_cat')
    lens_voc = voc_utt_id[1:] - voc_utt_id[:-1]
    voc_case = torch.split(voc)

    char2voc = Char2Voc(n_type, )
