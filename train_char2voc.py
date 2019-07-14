#!/usr/bin/env python3
import argparse
from glob import glob
import h5py
from itertools import repeat
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Sampler, Subset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.utils.rnn as R
from torch.utils.tensorboard import SummaryWriter
from utils import write_binfile, MagPhaseLoss
from model import init_by_class, Char2Voc
from operator import itemgetter
import os
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser('train Char2Voc model and pre-train character level embeddings')
parser.add_argument('-c', '--char_data', type=str, default='data/all_char.hdf5')
parser.add_argument('-d', '--voc_data', type=str, default='data/all_vocoder.hdf5')
parser.add_argument('-v', '--voc_synth_dir', type=str, default='data/synth_voc/', help='default=\'data/synth_voc/\'')
parser.add_argument('-w', '--wav_dir', type=str, default='data/wavs_syn')

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
encoded_size = args['hid_size']
embedding_size = 32
model_name = 'char2v_en{en}_b{bt}{init}{ss}_tf{tf}_d{dp}'.format(
    init=args['init'] * '_init',
    ss=args['scheduled_sampling'] * '_ss',
    tf=args['tf_rate'],
    dp=args['dropout'],
    en=encoded_size,
    bt=args['batch_size'])
model_path = os.path.join('data', model_name + '.torch')
tb_dir = os.path.join('data/tensorboard', model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mask(len_seq, r_mask, beginning_ids, end_ids):
    mask_ids = torch.bernoulli(r_mask * torch.ones(end_ids.shape)).nonzero()
    seq_mask = torch.zeros(len_seq, dtype=torch.uint8, device=device)
    for _i in mask_ids:
        seq_mask[beginning_ids[_i]:end_ids[_i]] = 1
    return seq_mask

def bipart_dataset(complete_set, split_index):
    return (Subset(complete_set, range(split_index)),
            Subset(complete_set, range(split_index, len(complete_set))))

def synth_scaled(out_vocoder, i):
    out_voiced = torch.bernoulli(out_vocoder[:, -2])
    out_magphase = out_vocoder[:, list(range(80))+[81]] * voc_std.to(device) \
                   + voc_mean.to(device)
    out_magphase[:, -1][(1 - out_voiced).byte()] = -1.0e+10
    out_split = torch.split(out_magphase, [60, 10, 10, 1], dim=1)
    wav_tkn = 'char2v_{0:05d}.wav'.format(i)
    wav_file = os.path.join(wav_dir, wav_tkn + '.wav')
    for out, ft in zip(out_split, ['mag', 'real', 'imag', 'lf0']):
        write_binfile(out, os.path.join(voc_dir, wav_tkn + '.' + ft))
                                        # 'char2v_{:05d}.{}'.format(i, ft)))
    if glob(wav_file):
        os.remove(wav_file)
    subprocess.check_output('./voc_extract.py -m synth -o --no_batch -F ' + wav_tkn +
                            ' -v {0} -w {1}'.format(voc_dir, wav_dir), shell=True)
    return sf.read(wav_file)[0]


def synth_gt_wavs(i):
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


class VariLenDataset(Dataset):
    """takes lists of tensors of variable lengths as
    Each sample will be retrieved by indexing lists.
    list(char_seq), list(voc_seq), tensor( )(, list(upper_case))
    """
    def __init__(self, *ls_cases):
        assert all(len(ls_cases[0]) == len(case) for case in ls_cases), \
            "Expected lists with same #training cases, got {}".format(list(map(len, ls_cases)))
        self.ls_cases = ls_cases
    
    def __getitem__(self, index):
        return tuple(case[index] for case in self.ls_cases)

    def __len__(self):
        return len(self.ls_cases[0])

class LenGroupedSampler(Sampler):
    def __init__(self, lens):
        """lens is a torch.tensor where the length of each sequence is stored"""
        self.sorted_ids = torch.argsort(lens, descending=True)

    def __iter__(self):
        return iter(self.sorted_ids)

    def __len__(self):
        return self.sorted_ids.shape[0]

class PaddedBatch:
    def __init__(self, data):
        """expects data = [char_seq, voc_seq, char_lens, voc_lens(, upper_case)]
        each being a batch.  """
        transposed_data = list(zip(*data))
        self.char = R.pad_sequence(transposed_data[0], batch_first=True)
        self.voc = R.pad_sequence(transposed_data[1], batch_first=True)
        inp_places = torch.arange(self.char.shape[1]).unsqueeze(0).to(device)
        self.in_mask = inp_places < torch.stack(transposed_data[2]).unsqueeze(1)
        voc_places = torch.arange(self.voc.shape[1]).unsqueeze(0).to(device)
        self.voc_mask = voc_places < torch.stack(transposed_data[3]).unsqueeze(1)
        try:
            self.upper = R.pad_sequence(transposed_data[4], batch_first=True)
        except IndexError:
            self.upper = None

    def pin_memory(self):
        self.char = self.char.pin_memory()
        self.voc = self.voc.pin_memory()
        self.in_mask = self.in_mask.pin_memory()
        self.voc_mask = self.voc_mask.pin_memory()
        if self.upper is not None:
            self.upper = self.upper.pin_memory()
        return self

def collate_wrapper(batch):
    return PaddedBatch(batch)


if __name__ == '__main__':
    # load vocoder array data
    with h5py.File(voc_path, 'r') as f:
        voc_dic = {k: torch.from_numpy(np.array(v)).to(device) for k, v in f.items()}
    global voc_cat, voc_uid, voc_mean, voc_std
    voc_cat, voc_uid, voc_mean, voc_std = itemgetter(
        'voc_scaled_cat', 'voc_utt_idx', 'voc_mean', 'voc_std')(voc_dic)
    lens_voc = voc_uid[1:] - voc_uid[:-1]
    sampling_rate = int(voc_dic.get('sampling_rate', 48000))
    voc_case = torch.split(voc_cat, torch.unbind(lens_voc), dim=0)

    # load character array data
    with h5py.File(char_path, 'r') as f:
        char_dic = {k: np.array(v) for k, v in f.items()}
        char_ls = itemgetter('bow_id', 'eow_id', 'cat_encoded_seq',
                             'cat_is_upper', 'utt_id')(char_dic)
    n_type = char_dic['char_types'].shape[0]
    bow_id, eow_id, cat_char_seq, cat_upper_seq, utt_id = \
        [torch.from_numpy(_t.astype(int)).to(device) for _t in char_ls]
    # loss_mask = get_mask(cat_char_seq.shape[0], mask_rate, bow_id, eow_id)
    assert all(cat_char_seq != 0), np.where(cat_char_seq == 0)
    lens_seq = (utt_id[1:] - utt_id[:-1])
    char_case, up_case = map(torch.split, [cat_char_seq, cat_upper_seq],
                             repeat(torch.unbind(lens_seq)))
    used_size = len(voc_case) if args['frac'] is None else args['frac']
    all_data = VariLenDataset(char_case[:used_size], voc_case[:used_size],
                              lens_seq[:used_size], lens_voc[:used_size],
                              up_case[:used_size])

    # split train, val set; and group batches by lengths
    train_size = np.floor(used_size * (1 - val_rate)).astype(int)
    train_set, val_set = bipart_dataset(all_data, train_size)
    train_samp, val_samp = map(LenGroupedSampler,
                               [lens_voc[:train_size], lens_voc[train_size:used_size]])
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True,
                              sampler=train_samp, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True,
                            sampler=val_samp, collate_fn=collate_wrapper)

    # synthesize ground truth wavs
    test_ids = torch.randint(int(train_size), used_size, (test_size, ))
    test_set = Subset(all_data, test_ids)
    test_loader = DataLoader(test_set, collate_fn=collate_wrapper)
    tb = SummaryWriter(log_dir=tb_dir)
    for _i in test_ids:
        if args['no_voice']:
            break
        tb.add_audio('wav{:05d}/ground_truth'.format(_i), synth_gt_wavs(_i),
                     sample_rate=sampling_rate, global_step=0)

    # load model
    char2voc = Char2Voc(n_type, encoded_size, 2 * encoded_size, upper_in=True,
                        embedding_size=embedding_size).to(device)
    if args['init']:
        # init_Module(char2voc)
        init_by_class[char2voc.__class__](char2voc)
    loss_criterion = MagPhaseLoss(batch_size=batch_size)
    optimizer = optim.Adam(char2voc.parameters(), lr=learning_rate)

    tf = args['tf_rate']
    id_loss = 0
    for _e in range(1, epochs + 1):
        if args['scheduled_sampling']:
            tf = 1 - _e / epochs
        char2voc.train()
        losses = []
        for d in tqdm(train_loader):
            char2voc.gen_init(batch_size)
            y = char2voc(d.char, d.voc, input_mask=d.in_mask, voc_mask=d.voc_mask,
                         tf_rate=tf, upper_case=d.upper)
            loss = loss_criterion(y, d.voc)
            loss.backward()
            nn.utils.clip_grad_norm_(char2voc.parameters(), 5.)
            optimizer.step()
            tb.add_scalar('loss/train', loss, id_loss)
            id_loss += 1
            losses.append(loss.item())
        print('Epoch: %d Training Loss: %.3f; ' % (_e, np.mean(losses)), end='| ')

        char2voc.eval()
        dev_loss = []
        for d in val_loader:
            char2voc.gen_init(batch_size)
            with torch.no_grad():
                y = char2voc(d.char, d.voc, input_mask=d.in_mask, voc_mask=d.voc_mask,
                             upper_case=d.upper, tf_rate=0)
            loss = loss_criterion(y, d.voc)
            dev_loss.append(loss.item())
        tb.add_scalar('loss/dev', np.mean(dev_loss), id_loss)
        print('Dev Loss: %.3f' % (np.mean(dev_loss)))

        for _i, d in zip(test_ids.unbind(), test_loader):
            if args['no_voice']:
                break
            char2voc.gen_init(1)
            with torch.no_grad():
                y = char2voc(d.char, d.voc, upper_case=d.upper,
                             input_mask=d.in_mask, tf_rate=0)
            tb.add_audio('wav{:05}/model'.format(_i),
                         synth_scaled(y.squeeze(0), _i),
                         global_step=id_loss,
                         sample_rate=sampling_rate)
    
    torch.save(char2voc.state_dict(), model_path)
