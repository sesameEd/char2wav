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
from utils import write_binfile, MagPhaseLoss, var, get_mask_from_lengths, VariLenDataset
from model import Char2Voc  # , var, init_by_class
from operator import itemgetter
import os
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

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
val_rate = .02
test_size = args['test_size']
embedding_size = args['hid_size']
model_name = 'char2voc_em{en}_b{bt}{init}{ss}_d{dp}_tf{tf}_L{lr}'.format(
    init=args['init'] * '_init',
    ss=args['scheduled_sampling'] * '_ss',
    tf=args['tf_rate'],
    en=embedding_size,
    lr=learning_rate,
    dp=args['dropout'],
    bt=args['batch_size'])
model_path = os.path.join('data', model_name + '.torch')
tb_dir = os.path.join('data/tensorboard', model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def bipart_dataset(complete_set, split_index):
    return (Subset(complete_set, range(split_index)),
            Subset(complete_set, range(split_index, len(complete_set))))


def synth_scaled(out_vocoder, i):
    out_voiced = torch.bernoulli(out_vocoder[:, -2])
    out_magphase = out_vocoder[:, list(range(80))+[81]] * var(voc_std) \
                   + var(voc_mean)
    out_magphase[:, -1][(1 - out_voiced).byte()] = -1.0e+10
    out_split = torch.split(out_magphase, [60, 10, 10, 1], dim=1)
    wav_tkn = 'char2v_{0:05d}.wav'.format(i)
    wav_file = os.path.join(wav_dir, wav_tkn + '.wav')
    for out, ft in zip(out_split, ['mag', 'real', 'imag', 'lf0']):
        write_binfile(out, os.path.join(voc_dir, wav_tkn + '.' + ft))
                                        # 'char2v_{:05d}.{}'.format(i, ft)))
    if glob(wav_file):
        os.remove(wav_file)
    cmd_line = ' '.join(['./voc_extract.py', '-m', 'synth', '-o', '--no_batch',
                         '-F', wav_tkn, '-v', voc_dir, '-w', wav_dir,
                         '-r', str(sampling_rate)])
    subprocess.check_output(cmd_line, shell=True)
    return sf.read(wav_file)[0]


def synth_gt_wavs(i):
    wav_tkn = 'ground_truth_{:05d}'.format(i)
    if not glob(os.path.join(wav_dir, wav_tkn + '.wav')):
        gt_vocoder = voc_cat[voc_uid[i]:voc_uid[i+1]]
        gt_magphase = gt_vocoder[:, list(range(80))+[81]] * voc_std + voc_mean
        assert ((gt_vocoder[:, -2] == 0) + (gt_vocoder[:, -2] == 1)).all(), \
            'wrong dimension for voicedness '
        gt_magphase[:, -1][(1 - gt_vocoder[:, -2]).byte()] = -1.0e+10
        gt_split = torch.split(gt_magphase, [60, 10, 10, 1], dim=1)
        for gt, ft in zip(gt_split, ['mag', 'real', 'imag', 'lf0']):
            write_binfile(gt, os.path.join(voc_dir, '.'.join([wav_tkn, ft])))
        cmd_line = ' '.join(['./voc_extract.py', '-m', 'synth',
                             '-F', wav_tkn, '-v', voc_dir, '-w', wav_dir,
                             '-r', str(sampling_rate), '--no_batch'])
        subprocess.check_output(cmd_line, shell=True)
    # print(os.path.join(wav_dir, wav_tkn + '.wav'))
    return sf.read(os.path.join(wav_dir, wav_tkn + '.wav'))[0]


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
        # self.in_mask = get_mask_from_lengths(transposed_data[2])
        # self.voc_mask = get_mask_from_lengths(transposed_data[3])
        inp_places = var(torch.arange(self.char.shape[1]).unsqueeze(0))
        self.in_mask = inp_places < torch.stack(transposed_data[2]).unsqueeze(1)
        voc_places = var(torch.arange(self.voc.shape[1]).unsqueeze(0))
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
        voc_dic = {k: var(torch.from_numpy(np.array(v))) for k, v in f.items()}
    global voc_cat, voc_uid, voc_mean, voc_std
    voc_cat, voc_uid, voc_mean, voc_std = itemgetter(
        'voc_scaled_cat', 'voc_utt_idx', 'voc_mean', 'voc_std')(voc_dic)
    lens_voc = voc_uid[1:] - voc_uid[:-1]
    sampling_rate = int(voc_dic.get('sampling_rate', 48000))
    voc_case = torch.split(voc_cat, torch.unbind(lens_voc), dim=0)
    print("vocoder sequence loaded")

    # load character array data
    with h5py.File(char_path, 'r') as f:
        char_dic = {k: np.array(v) for k, v in f.items()}
        char_ls = itemgetter('bow_id', 'eow_id', 'cat_encoded_seq',
                             'cat_is_upper', 'utt_id')(char_dic)
    n_type = char_dic['char_types'].shape[0]
    print('# types of symbols used: ', n_type)
    bow_id, eow_id, cat_char_seq, cat_upper_seq, utt_id = \
        [var(torch.from_numpy(_t.astype(int))) for _t in char_ls]
    assert all(cat_char_seq != 0), np.where(cat_char_seq == 0)
    lens_seq = (utt_id[1:] - utt_id[:-1])
    char_case, up_case = map(torch.split, [cat_char_seq, cat_upper_seq],
                             repeat(torch.unbind(lens_seq)))
    used_size = len(voc_case) if args['frac'] is None else args['frac']
    all_data = VariLenDataset(char_case[:used_size], voc_case[:used_size],
                              lens_seq[:used_size], lens_voc[:used_size],
                              up_case[:used_size])
    print('data loader ready')

    # split train, val set; and group batches by lengths
    train_size = np.floor(used_size * (1 - val_rate)).astype(int)
    train_set, val_set = bipart_dataset(all_data, train_size)
    train_samp, val_samp = map(LenGroupedSampler,
                               [lens_voc[:train_size], lens_voc[train_size:used_size]])
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_samp, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            sampler=val_samp, collate_fn=collate_wrapper)
    print('data loaded. ')

    # synthesize ground truth wavs
    test_ids = torch.arange(int(train_size), int(train_size) + test_size)
    # torch.randint(int(train_size), used_size, (test_size, ))
    test_set = Subset(all_data, test_ids)
    test_loader = DataLoader(test_set, collate_fn=collate_wrapper)
    tb = SummaryWriter(log_dir=tb_dir)
    for _i in test_ids:
        if args['no_voice']:
            break
        tb.add_audio('wav{:05d}/ground_truth'.format(_i), synth_gt_wavs(_i),
                     sample_rate=sampling_rate, global_step=0)

    # load model
    char2voc = Char2Voc(n_type, embedding_size, embedding_size, upper_in=True,
                        dropout=args['dropout']).to(device)
    # const_shift = (lens_seq.float() / lens_voc.float()).mean()
    # print("constant shift in attention set to: ", const_shift)
    # char2voc.attention.const_shift = const_shift
    pytorch_total_params = sum(p.numel() for p in char2voc.parameters())
    print('total number of parameters (including non-trainbale): ',
          pytorch_total_params)
    if args['init']:
        char2voc.weight_init()
    loss_criterion = MagPhaseLoss()  # (loss_type=nn.functional.mse_loss)
    optimizer = optim.Adam(char2voc.parameters(), lr=learning_rate)

    tf = args['tf_rate']
    id_loss = 0
    for _e in range(1, epochs + 1):
        if args['scheduled_sampling']:
            tf = min(args['tf_rate'], (1 - _e / epochs))
        char2voc.train()
        losses = []
        for d in tqdm(train_loader):
            y = char2voc(d.char, d.voc, input_mask=d.in_mask,
                         tf_rate=tf, upper_case=d.upper)[0]
            loss = loss_criterion(y, d.voc, d.voc_mask)
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
            with torch.no_grad():
                y = char2voc(d.char, d.voc, input_mask=d.in_mask,
                             upper_case=d.upper, tf_rate=0)[0]
            loss = loss_criterion(y, d.voc, d.voc_mask)
            dev_loss.append(loss.item())
        tb.add_scalar('loss/dev', np.mean(dev_loss), id_loss)
        print('Dev Loss: %.3f' % (np.mean(dev_loss)))

        for _i, d in zip(test_ids.unbind(), test_loader):
            if args['no_voice']:
                break
            with torch.no_grad():
                y, attn = char2voc(d.char, d.voc, upper_case=d.upper,
                                   input_mask=d.in_mask, tf_rate=0)
            tb.add_audio('wav{:05}/model'.format(_i),
                         synth_scaled(y.squeeze(0), _i),
                         global_step=id_loss,
                         sample_rate=sampling_rate)
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(attn.squeeze().cpu(), aspect='auto')
            tb.add_figure('attn{:05}'.format(_i),
                          fig,
                          global_step=id_loss)

    torch.save(char2voc.state_dict(), model_path)
