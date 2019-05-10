#!/usr/bin/python3
from model import init_by_class, shape_ast
import torch
import time
import torch.optim as optim
from torch.utils.data import Sampler, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import argparse
import h5py
import numpy as np
import math
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(
    "-s | --sample_rnn, train sample_rnn"
    )
parser.add_argument('-s', '--sample_rnn', dest='train_srnn', action='store_true')
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--truncate_size', type=int, default=512)
parser.add_argument('--lr', dest='learning_rate', type=float, default=4e-4)
parser.add_argument('--voc_synth_dir', type=str, default='data/synth_voc/')
args = vars(parser.parse_args())

truncate_size = args['truncate_size']
batch_size = args['batch_size']
epochs = args['epochs']
learning_rate = args['learning_rate']
test_size = 5
synth_dir = args['voc_synth_dir']

class MagPhaseLoss(nn.Module):
    def __init__(self, batch_size, vocoder_size=82, dim_lf0=-1, dim_uvu=-2,
                 loss_type=F.l1_loss):
        """loss_type must be a contiguous loss (instead of a categorical one),
                     must be of nn.functional class (instead of nn.Module) """
        super(MagPhaseLoss, self).__init__()
        self.B = batch_size
        self.V = vocoder_size
        self.dim_lf0 = dim_lf0
        self.dim_uvu = dim_uvu # dimension of voiced/unvoiced bool
        self.loss_type = loss_type

    def forward(self, input, target):
        shape_ast(input, (self.B, -1, self.V))
        shape_ast(target, (self.B, -1, self.V))
        losses = self.loss_type(input, target, reduction='none').transpose(0, -1)
        if self.dim_lf0 != -1 and self.dim_lf0 != self.V-1:
            assert self.dim_uvu != -1 and self.dim_uvu != self.V-1
            losses[[self.dim_lf0, -1]] = losses[[-1, self.dim_lf0]]
        uvu = target.transpose(0, -1)[self.dim_uvu]
        assert ((uvu == 0) + (uvu == 1)).all(), (uvu.shape, uvu[0])
        loss_lf = torch.masked_select(losses[-1], uvu.byte())
        loss_rest = losses[:-1].flatten()
        return torch.cat((loss_rest, loss_lf)).mean()

def split_by_size(tensor_2d, chunk_size, pad_val=0):
    """splits tensor_2d into equal-sized, padded sequences and deals with
    when last sequence shorter than split size"""
    num_full = tensor_2d.shape[0] // chunk_size
    res = tensor_2d[:num_full * chunk_size].view(num_full, chunk_size, -1)
    res = list(torch.unbind(res, 0))
    res.append(tensor_2d[num_full * chunk_size:])
    return nn.utils.rnn.pad_sequence(res, batch_first=True, padding_value=pad_val)

def bipart_dataset(complete_set, split_index):
    return (data.Subset(complete_set, range(split_index)),
            data.Subset(complete_set, range(split_index, len(complete_set))))

def load_stateful_batch(ds, data_size, batch_size):
    batches = StatefulSampler(data_size, batch_size)
    return batches, DataLoader(ds, batch_sampler=batches)

class StatefulSampler(Sampler):
    """Note that the actual batch size could be slightly smaller than given due to
    the residue being too small"""
    def __init__(self, num_seq, batch_size, padding_val=0):
        self.B = batch_size
        self.num_seq = num_seq
        self.num_batch = math.ceil(self.num_seq / self.B)
        _a = torch.arange(num_seq)
        batches = [_a[i::self.num_batch] for i in range(self.num_batch)]
        self.padded_batch = nn.utils.rnn.pad_packed_sequence(
            nn.utils.rnn.pack_sequence(batches),
            batch_first=True,
            total_length=batch_size,
            padding_value=padding_val
        )
        self.batch_id = self.padded_batch[0]
        assert self.batch_id.shape == (self.num_batch, batch_size)
        print("Split data into {0} x {1} batches".format(*self.batch_id.shape))

    def __iter__(self):
        return iter(self.batch_id)

    def __len__(self):
        return self.num_batch

def write_binfile(m_data, filename):
    m_data = np.array(m_data, 'float32') # Ensuring float32 output
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return

if __name__ == "__main__":
    if args['train_srnn']:
        from model import SampleRNN
        ratios = [1, 2, 2, 8]
        val_rate = .10
        up_rate = np.prod(ratios)

        with h5py.File('data/vocoder/all_vocoder.hdf5', 'r') as f:
            voc_dic = {k: torch.from_numpy(np.array(v)) for k, v in f.items()}
        input_data = voc_dic['voc_scaled_cat'][::up_rate, :]
        trunc_out = split_by_size(voc_dic['voc_scaled_cat'], truncate_size)
        trunc_in = split_by_size(input_data, truncate_size // up_rate)
        assert trunc_in.shape[0] == trunc_out.shape[0]
        all_data = data.TensorDataset(trunc_in, trunc_out)
        all_size = len(all_data)
        train_size = math.floor(all_size * (1 - val_rate))
        train_set, val_set = bipart_dataset(all_data, train_size)
        train_bch, train_loader = load_stateful_batch(
            train_set, train_size, batch_size)
        val_bch, val_loader = load_stateful_batch(
            val_set, len(val_set), batch_size)
        try:
            rs, frame_size = ratios[:-1], ratios[-1]
            srnn = SampleRNN(
                vocoder_size = 82,
                ratios = rs,
                frame_size = frame_size,
                batch_size = batch_size)
        except NameError:
            srnn = SampleRNN(vocoder_size = 82, batch_size = batch_size)
        init_by_class[srnn.__class__](srnn)

        optimizer = optim.Adam(srnn.parameters(), lr=learning_rate)
        loss_criterion = MagPhaseLoss(batch_size=batch_size)

        srnn.init_states()
        dev_loss = []
        print('Calculating initial loss on validation set...')
        for x, tar in val_loader:
            y = srnn(x.transpose(0, 1))[1].transpose(0, 1)
            loss = loss_criterion(y, tar)
            dev_loss.append(loss.item())
        print('Dev loss before training: %.3f' % (np.mean(dev_loss)))
        print('-------------------------------------------------------------')

        for epoch in range(1, epochs+1):
            losses = []
            start = time.time()
            srnn.init_states()
            # for i, case in enumerate(train_loader, 1):
            for i, case in tqdm(enumerate(train_loader, 1)):
                x, tar = case
                y = srnn(x.transpose(0, 1))[1].transpose(0, 1)
                loss = loss_criterion(y, tar)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(srnn.parameters(), 5.)
                optimizer.step()
                losses.append(loss.item())
                srnn.hid_detach()
            elapsed = time.time() - start
            print('Epoch: %d Training Loss: %.3f;' % (epoch, np.mean(losses)), end='| ')
                  # 'took %.3f sec ' % (elapsed))

            dev_nll = []
            for i, case in enumerate(val_loader, 1):
                x, tar = case
                y = srnn(x.transpose(0, 1))[1].transpose(0, 1)
                loss = loss_criterion(y, tar)
                dev_nll.append(loss.item())
            print('Epoch: %d Dev Loss: %.3f' % (epoch, np.mean(dev_nll)))
            print('-------------------------------------------------------------')

        voc_utt_idx = voc_dic['voc_utt_idx']
        voc_mean, voc_std = voc_dic['voc_mean'], voc_dic['voc_std']
        test_id = torch.randint(len(voc_utt_idx) - 1, (test_size,))
        srnn.B = 1
        from glob import glob
        if not glob(synth_dir):
            os.mkdir(synth_dir)
        for i in test_id:
            srnn.init_states()
            gt_vocoder = voc_dic['voc_scaled_cat'][voc_utt_idx[i]:voc_utt_idx[i+1]]
            gt_magphase = gt_vocoder[:, list(range(80))+[81]] * voc_std + voc_mean
            assert ((gt_vocoder[:, -2] == 0) + (gt_vocoder[:, -2] == 1)).all(), \
                    'wrong dimension for voicedness '
            gt_magphase[:, -1][(1 - gt_vocoder[:, -2]).byte()] = -1.0e+10
            gt_split = torch.split(gt_magphase, [60, 10, 10, 1], dim=1)
            inp = gt_vocoder[::up_rate, :].unsqueeze(0)
            with torch.no_grad():
                out_vocoder = srnn(inp.transpose(0, 1))[1].transpose(0, 1).squeeze()
            out_voiced = torch.bernoulli(out_vocoder[:, -2])
            out_magphase = out_vocoder[:, list(range(80))+[81]] * voc_std + voc_mean
            out_magphase[:, -1][(1 - out_voiced).byte()] = -1.0e+10
            out_split = torch.split(out_magphase, [60, 10, 10, 1], dim=1)
            for gt, out, ft in zip(gt_split, out_split, ['mag', 'real', 'imag', 'lf0']):
                write_binfile(gt, os.path.join(synth_dir,
                                               'ground_truth_{:05d}.{}'.format(i, ft)))
                write_binfile(out, os.path.join(synth_dir,
                                                'srnn_{:05d}.{}'.format(i, ft)))
        print('synthesizing voices from generated vocoder features.')
        os.system('./voc_extract.py -m  synth -v data/synth_voc/ -w data/wavs_syn/')
