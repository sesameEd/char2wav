#!/usr/bin/python3
from model import SampleRNN, init_weights
import torch
import time
import torch.optim as optim
from torch.utils.data import Sampler, DataLoader
import torch.utils.data as data
import torch.nn as nn
import argparse
import h5py
import numpy as np
import math

parser = argparse.ArgumentParser(
    "-s | --samplernn, train sample_rnn"
    )
parser.add_argument('-s', '--sample_rnn', dest='train_srnn', action='store_true')
args = vars(parser.parse_args())

def split_by_size(tensor_2d, chunk_size, pad_val=0):
    """splits tensor_2d into equal-sized, padded sequences and deals with
    when last sequence shorter than split size"""
    num_full = tensor_2d.shape[0] // chunk_size
    res = tensor_2d[:num_full * chunk_size].view(num_full, chunk_size, -1)
    res = list(torch.unbind(res, 0))
    res.append(tensor_2d[num_full * chunk_size:])
    return nn.utils.rnn.pad_sequence(res, batch_first=True, padding_value=pad_val)

class StatefulSampler(Sampler):
    """Note that the actual batch size could be slightly smaller than given due to
    the residue being too small"""
    def __init__(self, num_seq, batch_size):
        self.num_seq = num_seq
        self.B = batch_size
        self.batch_id = split_by_size(torch.arange(self.num_seq).unsqueeze(1),
                                      math.ceil(self.num_seq / self.B)
                                     ).squeeze(2).transpose_(0, 1)
        print("Split data into {0} batches, each consisting {1} training cases".format(*self.batch_id.shape))

    def __iter__(self):
        return iter(self.batch_id)

    def __len__(self):
        return self.batch_id.shape[0]


if __name__ == "__main__":
    if args['train_srnn']:
        ratios = [1, 2, 2, 8]
        truncate_size = 512
        batch_size = 32
        val_rate = .10
        epochs = 10
        up_rate = np.prod(ratios)

        with h5py.File('data/vocoder/all_vocoder.hdf5', 'r') as f:
            voc_dic = {k: torch.from_numpy(np.array(v)) for k, v in f.items()}
        input_data = voc_dic['voc_scaled_cat'][::32, :]
        trunc_out = split_by_size(voc_dic['voc_scaled_cat'], truncate_size)
        trunc_in = split_by_size(input_data, truncate_size // up_rate)
        assert trunc_in.shape[0] == trunc_out.shape[0]
        all_data = data.TensorDataset(trunc_in, trunc_out)
        all_size = len(all_data)
        train_size = math.floor(all_size * (1 - val_rate))
        train_set, val_set = data.Subset(all_data, range(train_size)), data.Subset(all_data, range(train_size, all_size))
        train_bch, val_bch = StatefulSampler(train_size, batch_size), StatefulSampler(all_size - train_size, batch_size)
        train_loader, val_loader = DataLoader(train_set, batch_sampler=train_bch), DataLoader(val_set, batch_sampler=val_bch)
        try:
            rs, frame_size = ratios[:-1], ratios[-1]
            srnn = SampleRNN(
                vocoder_size = 81,
                ratios = rs,
                frame_size = frame_size,
                batch_size = batch_size
            )
        except NameError:
            srnn = SampleRNN(vocoder_size = 81, batch_size = batch_size)
        for name, param in srnn.named_parameters():
            init_weights(name, param)

        optimizer = optim.Adam(srnn.parameters(), lr=4e-4)
        loss_criterion = nn.L1Loss()
        for epoch in range(epochs):
            losses = []
            start = time.time()
            srnn.init_states()
            for i, case in enumerate(train_loader, 1):
                optimizer.zero_grad()
                x, tar = case
                print(i, end=' ', flush=True)
                y = srnn(x.transpose(0, 1))[1].transpose(0, 1)
                loss = loss_criterion(y, tar)
                loss.backward()
                nn.utils.clip_grad_norm_(srnn.parameters(), 5.)
                optimizer.step()
                losses.append(loss.item())
                srnn.hid_detach()
            elapsed = time.time() - start
            print('Epoch: %d; Training Loss: %.3f;' % (epoch, np.mean(losses)),
                  'took %.3f sec ' % (elapsed))

            dev_nll = []
            for i, case in enumerate(val_loader, 1):
                x, tar = case
                y = srnn(x.transpose(0, 1))[1].transpose(0, 1)
                loss = loss_criterion(y, tar)
                dev_nll.append(loss.item())
            print('Epoch : %d Dev Loss : %.3f' % (epoch, np.mean(dev_nll)))
            print('-------------------------------------------------------------')
