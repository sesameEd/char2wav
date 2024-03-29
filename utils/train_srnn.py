#!/usr/bin/python3
import argparse
import h5py
import math
import numpy as np
import os
from operator import itemgetter
import torch
# import time
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import MagPhaseLoss, write_binfile
from model import init_by_class, shape_assert, SampleRNN
from tqdm import tqdm
from glob import glob
import soundfile as sf
import subprocess

parser = argparse.ArgumentParser('script to train sample rnn with different hyper-parameter settings')
parser.add_argument('--data', type=str, default='data/vocoder/all_vocoder.hdf5', help="default='data/vocoder/all_vocoder.hdf5'")
parser.add_argument('-e', '--epochs', type=int, default=10, help='default=10')
parser.add_argument('-B', '--batch_size', type=int, default=32, help='default=32')
parser.add_argument('--ts', '--truncate_size', type=int, dest='truncate_size', default=512, help='default=512')
parser.add_argument('--lr', '--learning_rate', dest='learning_rate', type=float, default=4e-4, help='default=4e-4')
parser.add_argument('-v', '--voc_synth_dir', type=str, default='data/synth_voc/', help='default=\'data/synth_voc/\'')
parser.add_argument('-w', '--wav_dir', type=str, default='data/wavs_syn')
parser.add_argument('-d', '--dropout', type=float, default=0.5, help='default=0.5')
parser.add_argument('-R', '--ratios', type=int, nargs='*', default=[2, 2, 8], help='default=[2, 2, 8]')
parser.add_argument('--ln', '--layer_norm', action='store_true', dest='layer_norm')
parser.add_argument('--ss', '--scheduled_sampling', action='store_true', dest='scheduled_sampling',
                    help='whether to use scheduled sampling for training, this will override tf_rate')
parser.add_argument('--tf', '--tf_rate', dest='tf_rate', type=float, default=1)
parser.add_argument('--res', '--res_net', action='store_true', dest='res')
parser.add_argument('-t', '--test_size', type=int, default=5)
parser.add_argument('--xi', '--no_init', dest='no_init', action='store_true')
args = vars(parser.parse_args())
print(args)
truncate_size = args['truncate_size']
batch_size = args['batch_size']
epochs = args['epochs']
learning_rate = args['learning_rate']
test_size = args['test_size']
wav_dir = args['wav_dir']
voc_dir = args['voc_synth_dir']
if not glob(voc_dir):
    os.mkdir(voc_dir)
data_path = args['data']
ratios = args['ratios']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_rate = .10
up_rate = np.prod(ratios)
print(device)
model_name = 'srnn_r{ratios}{ln}{res}{ss}_tf{tf}'.format(
    ratios='_'.join(map(str, args['ratios'])), ln=args['layer_norm'] * '_ln',
    res=args['res'] * '_res', ss=args['scheduled_sampling'] * '_ss',
    tf=args['tf_rate'])
print(model_name)
tb_dir = os.path.join('data/tensorboard', model_name)
model_path = os.path.join('data', model_name + '.torch')
for f in glob(os.path.join(tb_dir, '*')):
    os.remove(f)


def split_2d(tensor_2d, chunk_size, pad_val=0):
    """splits tensor_2d into equal-sized, padded sequences and deals with
    when last sequence shorter than split size"""
    res = torch.split(tensor_2d, chunk_size, dim=0)
    return nn.utils.rnn.pad_sequence(res, batch_first=True, padding_value=pad_val)

def bipart_dataset(complete_set, split_index):
    return (data.Subset(complete_set, range(split_index)),
            data.Subset(complete_set, range(split_index, len(complete_set))))

def load_stateful_batch(ds, data_size, batch_size):
    batches = StatefulSampler(data_size, batch_size)
    return batches, data.DataLoader(ds, batch_sampler=batches)

class StatefulSampler(data.Sampler):
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
        shape_assert(self.batch_id, (self.num_batch, batch_size))
        print("Split data into {0} x {1} batches".format(*self.batch_id.shape))

    def __iter__(self):
        return iter(self.batch_id)

    def __len__(self):
        return self.num_batch


def synth_model_wavs(model, i, vocoder_dic):
    voc_cat, voc_uid, voc_mean, voc_std = itemgetter(
        'voc_scaled_cat', 'voc_utt_idx', 'voc_mean', 'voc_std')(vocoder_dic)
    gt_vocoder = voc_cat[voc_uid[i]:voc_uid[i+1]]
    inp = gt_vocoder[::up_rate, :].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out_vocoder = model(
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


if __name__ == "__main__":
    # global voc_dic, voc_utt_idx, voc_mean, voc_std
    with h5py.File(data_path, 'r') as f:
        voc_dic = {k: torch.from_numpy(np.array(v)) for k, v in f.items()}
    voc_utt_idx = voc_dic['voc_utt_idx']
    # voc_mean, voc_std = voc_dic['voc_mean'], voc_dic['voc_std']
    sampling_rate = voc_dic.get('sampling_rate', 48000)
    test_ids = torch.randint(len(voc_utt_idx) - 1, (test_size,))

    input_data = voc_dic['voc_scaled_cat'][::up_rate, :]
    trunc_out = split_2d(voc_dic['voc_scaled_cat'], truncate_size)
    trunc_in = split_2d(input_data, truncate_size // int(up_rate))
    assert trunc_in.shape[0] == trunc_out.shape[0]
    all_data = data.TensorDataset(trunc_in, trunc_out)
    train_size = math.floor(len(all_data) * (1 - val_rate))
    train_set, val_set = bipart_dataset(all_data, train_size)
    train_bch, train_loader = load_stateful_batch(
        train_set, train_size, batch_size)
    val_bch, val_loader = load_stateful_batch(
        val_set, len(val_set), batch_size)

    rs, frame_size = ratios[:-1], ratios[-1]
    srnn = SampleRNN(
        vocoder_size = 82,
        ratios = rs,
        hid_up_size = 82,
        frame_size = frame_size,
        dropout = args['dropout'],
        do_layernorm = args['layer_norm'],
        do_res = args['res'],
        batch_size = batch_size)
    srnn.to(device)
    if not args['no_init']:
        init_by_class[srnn.__class__](srnn)
    srnn.init_states(batch_size = batch_size)
    loss_criterion = MagPhaseLoss(batch_size=batch_size)

    optimizer = optim.Adam(srnn.parameters(), lr=learning_rate)
    tb = SummaryWriter(log_dir=tb_dir)
    for _i in test_ids:
        tb.add_audio('wav{:05d}/ground_truth'.format(_i), synth_gt_wavs(_i, voc_dic),
                     sample_rate=sampling_rate,
                     global_step=0)
    id_loss = 0
    teacher_forcing = args['tf_rate']
    for _e in range(1, epochs+1):
        srnn.init_states(batch_size = batch_size)
        losses = []
        # start = time.time()
        if args['scheduled_sampling']:
            teacher_forcing = 1 - _e / epochs
        for x, tar in tqdm(train_loader):
            x, tar = x.to(device), tar.to(device)
            optimizer.zero_grad()
            y = srnn(
                x_in = tar.transpose(0, 1),
                hid_up = x.transpose(0, 1),
                tf_rate = teacher_forcing)[1].transpose(0, 1)
            loss = loss_criterion(y, tar)
            loss.backward()
            nn.utils.clip_grad_norm_(srnn.parameters(), 5.)
            optimizer.step()
            tb.add_scalar('loss/train', loss, id_loss)
            id_loss += 1
            losses.append(loss.item())
            srnn.hid_detach()
        print('Epoch: %d Training Loss: %.3f; ' % (_e, np.mean(losses)), end='| ')

        dev_nll = []
        srnn.eval()
        with torch.no_grad():
            for x, tar in val_loader:
                x, tar = x.to(device), tar.to(device)
                y = srnn(tar.transpose(0, 1), x.transpose(0, 1))[1].transpose(0, 1)
                loss = loss_criterion(y, tar)
                dev_nll.append(loss.item())
        print('Dev Loss: %.3f' % (np.mean(dev_nll)))
        tb.add_scalar('loss/dev', np.mean(dev_nll), id_loss)

        srnn.init_states(batch_size=1)
        for _i in test_ids:
            tb.add_audio('wav{:05}/model'.format(_i),
                         synth_model_wavs(srnn, _i, voc_dic),
                         global_step=id_loss,
                         sample_rate=sampling_rate)

    torch.save(srnn.state_dict(), model_path)
