#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)


def shape_assert(t, shape):
    assert len(t.shape) == len(shape) and all(
        x == y or y == -1 for x, y in zip(t.shape, shape)), \
        'Expected shape ({}), got shape ({})'.format(
        ', '.join(map(str, shape)), ', '.join(map(str, t.shape)))


def write_binfile(m_data, filename):
    m_data = np.array(m_data.cpu(), 'float32')  # Ensuring float32 output
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return


def sum_loss(loss3d):
    return loss3d.sum(-1).sum(1).mean()


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(max_len)  # .unsqueeze(0)
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


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


class LinearNorm(torch.nn.Module):
    """copied from https://github.com/NVIDIA/DeepLearningExamples.git"""
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """copied from https://github.com/NVIDIA/DeepLearningExamples.git"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear',
                 groups=1):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias, groups=groups)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class MagPhaseLoss(nn.Module):
    def __init__(self, vocoder_size=82, dim_lf0=-1, dim_vuv=-2,
                 loss_type=F.l1_loss):
        """loss_type must be a contiguous loss (instead of a categorical one),
                     must be of nn.functional class (instead of nn.Module) """
        super(MagPhaseLoss, self).__init__()
        self.V = vocoder_size
        self.dim_lf0 = dim_lf0
        self.dim_vuv = dim_vuv  # dimension of voiced/unvoiced bool
        self.loss_type = loss_type

    def forward(self, y, target, loss_mask=None):
        shape_assert(y, (-1, -1, self.V))
        _B, _T, _V = y.shape
        shape_assert(target, (_B, _T, _V))
        y_mp, y_vuv, y_lf = y.split((80, 1, 1), dim=-1)
        tar_mp, tar_vuv, tar_lf = target.split((80, 1, 1), dim=-1)
        if loss_mask is None:
            loss_mask = torch.ones(target.shape[:-1]).unsqueeze(-1)
        loss_mask = loss_mask.float().unsqueeze(-1)
        loss_mp = self.loss_type(y_mp, tar_mp, reduction='none') * loss_mask
        loss_v = F.binary_cross_entropy(y_vuv, tar_vuv, reduction='none') * loss_mask
        loss_l = self.loss_type(y_lf, tar_lf, reduction='none') * tar_vuv * loss_mask
        loss_mp, loss_v, loss_l = map(sum_loss, (loss_mp, loss_v, loss_l))
        return loss_mp + loss_v + loss_l


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


if __name__ == "__main__":
    # test shape_assert dunction
    new = torch.rand(2, 2, 4)
    shape_assert(new, (2, 2, 4))

    # test MagPhaseLoss
    random_voc = torch.zeros(20, 300, 82)
    random_voc.normal_()
    random_voc[:, :, -2].bernoulli_()
    random_tar = torch.zeros(20, 300, 82)
    random_tar.normal_()
    random_tar[:, :, -2].uniform_()
    loss_crit = MagPhaseLoss(20)
    loss = loss_crit(random_voc, random_tar).item()
    print('random magphase loss is ', loss)
    print('MagPhaseLoss passed test without error')