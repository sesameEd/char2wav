#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MagPhaseLoss(nn.Module):
    def __init__(self, batch_size, vocoder_size=82, dim_lf0=-1, dim_vuv=-2,
                 loss_type=F.l1_loss):
        """loss_type must be a contiguous loss (instead of a categorical one),
                     must be of nn.functional class (instead of nn.Module) """
        super(MagPhaseLoss, self).__init__()
        self.B = batch_size
        self.V = vocoder_size
        self.dim_lf0 = dim_lf0
        self.dim_vuv = dim_vuv  # dimension of voiced/unvoiced bool
        self.loss_type = loss_type

    def forward(self, y, target, loss_mask=None):
        shape_assert(y, (self.B, -1, self.V))
        _B, _T, _V = y.shape
        shape_assert(target, (_B, _T, _V))
        y_mp, y_vuv, y_lf = y.split((80, 1, 1), dim=-1)
        tar_mp, tar_vuv, tar_lf = target.split((80, 1, 1), dim=-1)
        # if loss_mask is None:
        #     loss_mask = torch.ones(target.shape)
        loss_mp = self.loss_type(y_mp, tar_mp, reduction='none').sum(-1).sum(1).mean()
        loss_v = F.binary_cross_entropy(y_vuv, tar_vuv, reduction='none').sum(-1).sum(1).mean()
        loss_l = (self.loss_type(y_lf, tar_lf, reduction='none') * tar_vuv).sum(-1).sum(1).mean()
        print(loss_mp.item(), loss_v.item(), loss_l.item())
        return loss_mp + loss_v + loss_l


if __name__ == "__main__":
    new = torch.rand(2, 2, 4)
    shape_assert(new, (2, 2, 4))
