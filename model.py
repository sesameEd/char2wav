import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def shape_ast(t, shape):
    assert len(t.shape) == len(shape) and (x == y or y == -1 for x, y in zip(t.shape, shape)).all(), \
    'Expected shape ({}), got shape ({})'.format(', '.join(shape), ', '.join(t.shape))

class Attn(nn.modules.Module):
    def __init__(self, N_de, Len_seq, G_attn=3):
        self.GM = G_attn # number of GM components
        self.linear = nn.Linear(N_de + Len_seq, self.GM * 3)

    def forward(self, s_de, attn_pre, kp_pre):
        shape_ast(s_de, (-1, -1))
        _B, _H = s_de.shape
        shape_ast(attn_pre, (-1, -1))
        _B, _L = attn_pre.shape

        rho, beta, kappa = self.linear(torch.cat([s_de, attn_pre], dim=-1)).split(self.GM, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        kappa = (kp_pre + torch.exp(kappa)).unsqueeze(dim=-1)

        shape_ast(rho, (_B, self.GM, 1))
        phi = (rho * torch.exp(-beta * (kappa - torch.arange(_L).view(1, 1, _L)) ** 2)).sum(dim=1)
        shape_ast(phi, (_B, _L))
        return phi, kappa

    def kappa_init(self, batch):
        return torch.zeros(batch, self.GM)


class FrameLevelRNN(nn.modules.Module):
    def __init__(self, ratio, H_up, H_in, Hid, N_samp):
        self.R = ratio
        self.N_samp = N_samp
        self.H_in = H_in
        self.Hid = Hid
        self.add_module('W_x', nn.Linear(H_in, Hid))
        self.add_module('W_j', nn.Linear(H_up, ratio * H_in))
        self.add_module('recursive', nn.LSTMCell(N_samp + H_up, Hid))
        self.add_module('sample_gen', SampleGen())

    def forward(self, x, h_up, hid, cell):
        """
        the x has to be fed to itself, so call the lower layers and obtain predicted samples
        :param x: a group of N_samp samples predicted by previous frame node
        :param h_up: hidden representation of top tier
        :param hid: previous hidden state (from the previous node of same tier)
        :param cell: previous cell state
        :return: prediction of the next N_samp samples, hid, cell
        """
        shape_ast(x, (-1, self.N_samp))
        _B, _N = x.shape
        c_j = self.W_j(h_up).split(self.H_in, dim=-1)
        out = []
        for _k in range(self.R):
            # inp = torch.cat([x, h_up], dim=-1)
            inp = self.W_x(x) + c_j[_k]
            hid, cell = self.recursive(inp, hid, cell)
            x = self.sample_gen(x, hid)
        return out, hid, cell

    def hid_init(self, batch):
        return torch.zeros(batch, self.Hid)


class SampleGen(nn.modules.Module):
    def __init__(self, N_up, N_in, N_hid=256, N_samp=4, embd=0, bit_depth=8):
        self.N_s = N_samp
        self.N_h = N_hid
        self.N_i = N_in # dimension of input vector
        self.N_up = N_up # dimension of output hidden state from previous tier
        self.embd_on = embd # whether to use an embdding layer for inputs
        self.B_depth = bit_depth
        # self.R = ratio
        if not embd:
            self.add_module('W_x', nn.Linear(N_samp, N_in))
        else:
            self.add_module('W_x', N_samp * embd, N_in)
        self.add_module('W_j', nn.Linear(N_up, N_samp * N_in))
        self.add_module('i2h', nn.Linear(N_in, N_hid))
        # self.add_module('h_nonlin', nn.Tanh()) #fill in?????
        self.add_module('h2s', nn.Linear(N_hid, 2 ** bit_depth)) # or a 8-way sigmoid?
        self.add_module('log_softmax', nn.LogSoftmax(dim=-1))

    def forward(self, x_cont, h_up):
        shape_ast(h_up, (-1, self.N_up))
        _B, _N_up = h_up.shape
        c_j = self.W_j(h_up).split(self.N_i, dim=-1)
        if not self.embd_on:
            shape_ast(x_cont, (-1, self.N_s))
            x_cont = deque(x_cont.split(1, dim=-1), maxlen=self.N_s)
        else:
            shape_ast(x_cont, (-1, self.N_s, -1))
            x_cont = deque(x_cont.split(1, dim=1), maxlen=self.N_s)
        for _k in range(self.N_s):
            inp = F.tanh(self.W_x(torch.cat(x_cont, dim=-1)) + c_j[_k])
            out = F.log_softmax(self.h2s(inp))
            shape_ast(out, (_B, 2 ** self.B_depth))
            x_cont.append(torch.multinomial(torch.exp(out), 1).squeeze(dim=-1)) # squeeze?????
        return x_cont


class Encoder(nn.modules.Module):
    def __init__(self, N_cat, L_seq, h_en, h_de, N_voc):
        super(Encoder, self).__init__()
        self.Len = L_seq
        self.N_cat = N_cat
        self.N_voc = N_voc

        self.add_module('encoder', nn.LSTM(N_cat, h_en, bidirectional=True))
        self.add_module('attention', Attn(h_en + h_de, L_seq))
        self.add_module('decoder', nn.LSTMCell(h_en, h_de))
        self.add_module('in_vocoder', nn.Linear(h_de, N_voc))

    def forward(self, x, y_tar, m):
        shape_ast(x, (-1, self.Len, self.N_cat))
        _B, _L, _C = x.shape
        input = torch.zeros (x.shape)
        encoded = self.encoder(x) # of shape (_B, _L, h_en)

        # initialize h, alpha (attention weight), and kappa (Gaussian mean)
        h, c = self.decoder(encoded[:, -1, :])
        result = [h]
        alpha = torch.zeros(_B, _L)
        kappa = self.attention.kappa_init(_B)

        for t in range(_L): # until decoder stops generating
            alpha, kappa = self.attention(result[-1], alpha, kappa)
            h, c= self.decoder(alpha.unsqueeze(dim=-1) * encoded, h, c)
            result.append(h)
        result = torch.stack(result).transpose(0, 1)
        return result

