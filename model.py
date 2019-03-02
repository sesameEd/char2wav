import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def shape_ast(t, shape):
    assert len(t.shape) == len(shape) and (x == y or y == -1 for x, y in zip(t.shape, shape)).all(), \
    'Expected shape ({}), got shape ({})'.format(', '.join(shape), ', '.join(t.shape))

class Attn(nn.modules.Module):
    def __init__(self, **kwargs):
        super(Attn, self).__init__()
        self.N_gm = kwargs.get('num_component', 3)
        # self.N_gm = num_component # number of GM components
        self.ha2gmm = nn.Linear(kwargs['decoded_size'] + kwargs['dim_attn'], self.N_gm * 3)

    def forward(self, prev_decoded, prev_attn):
        shape_ast(prev_decoded, (-1, -1))
        _B, _H = prev_decoded.shape
        shape_ast(prev_attn, (_B, -1))
        _B, _L = prev_attn.shape

        rho, beta, kappa = self.ha2gmm(torch.cat([prev_decoded, prev_attn], dim=-1)).split(self.N_gm, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        self.kappa = (self.kappa + torch.exp(kappa)).unsqueeze(dim=-1)
        # what about with ReLU???

        shape_ast(rho, (_B, self.N_gm, 1))
        phi = (rho * torch.exp(-beta * (self.kappa - torch.arange(_L).view(1, 1, _L)) ** 2)).sum(dim=1)
        # shape_ast(phi, (_B, _L))
        return phi

    def kappa_init(self, batch):
        self.kappa = zeros(batch, self.N_gm)
        # return torch.zeros(batch, self.N_gm)


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
        self.attention.kappa_init(_B)

        for t in range(_L): # until decoder stops generating
            alpha, kappa = self.attention(result[-1], alpha)
            h, c= self.decoder(alpha.unsqueeze(dim=-1) * encoded, h, c)
            result.append(h)
        result = torch.stack(result).transpose(0, 1)
        return result


class Parent2Inp(nn.modules.Module):
    def __init__(self, **kwargs):
        super(Parent2Inp, self).__init__()
        self.N_in = kwargs['in_size']
        self.h2c = nn.Linear(kwargs['parent_size'],
                             self.N_in * kwargs['frame_ratio'])

    def forward(self, hidden_up):
        return self.h2c(hidden_up).split(self.N_in, dim=-1)


class FrameLevel(nn.modules.Module):
    def __init__(self, tier_low, parent2inp, **kwargs):
        super(FrameLevel, self).__init__()
        self.R = kwargs['ratio']
        self.N_s = kwargs['sample_size']
        self.N_i = kwargs.get('input_size', 256)
        self.N_h = kwargs.get('hid_size', 1024)
        self.hid0 = nn.Parameter(torch.rand(1, self.N_h), requires_grad=True)
        self.add_module('W_x', nn.Linear(self.N_s, self.N_i))
        self.add_module('W_j', parent2inp)
        self.add_module('recursive', nn.GRUCell(self.N_i, self.N_h))
        self.add_module('child', tier_low)

    def forward(self, x_in, hid_up):
        shape_ast(x_in, (-1, self.N_s))
        _B, _N_s = x.shape
        shape_ast(hid_up, (_B, -1))
        c_j = self.W_J(hid_up).split(self.N_i, dim=-1)

        x_in = deque(x_in.split(_N_s / self.R, dim=-1), maxlen=self.R)
        logsm, x_out = [[]] * 2
        # c = self.W_j(self.hid)
        for _j in range(self.R):
            inp = self.W_x(x_in[-1]) + c_j[_j]
            self.hid = self.recursive(inp, self.hid)
            logsm_pre, x_pre = self.child(torch.cat(x_in[_j], dim=-1), self.hid)
            x_in.append(torch.cat(x_pre, dim=-1))
            x_out += x_pre
            logsm += logsm_pre
        return logsm, x_out

    def hid_init(self, batch):
        self.hid = self.hid0.expand(batch, -1)
        self.child.hid_init(batch)


class SampleLevel(nn.modules.Module):
    def __init__(self, parent2inp, **kwargs):
        super(SampleLevel, self).__init__()
        self.N_s = kwargs['frame_size']
        self.N_i = kwargs.get('in_size', 256) # dimension of input vector
        mlp_sizes = [self.N_i]+ kwargs.get('mlp_sizes', [1024,1024,256])
        embd = kwargs.get('audio_embd', 0)
        self.embd_on = True if embd else False # whether to use an embdding layer for inputs
        self.D_bt = kwargs.get('bit_depth', 8)
        # self.R = ratio
        if self.embd_on:
            self.add_module('embd', nn.Linear(2 ** self.D_bt, embd))
            self.add_module('W_x', nn.Linear(sample_size * embd, self.N_i))
        else:
            self.add_module('W_x', nn.Linear(sample_size, self.N_i))
        self.add_module('W_j', parent2inp)
        self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1]) for i in range(len(mlp_sizes)-1)])
        self.add_module('h2s', nn.Linear(mlp_sizes[-1], 2 ** self.D_bt)) # or a 8-way sigmoid?
        self.add_module('log_softmax', nn.LogSoftmax(dim=-1))

    def forward(self, x_in, hid_up):
        shape_ast(x_in, (-1, self.N_s))
        _B, _Ns = x_in.shape
        shape_ast(hid_up, (_B, -1))
        x_out = deque(x_in.split(1, dim=1), maxlen=_Ns)
        c_j = self.W_j(hid_up).split(self.N_i, dim=-1)
        logsm = []
        if self.embd_on:
            x_in = deque([self.x2vec(x, _B) for x in x_out], maxlen=_Ns)
        else:
            x_in = x_out
        for _j in range(_Ns):
            inp = self.W_x(torch.cat(x_in, dim=-1)) + c_j[_j]
            for fc in self.mlp:
                inp = F.relu(fc(inp))
            out = F.log_softmax(self.h2s(inp))
            logsm.append(out)
            x_pre = torch.multinomial(torch.exp(out), 1)
            x_out.append(x_pre)
            if self.embd:
                x_in.append(self.x2vec(x_pre))
            else:
                x_in.append(x_pre)
        return logsm, x_out

    def x2vec(self, x, batch):
        shape_ast(x, (batch, 1))
        one_hot = torch.zeros(batch, 2 ** self.D_bt).scatter_(1, x, 1.)
        return self.embd(one_hot)

    def hid_init(self):
        pass


class SampleRNN(nn.modules.Module):
    def __init__(self, decoded_size, **kwargs):
        super(SampleRNN, self).__init__()
        self.decoded_size = decoded_size
        self.ratios = kwargs.get('ratios', [1,2,2])
        self.input_sizes = kwargs.get('input_sizes', [256, 256, 256])
        self.gru_hid_sizes = kwargs.get('gru_hid_sizes', [1024, 1024, 1024])
        samp_gen_kw = {'frame_size' : kwargs.get('frame_size', 8),
                       'in_size' : kwargs.get('gen_in_size', 256),
                       'mlp_sizes' : kwargs.get('mlp_sizes', [1024, 1024, 256]),
                       'audio_embd' : kwargs.get('audio_embd', 0),
                       'bit_depth' : kwargs.get('bit_depth', 8)}
        if len(self.ratios) != len(self.input_sizes) or len(self.input_sizes) != len(self.gru_hid_sizes):
            print('Wrong size for the lists of params, {0} frame sizes,'.format(len(self.ratios)),
                  '{0} input sizes, and {1} recurrent sizes'.format(len(self.input_sizes), len(self.gru_hid_sizes)))
        self.n_tiers = len(self.ratios) + 1
        self.FS = self.ratios + [samp_gen_kw['frame_size']]
        self.samp_sizes = list(np.cumprod(self.FS[::-1]))[:0:-1]
        up_sizes = [decoded_size] + self.gru_hid_sizes

        params = zip(self.ratios, self.input_sizes, self.gru_hid_sizes, self.samp_sizes)
        params = [dict(zip(['ratio', 'input_size', 'hid_size', 'sample_size'], tp)) for tp in params]
        wj_prm = zip(up_sizes, self.FS, self.input_sizes + samp_gen_kw['in_size']) # 0, .., self.n_tiers-1
        wj_prm = [(tp[0], tp[1]*tp[2]) for tp in wj_prm]
        # wj_prm = [dict(zip(['parent_size', 'frame_ratio', 'in_size'], tp)) for tp in wj_prm]
        tier = SampleGen(nn.Linear(*wj_prm[-1]), **samp_gen_kw)
        for k in reversed(range(self.n_tiers-1)):
            # w_j = nn.Parent2Inp(**wj_prm[k])
            w_j = nn.Linear(*wj_prm[k])
            tier = FrameLevel(tier, w_j, **params[k])
        self.add_module('srnn', tier)

    def forward(self, decoded):
        # shape_ast(x, (-1, self.samp_sizes[0]))
        # _B, _Ns = x.shape
        shape_ast(decoded, (-1, -1, self.decoded_size))
        _B, _L, _N_de = decoded.shape
        self.srnn.hid_init(_B)
        # for frameRNN in self.framelevel:
        #     frameRNN.hid_init(_B)
        x_in = self.x_init(_B)
        logsm, x_out = [[]] * 2
        decoded_single = decoded.split(1, dim=1)
        # h_up = decoded
        for vec in decoded_single:
            logsm_pre, x_pre = self.srnn(x_in, vec)
            x_in = torch.cat(x_pre, dim=-1)
            x_out += x_pre
            logsm += logsm_pre
        logsm = torch.stack(logsm).transpose(0, 1)
        x_out = torch.stack(x_out).transpose(0, 1)
        return logsm, x_out

    def x_init(self, batch):
        return torch.zeros(batch, self.samp_sizes[0])

class Char2Wav(nn.modules.Module):
    def __init__(**kwargs):
