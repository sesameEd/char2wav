#!/usr/bin/python3
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
        self.Len = kwargs['len_seq']
        self.decoded_size = kwargs['decoded_size']
        self.ha2gmm = nn.Linear(self.decoded_size + self.Len, self.N_gm * 3)

    def forward(self, prev_decoded, prev_attn):
        shape_ast(prev_decoded, (-1, self.decoded_size))
        _B, _H = prev_decoded.shape
        shape_ast(prev_attn, (_B, self.Len))
        _B, _T = prev_attn.shape

        rho, beta, kappa = self.ha2gmm(torch.cat([prev_decoded, prev_attn], dim=-1)).split(self.N_gm, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        self.kappa = (self.kappa + torch.exp(kappa)).unsqueeze(dim=-1)
        # what about replacing torch.exp() with ReLU???

        shape_ast(rho, (_B, self.N_gm, 1))
        phi = (rho * torch.exp(-beta * (self.kappa - torch.arange(_T).view(1, 1, _T)) ** 2)).sum(dim=1)
        # shape_ast(phi, (_B, _T))
        return F.normalize(phi, p=1, dim=1)

    def kappa_init(self, batch):
        self.kappa = torch.zeros(batch, self.N_gm)
        # return torch.zeros(batch, self.N_gm)


class Char2Vocoder(nn.modules.Module):
    def __init__(self, **kwargs):
        """
        input / char2vec  -> encoder biLSTM(input)
                          -> attention recurrent(state_pre, alpha_pre)
                          -> decoded_1st recurrent(alpha * encoded, state_pre)
                          -> decoded2vocoder MLP(decoded), ReLU
        """
        super(Char2Vocoder, self).__init__()
        self.Len = kwargs['len_seq']
        self.in_size = kwargs['num_type']
        self.embd = kwargs.get('char_embd_size', None)
        if not (self.embd is None):
            self.in_size = self.embd
        encoded_size = kwargs.get('encoded_size', 1024)
        decoded_size = kwargs.get('decoded_size', 1024)
        mlp_sizes = kwargs.get('de2voc_mlp_sizes', [1024])
        vocoder_size = kwargs.get('vocoder_size', 512)
        mlp_i2o_sizes = list(zip([decoded_size] + mlp_sizes, mlp_sizes + [vocoder_size]))

        self.add_module('encoder', nn.LSTM(self.in_size, encoded_size, bidirectional=True))
        self.add_module('attention', Attn(num_component=3,
                                          decoded_size=decoded_size,
                                          len_seq=self.Len))
        self.add_module('decoder', nn.LSTMCell(encoded_size * 2, decoded_size))
        self.de2voc_mlp = nn.ModuleList([nn.Linear(*tp) for tp in mlp_i2o_sizes])

    def forward(self, x):
        """
        :x tensor of input, pre-trained char embedding or one-hot encoding
        :result of shape (_T, _B, vocoder_size),
        """
        shape_ast(x, (-1, self.Len, self.in_size))
        _B, _T, _C = x.shape
        # input = torch.zeros(x.shape)
        encoded = self.encoder(x) # of shape (_B, _T, 2 * encoded_size)

        hid, cell = self.decoder(encoded[:, -1, :])
        result = []
        alpha = torch.zeros(_B, _T)
        self.attention.kappa_init(_B)
        idx = ((_T-1) * torch.ones(_B, 1)).long()
        end_focus = torch.zeros(_B, _T).scattter(1, idx, 1.)

        while not alpha.allclose(end_focus):
            alpha = self.attention(hid, alpha)
            hid, cell= self.decoder(alpha.unsqueeze(dim=-1) * encoded, hid, cell)
            out = hid
            for fc in self.de2voc_mlp[:-1]:
                out = F.relu(fc(out))
            out = self.de2voc_mlp[-1](out)
            result.append(out)
        result = torch.stack(result).transpose(0, 1)
        return result


class FrameLevel(nn.modules.Module):
    def __init__(self, tier_low, parent2inp, **kwargs):
        super(FrameLevel, self).__init__()
        self.R = kwargs['ratio']
        self.N_sp = kwargs['sample_size']
        self.N_in = kwargs.get('input_size', 256)
        self.N_h = kwargs.get('hid_size', 1024)
        self.hid0 = nn.Parameter(torch.rand(1, self.N_h), requires_grad=True)
        self.add_module('w_x', nn.Linear(self.N_sp, self.N_in))
        self.add_module('w_j', parent2inp)
        self.add_module('recursive', nn.GRUCell(self.N_in, self.N_h))
        self.add_module('child', tier_low)

    def forward(self, x_in, hid_up):
        shape_ast(x_in, (-1, self.N_sp))
        _B, _S = x.shape
        shape_ast(hid_up, (_B, -1))
        c_j = self.w_j(hid_up).split(self.N_in, dim=-1)

        x_in = deque(x_in.split(_S / self.R, dim=-1), maxlen=self.R)
        logsm, x_out = [[]] * 2
        for _j in range(self.R):
            inp = self.w_x(x_in[-1]) + c_j[_j]
            self.hid = self.recursive(inp, self.hid)
            logsm_pre, x_pre = self.child(torch.cat(x_in[_j], dim=-1), self.hid)
            x_in.append(torch.stack(x_pre, dim=2))
            x_out += x_pre
            logsm += logsm_pre
        return logsm, x_out

    def hid_init(self, batch):
        self.hid = self.hid0.expand(batch, -1)
        self.child.hid_init(batch)


class SampleLevel(nn.modules.Module):
    def __init__(self, parent2inp, **kwargs):
        super(SampleLevel, self).__init__()
        self.N_sp = kwargs['frame_size']
        self.N_in = kwargs.get('input_size', 256) # dimension of input vector
        mlp_sizes = [self.N_in]+ kwargs.get('mlp_sizes', [1024,1024,256])
        embd_size = kwargs.get('audio_embd', 0)
        self.embd_on = True if embd_size > 0 else False # whether to use an embdding layer for inputs
        self.D_bt = kwargs.get('bit_depth', 8)
        # self.R = ratio
        if self.embd_on:
            self.add_module('embd', nn.Linear(2 ** self.D_bt, embd_size))
            self.add_module('w_x', nn.Linear(self.N_sp * embd_size, self.N_in))
        else:
            self.add_module('w_x', nn.Linear(self.N_sp, self.N_in))
        self.add_module('w_j', parent2inp)
        self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1]) for i in range(len(mlp_sizes)-1)])
        self.add_module('h2s', nn.Linear(mlp_sizes[-1], 2 ** self.D_bt)) # or a 8-way sigmoid?
        self.add_module('log_softmax', nn.LogSoftmax(dim=-1))

    def forward(self, x, hid_up):
        shape_ast(x, (-1, self.N_sp))
        _B, _S = x.shape
        shape_ast(hid_up, (_B, -1))
        x_out = deque(x.split(1, dim=1), maxlen=_S)
        c_j = self.w_j(hid_up).split(self.N_in, dim=-1)
        logsm = []
        if self.embd_on:
            x_in = deque([self.x2vec(x, _B) for _x in x_out], maxlen=_S)
        else:
            x_in = x_out
        for _j in range(_S):
            inp = self.w_x(torch.cat(x_in, dim=-1)) + c_j[_j]
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
    def __init__(self, **kwargs):
        super(SampleRNN, self).__init__()
        self.vocoder_size = kwargs['vocoder_size']
        self.ratios = kwargs.get('ratios', [1,2,2])
        self.input_sizes = kwargs.get('input_sizes', [256, 256, 256])
        self.gru_hid_sizes = kwargs.get('gru_hid_sizes', [1024, 1024, 1024])
        samp_gen_kw = {'frame_size' : kwargs.get('frame_size', 8),
                       'input_size' : kwargs.get('gen_in_size', 256),
                       'mlp_sizes' : kwargs.get('mlp_sizes', [1024, 1024, 256]),
                       'audio_embd' : kwargs.get('audio_embd', 0),
                       'bit_depth' : kwargs.get('bit_depth', 8)}
        if len(self.ratios) != len(self.input_sizes) or len(self.input_sizes) != len(self.gru_hid_sizes):
            print('Wrong size for the lists of params, {0} frame sizes,'.format(len(self.ratios)),
                  '{0} input sizes, and {1} recurrent sizes'.format(len(self.input_sizes), len(self.gru_hid_sizes)))
        self.n_tiers = len(self.ratios) + 1
        self.FS = self.ratios + [samp_gen_kw['frame_size']]
        self.samp_sizes = list(np.cumprod(self.FS[::-1]))[:0:-1]
        up_sizes = [vocoder_size] + self.gru_hid_sizes

        params = zip(self.ratios, self.input_sizes, self.gru_hid_sizes, self.samp_sizes)
        params = [dict(zip(['ratio', 'input_size', 'hid_size', 'sample_size'], tp)) for tp in params]
        wj_prm = zip(up_sizes, self.FS, self.input_sizes + samp_gen_kw['input_size']) # 0, .., self.n_tiers-1
        wj_prm = [(tp[0], tp[1]*tp[2]) for tp in wj_prm]
        tier = SampleGen(nn.Linear(*wj_prm[-1]), **samp_gen_kw)
        for k in reversed(range(self.n_tiers-1)):
            w_j = nn.Linear(*wj_prm[k])
            tier = FrameLevel(tier, w_j, **params[k])
        self.add_module('srnn', tier)

    def forward(self, vocoder):
        """
        :output log prob and prob, of shape (_T, _B, 2 ** depth)
        """
        shape_ast(vocoder, (-1, -1, self.vocoder_size))
        _T, _B, _V = vocoder.shape
        self.srnn.hid_init(_B)
        x_in = self.x_init(_B)
        logsm, x_out = [[]] * 2
        vocoder_single = vocoder.split(1, dim=0)
        for voc in vocoder_single:
            logsm_pre, x_pre = self.srnn(x_in, voc)
            x_in = torch.cat(x_pre, dim=-1)
            x_out += x_pre
            logsm += logsm_pre
        logsm = torch.stack(logsm, dim=0)
        x_out = torch.stack(x_out, dim=0)
        return logsm, x_out

    def x_init(self, batch):
        return torch.zeros(batch, self.samp_sizes[0])

class Char2Wav(nn.modules.Module):
    def __init__(self, **kwargs):
        super(Char2Wav, self).__init__()
        _L = kwargs['len_seq']
        _V = kwargs.get('vocoder_size', 512)
        self.char2voc = Char2Vocoder(
            len_seq = _L,
            vocoder_size = _V,
            num_type = kwargs['num_type'],
            char_embd_size = kwargs.get('char_embd_size', None),
            encoded_size = kwargs.get('encoded_size', 1024),
            decoded_size = kwargs.get('decoded_size', 1024),
            mlp_sizes = kwargs.get('de2voc_mlp_sizes', [1024])
            )
        self.samplernn = SampleRNN(
            vocoder_size = _V,
            ratios = kwargs.get('ratios', [1,2,2]),
            input_sizes = kwargs.get('input_sizes', [256, 256, 256]),
            gru_hid_sizes = kwargs.get('gru_hid_sizes', [1024, 1024, 1024]),
            frame_size = kwargs.get('frame_size', 8),
            input_size = kwargs.get('gen_in_size', 256),
            mlp_sizes = kwargs.get('mlp_sizes', [1024, 1024, 256]),
            audio_embd = kwargs.get('audio_embd', 0),
            bit_depth = kwargs.get('bit_depth', 8)
            )

    def forward(self, char_seq):
        # shape_ast(char_seq, )
        voc_pred = self.char2voc(char_seq)
        logsm, audio_out = self.samplernn(voc_pred)
        return logsm, audio_out
