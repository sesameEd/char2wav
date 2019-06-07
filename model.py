#!/usr/bin/python3
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.nn.init as init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def shape_assert(t, shape):
    assert len(t.shape) == len(shape) and all(
        x == y or y == -1 for x, y in zip(t.shape, shape)), \
        'Expected shape ({}), got shape ({})'.format(
        ', '.join(map(str, shape)), ', '.join(map(str, t.shape)))

def to_onehot(idx, batch_shape, N_cat, padded=False):
    """if the sequences are padded, padding must be -1 (instead of 0);
    then a matrix with padded timestamp vectors filled with all-zero will be returned
    """
    if isinstance(batch_shape, int):
        batch_shape = [batch_shape]
    elif isinstance(batch_shape, tuple):
        batch_shape = list(batch_shape)
    if not isinstance(batch_shape, list):
        raise TypeError('batch_shape variable should be int, tuple or list')

    if idx.shape[-1] != 1:
        idx = idx.unsqueeze(-1)
    assert idx.shape == (*batch_shape, 1), \
        print('Expected index shape', (*batch_shape, 1), 'got', idx.shape)

    if padded:
        return torch.zeros(*batch_shape, N_cat+1).scatter_(
                    -1, idx + torch.ones(1, 1).long(), 1.
                ).transpose_(-1, 0)[1:].transpose_(-1, 0)
    else:
        return torch.zeros(*batch_shape, N_cat).scatter_(-1, idx, 1.)

def var(tensor):
    return tensor.to(device)

def init_weights(name, net):
    suffix = name.split('.')[-1]
    if suffix.split('_')[0] == 'bias':
        net.data.fill_(0.)
    elif suffix == 'weight_ih':
        for ih in net.chunk(3, 0):
            init.xavier_uniform_(ih)
    elif suffix == 'weight_hh':
        for hh in net.chunk(3, 0):
            init.orthogonal_(hh)
    elif suffix == 'weight':
        if 'ln' in name.split('.')[-2]:
            return
        init.xavier_normal_(net)
    elif suffix == 'hid0':
        init.normal_(net)
    else:
        print(name)

class Attn(nn.modules.Module):
    def __init__(self, **kwargs):
        super(Attn, self).__init__()
        self.N_gm = kwargs.get('num_component', 3)
        self.T = kwargs['len_seq']
        self.decoded_size = kwargs['decoded_size']
        self.ha2gmm = nn.Linear(self.decoded_size + self.T, self.N_gm * 3)

    def forward(self, prev_decoded, prev_attn):
        shape_assert(prev_decoded, (-1, self.decoded_size))
        _B, _H = prev_decoded.shape
        shape_assert(prev_attn, (_B, self.T))
        _B, _T = prev_attn.shape

        rho, beta, kappa = self.ha2gmm(torch.cat([prev_decoded, prev_attn], dim=-1)
                                      ).split(self.N_gm, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        self.kappa = F.softmax(self.kappa.squeeze() + F.relu(kappa), dim=-1).unsqueeze(dim=-1)
        # what about replacing torch.exp() with ReLU???

        shape_assert(rho, (_B, self.N_gm, 1))
        phi = (rho * torch.exp(
                     -beta * (self.kappa - torch.arange(float(_T)).view(1, 1, _T)) ** 2)
              ).sum(dim=1)
        return F.normalize(phi, p=1, dim=1)

    def kappa_init(self, batch):
        self.kappa = torch.zeros(batch, self.N_gm)


class Char2Voc(nn.modules.Module):
    def __init__(self, **kwargs):
        """
        input / char2vec  -> encoder biLSTM(input)
                          -> attention recurrent(state_pre, alpha_pre)
                          -> decoded_1st recurrent(alpha * encoded, state_pre)
                          -> decoded2vocoder MLP(decoded), ReLU
        """
        super(Char2Voc, self).__init__()
        self.do_maskedLM = kwargs.get('do_maskedLM', False)
        self.C = kwargs['num_type']
        self.T = kwargs['len_seq']
        self.E = kwargs.get('char_embd_size', None)
        self.I = self.E if self.E else self.C
        self.G = self.C if self.do_maskedLM else kwargs.get('vocoder_size', 82)
        self.padded = 2 if self.do_maskedLM else 1
        # generated samples' size
        encoded_size = kwargs.get('encoded_size', 512)
        decoded_size = kwargs.get('decoded_size', encoded_size)
        mlp_sizes = kwargs.get('de2voc_mlp_sizes', [1024])
        mlp_i2o_sizes = list(zip([decoded_size] + mlp_sizes, mlp_sizes + [self.G]))

        # self.embedding = nn.Embedding(self.C+1, self.I, padding_idx=-1)
        if self.E:
            self.embedding = nn.Sequential(nn.Linear(self.C, self.E),
                                           nn.LayerNorm([self.T, self.E]))
        self.add_module('encoder', nn.GRU(self.I, encoded_size, bidirectional=True))
        self.add_module('attention', Attn(num_component=3,
                                          decoded_size=decoded_size,
                                          len_seq=self.T))
        self.add_module('decoder', nn.LayerNormGRUCell(encoded_size * 2, decoded_size))
        self.de2voc_mlp = nn.ModuleList([nn.Linear(*tp) for tp in mlp_i2o_sizes])
        self.layrnorms = nn.ModuleList([nn.LayerNorm(io[1]) for io in mlp_i2o_sizes])

    def forward(self, x, ground_truth, is_upper=None, teacher_forcing=0):
        """
        :x  of shape (_B, _T) tensor of input, sequences of -1-padded indices
        :result of shape (_T, _B, _G),
        """
        shape_assert(x, (-1, self.T))
        _B, _T = x.shape
        x = to_onehot(x, batch_shape=(_B, _T), N_cat=self.C, padded=self.padded)
        if self.E:
            x = torch.sigmoid(self.embedding(x))
        encoded = self.encoder(x)[0] # of shape (_B, _T, 2 * encoded_size)
        hid = self.decoder(encoded[:, -1, :])
        result = []
        alpha = torch.zeros(_B, _T)
        self.attention.kappa_init(_B)
        # masked sequences and padded values??
        idx = ((_T-1) * torch.ones(_B, 1)).long()
        end_focus = torch.zeros(_B, _T).scatter_(-1, idx, 1.)

        while not alpha.allclose(end_focus) and len(result) < ground_truth.shape[-2]:
            alpha = self.attention(hid, alpha)
            inp = (alpha.unsqueeze(-1) * encoded).sum(1)
            hid = self.decoder(inp, hid)
            out = hid
            for fc, ln in zip(self.de2voc_mlp, self.layrnorms):
                out = out + ln(fc(out))
            if self.do_maskedLM:
                out = F.log_softmax(out, dim=-1)
            result.append(out)
        result = torch.stack(result)
        return result

class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.ln_zr_ih, self.ln_zr_hh = [nn.LayerNorm(2*hidden_size) for _ in range(2)]
        self.ln_hc_ih, self.ln_hc_hh = [nn.LayerNorm(hidden_size) for _ in range(2)]

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size,
                dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '[0]')

        wXx = F.linear(input, self.weight_ih, self.bias_ih).chunk(3, 1)
        wXh = F.linear(hx, self.weight_hh, self.bias_hh).chunk(3, 1)
        gates = self.ln_zr_ih(torch.cat(wXx[:2], dim=1)) + \
                self.ln_zr_hh(torch.cat(wXh[:2], dim=1))
        _z, _r = gates.sigmoid().chunk(2, 1)
        cand = (self.ln_hc_ih(wXx[2]) + _r * self.ln_hc_hh(wXh[2])).tanh()
        hy = (1 - _z) * hx + _z * cand
        return hy

class FrameLevel(nn.modules.Module):
    def __init__(self, tier_low, parent2inp=None, **kwargs):
        super(FrameLevel, self).__init__()
        self.R = kwargs['ratio']
        self.S = kwargs['sample_size']
        self.I = kwargs.get('input_size', 512)
        self.H = kwargs.get('hid_size', 512)
        self.G = kwargs.get('gen_size', 1)
        self.dropout = kwargs.get('drop_out', .3)
        self.do_layernorm = kwargs.get('do_layernorm', False)
        self.do_res = kwargs.get('do_res', False)
        self.hid0 = nn.Parameter(torch.rand(1, self.H), requires_grad=True)
        assert (self.S / self.R).is_integer(), print('frame size wrong')
        self.add_module('w_x', nn.Linear(int((self.S / self.R) * self.G), self.I))
        if not parent2inp is None:
            self.add_module('w_j', parent2inp)
        self.add_module('recursive', LayerNormGRUCell(self.I, self.H) if self.do_layernorm
                                     else nn.GRUCell(self.I, self.H))
        self.add_module('child', tier_low)

    def forward(self, x_in, hid_up, res_x=None):
        """
        x_in: list of self.S tensors, each of shape (_B, self.G)
        """
        shape_assert(x_in, (-1, self.S * self.G))
        _B, _SG = x_in.shape
        shape_assert(hid_up, (_B, -1))
        if (hid_up == 0).all():
            c_j = [torch.zeros(_B, self.I).to(device) for _ in range(self.R)]
        else:
            c_j = self.w_j(hid_up).split(self.I, dim=-1)
        x_in = deque(x_in.chunk(self.R, dim=-1), maxlen=self.R)
        # shape of each element: (B, S/R * G)
        logsm, x_out = [], []
        for _j in range(self.R):
            inp = self.w_x(x_in[-1]) + c_j[_j]
            self.hid = self.recursive(
                F.dropout(inp, self.dropout),
                F.dropout(self.hid, self.dropout)
            )
            if not self.do_res:
                logsm_pre, x_pre = self.child(x_in[-1], self.hid)
            else:
                if res_x is None: res_x = inp
                logsm_pre, x_pre = self.child(x_in[-1], self.hid + res_x, res_x)
            x_in.append(torch.cat(x_pre, dim=-1))
            x_out += x_pre
            logsm += logsm_pre
        return logsm, x_out

    def hid_init(self, batch):
        self.hid = var(self.hid0.expand(batch, -1))
        if isinstance(self.child, FrameLevel):
            self.child.hid_init(batch)

    def hid_detach(self):
        self.hid = var(self.hid.data)
        if isinstance(self.child, FrameLevel):
            self.child.hid_detach()


class SampleLevel(nn.modules.Module):
    def __init__(self, parent2inp, **kwargs):
        super(SampleLevel, self).__init__()
        self.S = kwargs.get('frame_size', 8)
        self.I = kwargs.get('gen_in_size', 256) # dimension of input vector
        self.D = kwargs.get('bit_depth', 8)
        self.do_res = kwargs.get('do_res', False)
        mlp_sizes = [self.I]+ kwargs.get('mlp_sizes', [1024,1024,256])
        embd_size = kwargs.get('audio_embd', 0)
        self.embd_on = True if embd_size > 0 else False # whether to use an embdding layer for inputs
        if self.embd_on:
            self.w_x = nn.Sequential(nn.Linear(2 ** self.D, embd_size),
                                     nn.Linear(self.S * embd_size, self.I))
            # self.add_module('embd', nn.Linear(2 ** self.D, embd_size))
            # self.add_module('w_x', nn.Linear(self.S * embd_size, self.I))
        else:
            self.w_x = nn.Linear(self.S, self.I)
        self.add_module('w_j', parent2inp)
        self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1]) for i in range(len(mlp_sizes)-1)])
        self.add_module('h2s', nn.Linear(mlp_sizes[-1], 2 ** self.D)) # or a 8-way sigmoid?

        self.do_layernorm = kwargs.get('do_layernorm', False)
        if self.do_layernorm:
            self.mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(mlp_sizes[i], mlp_sizes[i+1]),
                nn.LayerNorm(mlp_sizes[i+1])) for i in range(len(mlp_sizes)-1)])
        else:
            self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1])
                                      for i in range(len(mlp_sizes)-1)])

    def forward(self, x, hid_up):
        shape_assert(x, (-1, self.S))
        _B, _S = x.shape
        shape_assert(hid_up, (_B, -1))
        x_out = deque(x.split(1, dim=1), maxlen=_S)
        c_j = self.w_j(hid_up).split(self.I, dim=-1)
        logsm = []
        if self.embd_on:
            x_in = deque([self.x2vec(_x.long(), _B) for _x in x_out], maxlen=_S)
        else:
            x_in = x_out
        for _j in range(_S):
            inp = self.w_x(torch.cat(list(x_in), dim=-1)) + c_j[_j]
            for fc in self.mlp:
                inp = fc(F.relu(inp))
            out = F.log_softmax(self.h2s(inp), dim=-1)
            logsm.append(out)
            x_pre = torch.multinomial(torch.exp(out), 1)
            if self.embd_on:
                x_in.append(self.x2vec(x_pre, _B))
            else:
                x_in.append(x_pre)
            x_out.append(x_pre.float())
        return logsm, list(x_out)

    def x2vec(self, x, batch):
        shape_assert(x, (batch, 1))
        return self.embd(to_onehot(x, batch, 2 ** self.D))


# class VocoderLevel(nn.modules.Module):
#     def __init__(self, parent2inp, **kwargs):
#         super(VocoderLevel, self).__init__()
#         self.sample_size = kwargs.get('frame_size', 8)
#         self.I = kwargs.get('gen_in_size', 256) # dimension of input vector
#         self.vocoder_size = kwargs.get('vocoder_size', 81)
#         mlp_sizes = [self.I] + kwargs.get('mlp_sizes', [1024,512]) + [self.vocoder_size]
#         self.add_module('w_x', nn.Linear(self.sample_size * self.vocoder_size, self.I))
#         self.add_module('w_j', parent2inp)
#         self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1]) for i in range(len(mlp_sizes)-1)])
#         # self.add_module('conv1', nn.Conv1d(in_channels=1, out_channels=6, kernel_size=10, stride=5))
#
#     def forward(self, x, hid_up):
#         shape_assert(x, (-1, self.sample_size * self.vocoder_size))
#         _B, _S  = x.shape
#         shape_assert(hid_up, (_B, -1))
#         x_out = deque(x.split(self.vocoder_size, dim=1), maxlen=self.sample_size)
#         c_j = self.w_j(hid_up).split(self.I, dim=-1)
#         for _j in range(self.sample_size):
#             inp = self.w_x(torch.cat(list(x_out), dim=-1)) + c_j[_j]
#             for fc in self.mlp:
#                 inp = fc(F.relu(inp))
#             x_out.append(inp)
#         return [], list(x_out)
#
class SampleMagPhase(nn.modules.Module):
    def __init__(self, parent2inp, **kwargs):
        super(SampleMagPhase, self).__init__()
        self.sample_size = kwargs.get('frame_size', 8)
        self.I = kwargs.get('gen_in_size', 512) # dimension of input vector
        self.V = kwargs.get('vocoder_size', 82)
        mlp_sizes = [self.I] + kwargs.get('mlp_sizes', [512,512]) + [self.V]
        self.add_module('w_x', nn.Linear(self.sample_size * self.V, self.I))
        self.add_module('w_j', parent2inp)
        self.do_res = kwargs.get('do_res', False)
        self.do_layernorm = kwargs.get('do_layernorm', False)
        if self.do_layernorm:
            self.mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(mlp_sizes[i], mlp_sizes[i+1]),
                nn.LayerNorm(mlp_sizes[i+1])) for i in range(len(mlp_sizes)-1)])
        else:
            self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1])
                                      for i in range(len(mlp_sizes)-1)])

    def forward(self, x, hid_up, res_x=None):
        shape_assert(x, (-1, self.sample_size * self.V))
        _B, _S  = x.shape
        shape_assert(hid_up, (_B, -1))
        x_out = deque(x.split(self.V, dim=1), maxlen=self.sample_size)
        c_j = self.w_j(hid_up).split(self.I, dim=-1)
        for _j in range(self.sample_size):
            res = torch.sigmoid(self.w_x(torch.cat(list(x_out), dim=-1)) + c_j[_j])
            inp = res
            for _i, fc in enumerate(self.mlp, 1):
                if self.do_res and _i < len(self.mlp):
                    res = inp + fc(res)
                else:
                    res = fc(res)
            res[:, -2] = torch.sigmoid(res[:, -2]) # extra dim for voicedness prediction
            x_out.append(res)
        return [], list(x_out)


class SampleRNN(nn.modules.Module):
    def __init__(self, **kwargs):
        super(SampleRNN, self).__init__()
        # self.B = kwargs['batch_size']
        self.B = -1
        self.dropout = kwargs.get('drop_out', .3)
        self.ratios = kwargs.get('ratios', [2,2])
        self.input_sizes = kwargs.get('input_sizes', [512, 512])
        self.gru_hid_sizes = kwargs.get('gru_hid_sizes', [512, 512])
        self.gen_sample = kwargs.get('sample_level', False)
        # self.vocoder_size = kwargs['vocoder_size']
        # self.gen_size = 1 if self.gen_sample else self.vocoder_size
        self.gen_size = 1 if self.gen_sample else kwargs['vocoder_size']
        if (len(self.ratios) != len(self.input_sizes) or
            len(self.input_sizes) != len(self.gru_hid_sizes)):
            print('Wrong size for the lists of params, {0} frame sizes,'.format(len(self.ratios)),
                  '{0} input sizes, and {1} recurrent sizes'.format(
                      len(self.input_sizes), len(self.gru_hid_sizes)))
        self.n_tiers = len(self.ratios) + 1
        self.FS = self.ratios + [kwargs.get('frame_size', 8)]
        self.samp_sizes = list(np.cumprod(self.FS[::-1]))[:0:-1]
        self.hid_up_size = kwargs.get('hid_up_size', None)
        up_sizes = [self.hid_up_size] + self.gru_hid_sizes

        params = zip(self.ratios,
                     self.input_sizes,
                     self.gru_hid_sizes,
                     self.samp_sizes,
                     [self.gen_size] * len(self.ratios)) #,
                     # [self.dropout] * len(self.ratios),
                     # [kwargs.get('do_layernorm', False)] * len(self.ratios),
                     # [kwargs.get('do_res', False)] * len(self.ratios))
        params = [dict(zip(['ratio',
                            'input_size',
                            'hid_size',
                            'sample_size',
                            'gen_size',], tp)) for tp in params]
                            # 'drop_out',
                            # 'do_layernorm',
                            # 'do_res'], tp)) for tp in params]
        wj_prm = zip(up_sizes, self.FS, self.input_sizes + [kwargs.get('gen_in_size', 512)])
        wj_prm = [(tp[0], tp[1]*tp[2]) for tp in wj_prm]
        # 0, .., self.n_tiers-1
        if self.gen_sample:
            """default args for SampleLevel :
                frame_size  -> 8
                gen_in_size -> 256
                mlp_sizes   -> [1024, 1024, 256]
                audio_embd  -> 0
                bit_depth   -> 8"""
            tier = SampleLevel(nn.Linear(*wj_prm[-1]), **kwargs)
        else:
            """default args for VocoderLevel:
                frame_size  -> 8
                gen_in_size -> 512
                mlp_sizes   -> [512, 512]
                vocoder_size-> 81
                bit_depth   -> 8"""
            tier = SampleMagPhase(nn.Linear(*wj_prm[-1]), **kwargs)
        # assert len(wj_prm) == len(params), (wj_prm, params, self.n_tiers)
        for k in reversed(range(self.n_tiers-1)):
            if wj_prm[k][0] is None:
                tier = FrameLevel(tier, parent2inp=None, **params[k])
            else:
                tier = FrameLevel(tier, nn.Linear(*wj_prm[k]), **params[k], **kwargs)
        self.add_module('srnn', tier)

    def forward(self, x_in=None, hid_up=None, tf_rate=0):
        """
        :x_in of desired output shape (_T * pi(FS), _B, _V), it can be an all-zero tensor
            if tf_rate=0, but it's mandatory
        :hid_up of shape (_T, _B, _V), can be None
        :tf_rate, the rate of drawing
        :output
            if self.gen_sample (generates raw waves):
                log prob and prob, of shape (_T * pi(FS), _B, 2 ** depth)
            if not (generates vocoder features):
                [] and vocoders, of shape (_T * pi(FS), _B, _V)
        """
        shape_assert(x_in, (-1, self.B, self.gen_size))
        _, _B, _V = x_in.shape
        gen_steps = torch.split(x_in, int(self.samp_sizes[0]), dim=0)
        if hid_up is None:
            assert self.hid_up_size is None, 'expected hid_up, as hid_up_size was set! '
            hid_up = torch.zeros(len(gen_steps), 1)
        else:
            shape_assert(hid_up, (len(gen_steps), _B, self.hid_up_size))
        hu_unbind = torch.unbind(hid_up, dim=0)
        assert 0 <= tf_rate and 1 >= tf_rate, 'teacher forcing must be between 0 and 1!'
        logsm, x_out = [], []
        for hu, x in zip(hu_unbind, gen_steps):
            # mask = torch.bernoulli(tf_rate * torch.ones(self.samp_sizes[0]))
            mask = torch.bernoulli(tf_rate * torch.ones(x.shape[0]))
            self.x_in.view(_B, -1, _V).transpose(0, 1)[mask.nonzero().flatten()] = x[mask.byte()]
            logsm_pre, x_pre = self.srnn(self.x_in, hu)
            self.x_in = torch.cat(x_pre, dim=-1)
            x_out += x_pre
            logsm += logsm_pre
        x_out = torch.stack(x_out, dim=0)
        if self.gen_sample:
            logsm = torch.stack(logsm, dim=0)
        return logsm, x_out

    def init_states(self, batch_size):
        self.B = batch_size
        self.srnn.hid_init(self.B)
        # self.x_in = var(torch.zeros(self.B, self.samp_sizes[0] * self.gen_size))
        self.x_in = var(torch.zeros(self.B, self.samp_sizes[0] * self.gen_size))
        self.x_in.view(self.B, self.samp_sizes[0], self.gen_size)[:, :, :60].normal_(mean=-1.2, std=.5)

    def hid_detach(self):
        self.x_in = var(self.x_in.data)
        self.srnn.hid_detach()

class Char2Wav(nn.modules.Module):
    def __init__(self, **kwargs):
        super(Char2Wav, self).__init__()
        _L = kwargs['len_seq']
        _V = kwargs.get('vocoder_size', 81)
        self.gen_sample = kwargs.get('sample_level', False)
        self.char2v = Char2Voc(
            len_seq = _L,
            vocoder_size = _V,
            num_type = kwargs['num_type'],
            char_embd_size = kwargs.get('char_embd_size', None),
            encoded_size = kwargs.get('encoded_size', 1024),
            decoded_size = kwargs.get('decoded_size', 1024),
            mlp_sizes = kwargs.get('de2voc_mlp_sizes', [1024])
        )
        if self.gen_sample:
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
        else:
            self.samplernn = SampleRNN(
                vocoder_size = _V,
                ratios = kwargs.get('ratios', [1,2]),
                input_sizes = kwargs.get('input_sizes', [256, 256]),
                gru_hid_sizes = kwargs.get('gru_hid_sizes', [1024, 1024]),
                frame_size = kwargs.get('frame_size', 8),
                input_size = kwargs.get('gen_in_size', 256),
                mlp_sizes = kwargs.get('mlp_sizes', [1024, 256]),
            )

    def forward(self, char_seq):
        voc_pred = self.char2v(char_seq)
        logsm, audio_out = self.samplernn(voc_pred)
        return logsm, audio_out

def init_Linear(net):
    init.zeros_(net.bias)
    init.xavier_normal_(net.weight)

def init_recurrent(weight, num_chunks, initializer):
    for _w in weight.chunk(num_chunks, 0):
        initializer(_w)

def init_ModuleList(m_list):
    for layer in m_list:
        init_by_class[layer.__class__](layer)

def init_Module(net):
    for layer in net.children():
        init_by_class[layer.__class__](layer)

def init_GRU(net):
    init_recurrent(net.weight_ih, num_chunks=3,
                   initializer=init.xavier_uniform_)
    init_recurrent(net.weight_hh, num_chunks=3,
                   initializer=init.orthogonal_)

def init_LSTM(lstm):
    for n, p in lstm.named_parameters():
        if 'bias' in n:
            init.zeros_(p)
        if 'weight_ih' in n:
            init_recurrent(p, 4, init.xavier_uniform_)
        if 'weight_hh' in n:
            init_recurrent(p, 4, init.orthogonal_)

def init_FrameLevel(net):
    init.normal_(net.hid0)
    init_Module(net)

def init_LayerNormGRUCell(net):
    init_GRU(net)
    init_Module(net)

def init_LayerNorm(net):
    return

global init_by_class
init_by_class = {
    nn.Linear : init_Linear,
    nn.LayerNorm : init_LayerNorm,
    nn.ModuleList : init_ModuleList,
    nn.LSTM : init_LSTM,
    nn.LSTMCell : init_LSTM,
    LayerNormGRUCell : init_LayerNormGRUCell,
    FrameLevel : init_FrameLevel,
    SampleLevel : init_Module,
    SampleMagPhase : init_Module,
    Attn : init_Module,
    Char2Voc : init_Module,
    SampleRNN : init_Module,
    nn.Sequential : init_Module,
    nn.GRUCell : init_GRU
}

if __name__ == "__main__":
    rand_voc = torch.rand((20, 1, 81))
    _T, _B, _V = rand_voc.shape

    test_lngru = True
    test_srnn = True
    test_attn = True
    test_char2v = False

    if test_lngru:
        _H = 30
        print('Running toy forward prop for LayerNormGRUCell...')
        lngru = LayerNormGRUCell(_V, _H)
        init_by_class[lngru.__class__](lngru)
        with torch.no_grad():
            x_unbind = torch.unbind(rand_voc, dim=0)
            hid = lngru(x_unbind[0])
            res = [hid]
            for x in x_unbind[1:]:
                hid = lngru(x, hid)
                res.append(hid)
            res = torch.stack(res)
        assert res.shape == (_T, _B, _H)
        print('Forward Propagation for LayerNormGRUCell ran without error. ')

    if test_srnn:
        print('Running toy forward prop for SampleRNN...')
        gen_sample = False
        if gen_sample:
            srnn = SampleRNN(
                vocoder_size = _V,
                ratios = [2, 2],
                input_sizes = [512, 512],
                gru_hid_sizes = [1024, 1024],
                sample_level = True,
                hid_up_size = _V,
                audio_embd = 20
            )
            tar_out = torch.zeros(_T * 32, _B, _V)
        else:
            srnn = SampleRNN(
                ratios = [2, 2],
                input_sizes = [512, 512],
                gru_hid_sizes = [512, 512],
                hid_up_size = _V,
                vocoder_size = _V,
                do_layernorm = True,
                gen_in_size = 512,
                do_res = True,
                drop_out = 0.5
            )
            tar_out = torch.zeros(_T * 32, _B, 1)
        init_by_class[srnn.__class__](srnn)
        srnn.init_states(_B)
        with torch.no_grad():
            out = srnn(x_in = torch.zeros(_T * 32, _B, _V), hid_up = rand_voc)
        if gen_sample:
            assert out[1].shape == (_T * 32, _B, 1)
            assert out[0].shape == (_T * 32, _B, 256)
        else:
            assert out[1].shape == (_T * 32, _B, _V)
        print('Forward Propagation for SampleRNN ran without error. ')

    if test_attn:
        print('Running toy forward prop for attention implementation...')
        attion = Attn(
            len_seq = _T,
            decoded_size = _V
        )
        init_by_class[attion.__class__](attion)
        alpha = torch.zeros(1, 20)
        attion.kappa_init(1)
        test_de = torch.rand(1, 81)
        with torch.no_grad():
            alpha = attion(rand_voc[0], alpha)
        assert alpha.shape == (1, 20)
        print('Forward Propagation for Attn ran without error. ')

    if test_char2v:
        test_char_seq = torch.multinomial(rand_voc.squeeze(), 50, replacement=True) - 2.
        _B, _T = test_char_seq.shape
        _C = _V - 2
        print('Running toy forward prop for Char2Voc...')
        char2v = Char2Voc(
            len_seq = _T,
            num_type = _C,
            char_embd_size = 100,
            do_maskedLM = True
        )
        init_by_class[char2v.__class__](char2v)
        with torch.no_grad():
            decoded = char2v(test_char_seq, max_gen=50)
        assert decoded.shape == (_T, _B, _C)
        print('Char2Voc encoder-decoder with Attn ran without error. ')
