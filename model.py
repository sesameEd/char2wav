#!/usr/bin/python3
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def shape_ast(t, shape):
    assert len(t.shape) == len(shape) and all(x == y or y == -1 for x, y in zip(t.shape, shape)), \
    'Expected shape ({}), got shape ({})'.format(', '.join(map(str, shape)), ', '.join(map(str, t.shape)))

def init_weights(name, net):
    suffix = name.split('.')[-1]
    if suffix.split('_')[0] == 'bias':
        net.data.fill_(0.)
    elif suffix == 'weight_ih':
        for ih in net.chunk(3, 0):
            torch.nn.init.xavier_uniform_(ih)
    elif suffix == 'weight_hh':
        for hh in net.chunk(3, 0):
            torch.nn.init.orthogonal_(hh)
    elif suffix == 'weight':
        torch.nn.init.xavier_normal_(net)
    elif suffix == 'hid0':
        torch.nn.init.normal_(net)
    else:
        print(name)

def to_onehot(idx, batch, N_cat):
    return torch.zeros(batch, N_cat).scatter_(1, idx, 1.)

class Attn(nn.modules.Module):
    def __init__(self, **kwargs):
        super(Attn, self).__init__()
        self.N_gm = kwargs.get('num_component', 3)
        self.T = kwargs['len_seq']
        self.decoded_size = kwargs['decoded_size']
        self.ha2gmm = nn.Linear(self.decoded_size + self.T, self.N_gm * 3)

    def forward(self, prev_decoded, prev_attn):
        shape_ast(prev_decoded, (-1, self.decoded_size))
        _B, _H = prev_decoded.shape
        shape_ast(prev_attn, (_B, self.T))
        _B, _T = prev_attn.shape

        rho, beta, kappa = self.ha2gmm(torch.cat([prev_decoded, prev_attn], dim=-1)).split(self.N_gm, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        self.kappa = (self.kappa + F.relu(kappa)).unsqueeze(dim=-1)
        # what about replacing torch.exp() with ReLU???

        shape_ast(rho, (_B, self.N_gm, 1))
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

def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

class FrameLevel(nn.modules.Module):
    def __init__(self, tier_low, parent2inp, **kwargs):
        super(FrameLevel, self).__init__()
        self.R = kwargs['ratio']
        self.S = kwargs['sample_size']
        self.I = kwargs.get('input_size', 256)
        self.H = kwargs.get('hid_size', 1024)
        self.G = kwargs.get('gen_size', 1)
        self.dropout = kwargs.get('drop_out', .3)
        self.hid0 = nn.Parameter(torch.rand(1, self.H), requires_grad=True)
        assert (self.S / self.R).is_integer(), print('frame size wrong')
        self.add_module('w_x', nn.Linear(int((self.S / self.R) * self.G), self.I))
        self.add_module('w_j', parent2inp)
        self.add_module('recursive', nn.GRUCell(self.I, self.H))
        self.add_module('child', tier_low)

    def forward(self, x_in, hid_up):
        shape_ast(x_in, (-1, self.S * self.G))
        _B, _SG = x_in.shape
        shape_ast(hid_up, (_B, -1))
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
            logsm_pre, x_pre = self.child(x_in[-1], self.hid)
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
        mlp_sizes = [self.I]+ kwargs.get('mlp_sizes', [1024,1024,256])
        embd_size = kwargs.get('audio_embd', 0)
        self.embd_on = True if embd_size > 0 else False # whether to use an embdding layer for inputs
        self.D_bt = kwargs.get('bit_depth', 8)
        if self.embd_on:
            self.add_module('embd', nn.Linear(2 ** self.D_bt, embd_size))
            self.add_module('w_x', nn.Linear(self.S * embd_size, self.I))
        else:
            self.add_module('w_x', nn.Linear(self.S, self.I))
        self.add_module('w_j', parent2inp)
        self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1]) for i in range(len(mlp_sizes)-1)])
        self.add_module('h2s', nn.Linear(mlp_sizes[-1], 2 ** self.D_bt)) # or a 8-way sigmoid?

    def forward(self, x, hid_up):
        shape_ast(x, (-1, self.S))
        _B, _S = x.shape
        shape_ast(hid_up, (_B, -1))
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
        shape_ast(x, (batch, 1))
        return self.embd(to_onehot(x, batch, 2 ** self.D_bt))


class VocoderLevel(nn.modules.Module):
    def __init__(self, parent2inp, **kwargs):
        super(VocoderLevel, self).__init__()
        self.sample_size = kwargs.get('frame_size', 8)
        self.I = kwargs.get('gen_in_size', 256) # dimension of input vector
        self.vocoder_size = kwargs.get('vocoder_size', 81)
        mlp_sizes = [self.I] + kwargs.get('mlp_sizes', [1024,512]) + [self.vocoder_size]
        self.add_module('w_x', nn.Linear(self.sample_size * self.vocoder_size, self.I))
        self.add_module('w_j', parent2inp)
        self.mlp = nn.ModuleList([nn.Linear(mlp_sizes[i], mlp_sizes[i+1]) for i in range(len(mlp_sizes)-1)])
        # self.add_module('conv1', nn.Conv1d(in_channels=1, out_channels=6, kernel_size=10, stride=5))

    def forward(self, x, hid_up):
        shape_ast(x, (-1, self.sample_size * self.vocoder_size))
        _B, _S  = x.shape
        shape_ast(hid_up, (_B, -1))
        x_out = deque(x.split(self.vocoder_size, dim=1), maxlen=self.sample_size)
        c_j = self.w_j(hid_up).split(self.I, dim=-1)
        for _j in range(self.sample_size):
            inp = self.w_x(torch.cat(list(x_out), dim=-1)) + c_j[_j]
            for fc in self.mlp:
                inp = fc(F.relu(inp))
            x_out.append(inp)
        return [], list(x_out)


class SampleRNN(nn.modules.Module):
    def __init__(self, **kwargs):
        super(SampleRNN, self).__init__()
        self.vocoder_size = kwargs['vocoder_size']
        self.B = kwargs['batch_size']
        self.ratios = kwargs.get('ratios', [1,2,2])
        self.input_sizes = kwargs.get('input_sizes', [256, 256, 256])
        self.gru_hid_sizes = kwargs.get('gru_hid_sizes', [1024, 1024, 1024])
        self.gen_sample = kwargs.get('sample_level', False)
        self.gen_size = 1 if self.gen_sample else self.vocoder_size
        """default args for SampleLevel :
                frame_size  -> 8
                gen_in_size -> 256
                mlp_sizes   -> [1024, 1024, 256]
                audio_embd  -> 0
                bit_depth   -> 8"""
        """default args for VocoderLevel:
                frame_size  -> 8
                gen_in_size -> 256
                mlp_sizes   -> [1024, 512]
                vocoder_size-> 81
                bit_depth   -> 8"""
        if (len(self.ratios) != len(self.input_sizes) or
            len(self.input_sizes) != len(self.gru_hid_sizes)):
            print('Wrong size for the lists of params, {0} frame sizes,'.format(len(self.ratios)),
                  '{0} input sizes, and {1} recurrent sizes'.format(len(self.input_sizes), len(self.gru_hid_sizes)))
        self.n_tiers = len(self.ratios) + 1
        self.FS = self.ratios + [kwargs.get('frame_size', 8)]
        self.samp_sizes = list(np.cumprod(self.FS[::-1]))[:0:-1]
        up_sizes = [self.vocoder_size] + self.gru_hid_sizes

        params = zip(
            self.ratios,
            self.input_sizes,
            self.gru_hid_sizes,
            self.samp_sizes,
            [self.gen_size] * len(self.ratios)
        )
        params = [dict(zip(['ratio',
                            'input_size',
                            'hid_size',
                            'sample_size',
                            'gen_size'],
                           tp))
                  for tp in params]
        wj_prm = zip(up_sizes, self.FS, self.input_sizes + [kwargs.get('gen_in_size', 256)]) # 0, .., self.n_tiers-1
        wj_prm = [(tp[0], tp[1]*tp[2]) for tp in wj_prm]
        if self.gen_sample:
            tier = SampleLevel(nn.Linear(*wj_prm[-1]), **kwargs)
        else:
            tier = VocoderLevel(nn.Linear(*wj_prm[-1]), **kwargs)
        for k in reversed(range(self.n_tiers-1)):
            w_j = nn.Linear(*wj_prm[k])
            tier = FrameLevel(tier, w_j, **params[k])
        self.add_module('srnn', tier)

    def forward(self, vocoder):
        """
        :input array of vocoders, of shape (_T, _B, _V)
        :output
            if self.gen_sample (generates raw waves):
                log prob and prob, of shape (_T * pi(FS), _B, 2 ** depth)
            if not (generates vocoder features):
                [] and vocoders, of shape (_T * pi(FS), _B, _V)
        """
        shape_ast(vocoder, (-1, self.B, self.vocoder_size))
        _T, _B, _V = vocoder.shape
        logsm, x_out = [], []
        vocoder_single = torch.unbind(vocoder, dim=0)
        for voc in vocoder_single:
            logsm_pre, x_pre = self.srnn(self.x_in, voc)
            self.x_in = torch.cat(x_pre, dim=-1)
            x_out += x_pre
            logsm += logsm_pre
        x_out = torch.stack(x_out, dim=0)
        if self.gen_sample:
            logsm = torch.stack(logsm, dim=0)
        return logsm, x_out

    def init_states(self):
        self.srnn.hid_init(self.B)
        self.x_in = var(torch.zeros(self.B, self.samp_sizes[0] * self.gen_size))

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

if __name__ == "__main__":
    test_voc = torch.rand((20, 1, 81))
    _T, _B, _V = test_voc.shape

    run_srnn = True
    run_attn = True
    run_char2v = False

    if run_srnn:
        print('Running toy forward prop for SampleRNN...')
        srnn = SampleRNN(
            vocoder_size = 81,
            ratios = [2],
            input_sizes = [512],
            gru_hid_sizes = [1024],
            sample_level = True,
            audio_embd = 20,
            batch_size = _B
            )
        for name, param in srnn.named_parameters():
            init_weights(name, param)
        srnn.init_states()
        with torch.no_grad():
            out = srnn(test_voc)
            assert out[1].shape == (_T * 16, _B, 1)
            assert out[0].shape == (_T * 16, _B, 256)
        print('Forward Propagation for SampleRNN ran without error. ')

    if run_attn:
        print('Running toy forward prop for attention implementation...')
        attion = Attn(
                len_seq = 20,
                decoded_size = 81
            )
        alpha = torch.zeros(1, 20)
        attion.kappa_init(1)
        test_de = torch.rand(1, 81)
        with torch.no_grad():
            alpha = attion(test_voc[0], alpha)
        assert alpha.shape == (1, 20)
        print('Forward Propagation for Attn ran without error. ')
