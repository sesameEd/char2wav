#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from utils import shape_assert
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_torchtsr(a):
    return 'ndarray(shape={}, dtype={})'.format(a.shape, a.dtype)

def var(tensor):
    return tensor.to(device)

class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()
        self.ln_zr_ih, self.ln_zr_hh = [nn.LayerNorm(2 * hidden_size) for _ in range(2)]
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

class Attention(nn.modules.Module):
    def __init__(self, query_size, num_component=3):
        super(Attention, self).__init__()
        self.Q = query_size
        self.K = num_component
        self.ha2gmm = nn.Linear(self.Q, self.K * 3)

    def forward(self, query, input_mask):
        shape_assert(query, (-1, self.Q))
        rho, beta, kappa = self.ha2gmm(query).split(self.K, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        self.kappa = F.softmax(self.kappa.squeeze() + F.relu(kappa), dim=-1
                               ).unsqueeze(dim=-1)
        # shape_assert(rho, (query.shape[0], self.K, 1))
        phi = (rho * torch.exp(
               -beta * (self.kappa - torch.arange(float(self.T)).view(1, 1, self.T)) ** 2)
               ).sum(dim=1)
        return F.softmax(phi, dim=1) * input_mask.float()

    def kappa_init(self, batch_size, T):
        self.T = T
        self.kappa = var(torch.zeros(batch_size, self.K))


class Char2Voc(nn.modules.Module):
    def __init__(self, num_types, encoded_size, decoded_size,
                 *, gen_size=82, **kwargs):
        super(Char2Voc, self).__init__()
        self.E = kwargs.get('embedding_size', 32)
        self.K = kwargs.get('num_GMM', 3)
        self.G = gen_size
        self.upper_in = kwargs.get('upper_in', False)
        num_embd = self.E-1 if self.upper_in else self.E
        self.embedding = nn.Embedding(num_types+1, num_embd, padding_idx=0)
        self.y0 = nn.Parameter(torch.rand(1, self.G), requires_grad=True)
        self.encoder = nn.GRU(self.E, encoded_size, bidirectional=True, batch_first=True)
        self.attention = Attention(2 * encoded_size + decoded_size, self.K)
        self.decoder = nn.GRUCell(2 * encoded_size + gen_size, decoded_size)
        self.gen = nn.Linear(decoded_size, gen_size)

    def forward(self, char_seq, y_tar, upper_case=None, tf_rate=0, input_mask=None, voc_mask=None):
        """takes padded sequence as input"""
        shape_assert(char_seq, (self.B, -1))
        _B, _T = char_seq.shape
        shape_assert(y_tar, (-1, -1, self.G))
        if input_mask is None:
            input_mask = torch.ones(_B, _T).to(device)
        if voc_mask is None:
            voc_mask = torch.ones(_B, y_tar.shape[1], 1)
        elif voc_mask.dim() == 2:
            voc_mask.unsqueeze_(2)
        x_embd = self.embedding(char_seq) * input_mask.float().unsqueeze(-1)
        if upper_case is not None:
            shape_assert(upper_case, (_B, _T))
            assert self.upper_in, ("did not expect upper_tensor, re-initialize "
                                   "model with kwarg 'upper_in' switched on")
            x_embd = torch.cat((upper_case.unsqueeze(dim=-1).float(), x_embd), dim=2)
        encoded = self.encoder(x_embd)[0]  # of shape (_B, _T, 2 * _H)
        res = [self.y_pre]
        attn_val = encoded[:, 0, :]
        hid = self.decoder(torch.cat([self.y_pre, attn_val], dim=-1))
        self.attention.kappa_init(_B, _T)
        for y_t in y_tar.unbind(dim=1)[:-1]:
            query = torch.cat([attn_val, hid], dim=-1)
            attn_key = self.attention(query, input_mask).unsqueeze(-1)
            attn_val = (attn_key * encoded).sum(dim=1)
            y_ss = {0: self.y_pre, 1: y_t}
            hid = self.decoder(torch.cat([y_ss[np.random.binomial(1, tf_rate)],
                                          attn_val], dim=-1), hid)
            self.y_pre = self.gen(hid)
            self.y_pre[:, -2] = torch.sigmoid(self.y_pre[:, -2])
            res.append(self.y_pre)
        return torch.stack(res, dim=1) * voc_mask

    def gen_init(self, batch_size):
        self.B = batch_size
        self.y_pre = var(self.y0.expand(batch_size, -1))
        # self.y_pre = var(torch.zeros(batch_size, self.G))

def init_Linear(net):
    init.zeros_(net.bias)
    init.xavier_normal_(net.weight)

def init_recurrent(weight, num_chunks, initializer):
    for _w in weight.chunk(num_chunks, 0):
        initializer(_w)

def init_ModuleList(m_list):
    for layer in m_list:
        init_by_class.get(layer.__class__, init_Module)(layer)

def init_Module(net):
    for layer in net.children():
        init_by_class.get(layer.__class__, init_Module)(layer)

def init_GRU(net):
    for n, p in net.named_parameters():
        if 'bias' in n:
            init.zeros_(p)
        if 'weight_ih' in n:
            init_recurrent(p, 3, init.xavier_uniform_)
        if 'weight_hh' in n:
            init_recurrent(p, 3, init.orthogonal_)
    # init_recurrent(net.weight_ih, num_chunks=3,
    #                initializer=init.xavier_uniform_)
    # init_recurrent(net.weight_hh, num_chunks=3,
    #                initializer=init.orthogonal_)

def init_LSTM(lstm):
    for n, p in lstm.named_parameters():
        if 'bias' in n:
            init.zeros_(p)
        if 'weight_ih' in n:
            init_recurrent(p, 4, init.xavier_uniform_)
        if 'weight_hh' in n:
            init_recurrent(p, 4, init.orthogonal_)

def init_Char2Voc(net):
    net.y0[:60].normal_(mean=-1.2, std=.5)
    init_Module(net)


global init_by_class
init_by_class = {
    nn.Linear : init_Linear,
    nn.ModuleList : init_ModuleList,
    nn.GRUCell : init_GRU,
    nn.GRU : init_GRU,
    Char2Voc : init_Char2Voc}

if __name__ == "__main__":
    test_char2v = True

    if test_char2v:
        c2v = Char2Voc(num_types=40, encoded_size=256, decoded_size=512).to(device)
        init_by_class.get(c2v.__class__, init_Module)(c2v)
        print("running test Char2Voc model...")
        x_ls = [torch.from_numpy(np.random.randint(1, 41, size=_i))
                for _i in np.random.randint(10, size=10) if _i >= 2]
        c2v.gen_init(batch_size=len(x_ls))
        rand_voc = torch.rand((len(x_ls), 8, 82))
        x_in, lens_seq = R.pad_packed_sequence(
            R.pack_sequence(x_ls, enforce_sorted=False), batch_first=True)
        x_places = torch.arange(x_in.shape[1]).unsqueeze(0)
        pad_mask = x_places < lens_seq.unsqueeze(-1)
        # print(pad_mask)
        y_pre = c2v(var(x_in), var(rand_voc), input_mask=var(pad_mask), tf_rate=.5)
        assert y_pre.shape == rand_voc.shape, "expected output shape {0}, got {1}".format(
            rand_voc.shape, y_pre.shape)
        print("Char2Voc model with attention ran without error!")
