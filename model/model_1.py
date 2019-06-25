#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.init as init
import torch.nn.utils.rnn as R
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def shape_assert(t, shape):
    assert len(t.shape) == len(shape) and all(
        x == y or y == -1 for x, y in zip(t.shape, shape)), \
        'Expected shape ({}), got shape ({})'.format(
        ', '.join(map(str, shape)), ', '.join(map(str, t.shape)))


def var(tensor):
    return tensor.to(device)

class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()
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
        self.kappa = var(torch.zeros(batch_size, T, self.K))


class Char2Voc(nn.modules.Module):
    def __init__(self, num_types, encoded_size, decoded_size,
                 *, gen_size=82, **kwargs):
        super(Char2Voc, self).__init__()
        self.E = kwargs.get('embedding_size', 32)
        self.K = kwargs.get('num_GMM', 3)
        self.G = gen_size
        self.upper_in = kwargs.get('upper_in', False)
        if self.upper_in:
            self.embedding = nn.Embedding(num_types+1, self.E-1, padding_idx=0)
        else:
            self.embedding = nn.Embedding(num_types+1, self.E, padding_idx=0)
        self.encoder = nn.GRU(self.E, encoded_size, bidirectional=True)
        self.attention = Attention(2 * encoded_size + decoded_size, self.K)
        self.decoder = nn.GRUCell(2 * encoded_size + gen_size, decoded_size)
        self.gen = nn.Linear(decoded_size, gen_size)

    def forward(self, char_seq, y_tar, upper_case=None, tf_rate=0, input_mask=None):
        """takes padded sequence as input"""
        shape_assert(char_seq, (self.B, -1))
        _B, _T = char_seq.shape
        shape_assert(y_tar, (-1, -1, self.G))
        x_embd = self.embedding(char_seq) * input_mask.float().unsqueeze(-1)
        if upper_case is not None:
            shape_assert(upper_case, (_B, _T))
            assert self.upper_in, ("did not expect upper case indicator," 
                "please re-initialize model with kwarg 'upper_in' set to True")
            x_embd = torch.cat(x_embd, upper_case.unsqueeze(dim=-1))
        encoded = self.encoder(x_embd)[0].transpose(0, 1) # of shape (_B, _T, 2 * _H)
        res = [self.y_pre]
        attn_val = encoded[:, 0, :]
        print(attn_val.shape, self.y_pre.shape)
        hid = self.decoder(torch.cat([self.y_pre, attn_val], dim=-1))
        self.attention.kappa_init(_B, _T)
        for y_t in y_tar.unbind(dim=1):
            query = torch.cat([attn_val, hid], dim=-1)
            attn_val = self.attention(query, input_mask).unsqueeze(-1) * encoded
            y_ss = {0 : self.y_pre, 1 : y_t}
            hid = self.decoder(torch.cat([y_ss[np.random.binomial(1, tf_rate)],
                                          attn_val], dim=-1), hid)
            self.y_pre = self.gen(hid)
            res.append(self.y_pre)
        return torch.stack(res, dim=1)

    def gen_init(self, batch_size):
        self.B = batch_size
        self.y_pre = var(torch.zeros(batch_size, self.G))
        self.y_pre[:, :60].normal_(mean=-1.2, std=.5)


if __name__ == "__main__":
    rand_voc = torch.rand((4, 8, 82))
    _T, _B, _V = rand_voc.shape
    test_char2v = True

    if test_char2v:
        c2v = Char2Voc(num_types=40, encoded_size = 256, decoded_size = 512).to(device)
        x_ls = [torch.from_numpy(np.random.randint(1, 41, size=_i))
                for _i in np.random.randint(10, size=10) if _i >= 2]
        c2v.gen_init(batch_size=len(x_ls))
        x_in, lens_seq = R.pad_packed_sequence(
            R.pack_sequence(x_ls, enforce_sorted=False), batch_first=True)
        x_places = torch.arange(x_in.shape[1]).unsqueeze(0)
        pad_mask = x_places < lens_seq.unsqueeze(-1)
        print(pad_mask)
        y_pre = c2v(var(x_in), var(rand_voc), input_mask=var(pad_mask))
