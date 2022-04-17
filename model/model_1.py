#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from utils import shape_assert, var, ConvNorm, LinearNorm
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_torchtsr(a):
    return 'ndarray(shape={}, dtype={})'.format(a.shape, a.dtype)


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


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    copied from https://github.com/NVIDIA/DeepLearningExamples.git/
    """
    def __init__(self, encoder_embedding_dim=512, encoder_n_convolutions=3,
                 encoder_kernel_size=5):
        super(Encoder, self).__init__()
        self.E = encoder_embedding_dim
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(self.E, self.E, stride=1, dilation=1, groups=self.E,
                         kernel_size=encoder_kernel_size, w_init_gain='relu',
                         padding=int((encoder_kernel_size - 1) / 2)),
                nn.BatchNorm1d(self.E))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.recurrent = nn.GRU(self.E, int(self.E / 2), 1,
                                batch_first=True, bidirectional=True)

    def forward(self, x):
        shape_assert(x, (-1, -1, self.E))
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
#         self.recurrent.flatten_parameters()
        # outputs, _ = self.recurrent(x)
        return self.recurrent(x)


class AttentionOld(nn.modules.Module):
    def __init__(self, query_size, num_component=3):
        super(Attention, self).__init__()
        self.Q = query_size
        self.K = num_component
        self.ha2gmm = nn.Linear(self.Q, self.K * 3)
        self.const_shift = torch.zeros([], requires_grad=False)

    def forward(self, query, input_mask):
        shape_assert(query, (-1, self.Q))
        rho, beta, kp = self.ha2gmm(query).split(self.K, dim=-1)
        rho = torch.exp(rho).unsqueeze(dim=-1)
        beta = torch.exp(beta).unsqueeze(dim=-1)
        self.kappa = self.kappa + F.relu(kp).unsqueeze(dim=-1)
        phi = (rho * torch.exp(
               -beta * (self.kappa - var(torch.arange(self.T)).view(1, 1, -1).float()) ** 2)
               ).sum(dim=1)
        return F.softmax(phi, dim=1) * input_mask.float()

    def weight_init(self):
        init_Linear(self.ha2gmm)
        with torch.no_grad():
            self.ha2gmm.bias[-self.K:] = var(self.const_shift)
        # print(self.ha2gmm.bias.grad_fn, self.ha2gmm.weight.grad_fn)

    def kappa_init(self, batch_size, T):
        self.T = T
        self.kappa = var(torch.zeros(batch_size, self.K, 1))


class LocationLayer(nn.Module):
    """copied from https://github.com/NVIDIA/DeepLearningExamples.git/"""
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    """copied from https://github.com/NVIDIA/DeepLearningExamples.git/
    minor alterations"""
    # git@github.com:NVIDIA/DeepLearningExamples.git/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py"""
    def __init__(self, prev_dec_dim=1024, embedding_dim=512,
                 attention_dim=64, attention_location_n_filters=32,
                 attention_location_kernel_size=31):
        super(Attention, self).__init__()
        self.prev_dec_layer = LinearNorm(prev_dec_dim, attention_dim,
                                         bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.masked_score = -float("inf")

    def get_alignment_energies(self, prev_dec, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        prev_dec: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_prev_dec = self.prev_dec_layer(prev_dec.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_prev_dec + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, prev_decoded, processed_memory,
                attention_weights_cat, input_mask=None):
        # memory,
        """
        PARAMS
        ------
        prev_decoded: attention rnn last output
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        input_mask: binary mask for padded data
        RETURNS
        -------
        attention_weights: (batch, max_time)
        """
        alignment = self.get_alignment_energies(
            prev_decoded, processed_memory, attention_weights_cat)

        if input_mask is not None:
            alignment.data.masked_fill_(~input_mask, self.masked_score)

        attention_weights = F.softmax(alignment, dim=1)
        # memory: encoder outputs
        # attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # attention_context = attention_context.squeeze(1)
        return attention_weights  # , attention_context


class Char2Voc(nn.modules.Module):
    def __init__(self, num_types, embedding_size, decoded_size,
                 *, gen_size=82, do_maskedLM=False, **kwargs):  # encoded_size,
        super(Char2Voc, self).__init__()
        self.do_maskedLM = do_maskedLM
        self.E = embedding_size
        self.K = kwargs.get('num_GMM', 3)
        self.G = num_types if self.do_maskedLM else gen_size
        self.D = kwargs.get('dropout', 0)
        self.upper_in = kwargs.get('upper_in', False)
        encoder_n_convolutions = kwargs.get('encoder_n_convolutions', 3)
        encoder_kernel_size = kwargs.get('encoder_kernel_size', 5)
        num_embd = self.E-1 if self.upper_in else self.E
        self.m = nn.Dropout(p=self.D)
        self.y0 = nn.Parameter(var(torch.zeros(self.G)), requires_grad=True)
        self.embedding = nn.Embedding(num_types+1, num_embd, padding_idx=0)
        # self.encoder = nn.GRU(self.E, self.E // 2, bidirectional=True,
        #                       batch_first=True, dropout=self.D, num_layers=2)
        # self.attention = Attention(2 * encoded_size + decoded_size, self.K)
        self.encoder = Encoder(self.E, encoder_n_convolutions, encoder_kernel_size)
        self.attention = Attention(decoded_size, embedding_dim=self.E)
        self.decoder = nn.GRUCell(self.E + self.G, decoded_size)
        self.gen = nn.Linear(decoded_size, self.G)

    def forward(self, char_seq, y_tar, upper_case=None, tf_rate=0, input_mask=None):
        """takes padded sequence as input"""
        shape_assert(char_seq, (-1, -1))
        _B, _T = char_seq.shape
        shape_assert(y_tar, (-1, -1, self.G))
        if input_mask is None:
            input_mask = var(torch.ones(_B, _T))
        x_embd = self.embedding(char_seq) * input_mask.float().unsqueeze(-1)
        if upper_case is not None:
            shape_assert(upper_case, (_B, _T))
            assert self.upper_in, ("did not expect upper_tensor, re-initialize "
                                   "model with kwarg 'upper_in' switched on")
            x_embd = torch.cat((upper_case.unsqueeze(dim=-1).float(), x_embd), dim=2)
        encoded = self.encoder(x_embd)[0]  # of shape (_B, _T, 2 * _H)
        p_encoded = self.attention.memory_layer(encoded)
        res = [var(self.y0.expand(_B, -1))]
        attn_val = encoded[:, 0, :]
        attn_key = [var(torch.zeros(_B, _T).scatter_(1, torch.zeros(_B, 1).long(), 1.))]
        cum_attn = attn_key[-1].unsqueeze(1)
        hid = self.decoder(torch.cat([res[-1], attn_val], dim=-1))
        # self.attention.kappa_init(_B, _T)
        for y_t in y_tar.unbind(dim=1)[:-1]:
            # query = torch.cat([attn_val, hid], dim=-1)
            attn_cat = torch.cat([attn_key[-1].unsqueeze(1), cum_attn], dim=1)
            attn_key.append(self.attention(hid, p_encoded, attn_cat, input_mask))
            cum_attn += attn_key[-1].unsqueeze(1)
            attn_val = (attn_key[-1].unsqueeze(-1) * encoded).sum(dim=1)
            y_ss = {0: res[-1], 1: y_t}
            hid = self.decoder(torch.cat([y_ss[np.random.binomial(1, tf_rate)],
                                          attn_val], dim=-1), hid)
            y_pre = self.gen(hid)
            if self.do_maskedLM:
                F.log_softmax_(y_pre, dim=-1)
            else:
                torch.sigmoid_(y_pre[:, -2])
            res.append(y_pre)
        return torch.stack(res, dim=1), torch.stack(attn_key, dim=1)  # * voc_mask.float()

    def weight_init(self):
        for net in (self.encoder, self.decoder):
            init_GRU(net)
        init_Linear(self.gen)
        init.xavier_normal_(self.embedding.weight)
        # self.attention.weight_init()
        init.normal_(self.y0[:60], mean=-1.2, std=.5)
        init.normal_(self.y0[60:])
        init.uniform_(self.y0[-2])


def init_Linear(net):
    init.zeros_(net.bias)
    init.xavier_normal_(net.weight)


def init_recurrent(weight, num_chunks, initializer):
    for _w in weight.chunk(num_chunks, 0):
        initializer(_w)


def init_GRU(net):
    for n, p in net.named_parameters():
        if 'bias' in n:
            init.zeros_(p)
        if 'weight_ih' in n:
            init_recurrent(p, 3, init.xavier_uniform_)
        if 'weight_hh' in n:
            init_recurrent(p, 3, init.orthogonal_)


def init_LSTM(lstm):
    for n, p in lstm.named_parameters():
        if 'bias' in n:
            init.zeros_(p)
        if 'weight_ih' in n:
            init_recurrent(p, 4, init.xavier_uniform_)
        if 'weight_hh' in n:
            init_recurrent(p, 4, init.orthogonal_)


if __name__ == "__main__":
    test_char2v = True

    if test_char2v:
        c2v = var(Char2Voc(num_types=40, embedding_size=256, decoded_size=512,
                           dropout=.5))
        c2v.weight_init()
        print("running test Char2Voc model...")
        x_ls = [torch.from_numpy(np.random.randint(1, 41, size=_i))
                for _i in np.random.randint(10, size=10) if _i >= 2]
        rand_voc = torch.rand((len(x_ls), 8, 82))
        x_in, lens_seq = R.pad_packed_sequence(
            R.pack_sequence(x_ls, enforce_sorted=False),
            batch_first=True)
        print(x_in.shape, lens_seq)
        x_places = torch.arange(x_in.shape[1]).unsqueeze(0)
        pad_mask = x_places < lens_seq.unsqueeze(-1)
        # print(pad_mask)
        y_pre = c2v(var(x_in), var(rand_voc), input_mask=var(pad_mask), tf_rate=.5)[0]
        assert y_pre.shape == rand_voc.shape, "expected output shape {0}, got {1}".format(
            rand_voc.shape, y_pre.shape)
        print("Char2Voc model with attention ran without error!")
