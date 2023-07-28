import torch
import torch.nn as nn
from modules import (
    PreNet,
    ResGRUCell,
    ContentGeneralAttention,
    ContentMarkovAttention,
)


class Decoder1Cell(nn.Module):
    def __init__(self, dim_ctx, dim_mel, r, dim_pre=128, dim_att=256):
        super().__init__()
        self.dim_ctx = dim_ctx
        self.dim_mel = dim_mel
        self.dim_pre = dim_pre
        self.dim_att = dim_att
        self.r = r
        self.decoder_depth = 2
        self.dim_rnn = dim_att + dim_ctx

        self.pre_net = PreNet(dim_mel, dim_pre)
        # self.attention_module = ContentGeneralAttention(dim_ctx, dim_att)
        self.attention_module = ContentMarkovAttention(dim_ctx, dim_att)
        self.attention_rnn = nn.GRUCell(dim_pre, dim_att)
        self.decoder_rnn_list = nn.ModuleList(
            [ResGRUCell(self.dim_rnn) for _ in range(self.decoder_depth)]
        )

    def initial_state(self, batch_size, memory_size, dtype, device):
        w_0 = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=dtype, device=device),
                torch.zeros((batch_size, memory_size - 1), dtype=dtype, device=device),
            ],
            dim=1,
        )  # Initial attention weights B x L
        h_att_0 = torch.zeros((batch_size, self.dim_att), dtype=dtype, device=device)
        h_dec_0 = [
            torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device)
            for _ in range(self.decoder_depth)
        ]
        return w_0, h_att_0, h_dec_0

    def forward(self, x, dec_state, memory):
        w, h_att, h_dec = dec_state
        # x:    B x r x D_mel
        # w:    B x L
        # h_att:B x D_att
        # h_dec:B x D_dec
        x = self.pre_net(x.flatten(1, 2))  # B x D_pre

        ctx_att = torch.bmm(w.unsqueeze(1), memory).squeeze(1)  # B x D_ctx
        h_att = self.attention_rnn(x, h_att)  # B x D_att
        w = self.attention_module(h_att, w, memory)  # B x D_enc

        x_dec = torch.cat((h_att, ctx_att), dim=1)  # B x (D_att+D_ctx)
        for idx, rnn in enumerate(self.decoder_rnn_list):
            h_dec[idx] = rnn(x_dec, h_dec[idx])  # B x D_rnn
            x_dec = h_dec[idx]

        dec_state = w, h_att, h_dec
        return x_dec, dec_state


class DecoderCell(nn.Module):
    def __init__(self, dim_ctx, dim_mel, r, dim_pre=128, dim_att=128):
        super().__init__()
        self.dim_ctx = dim_ctx
        self.dim_mel = dim_mel
        self.dim_pre = dim_pre
        self.dim_att = dim_att
        self.r = r
        self.decoder_depth = 2
        self.dim_rnn = dim_pre + dim_ctx

        self.attention_module = ContentMarkovAttention(dim_ctx, self.dim_rnn)
        # self.attention_module = ContentMarkovAttention(dim_ctx, dim_att)
        self.pre_net = PreNet(r * dim_mel, dim_pre)
        # self.attention_rnn = nn.LSTMCell(self.dim_rnn + dim_ctx, dim_att)
        # self.attention_fc = nn.Linear(self.dim_rnn, dim_att, bias=False)
        self.decoder_rnn_list = nn.ModuleList(
            # [ResGRUCell(self.dim_rnn) for _ in range(self.decoder_depth)]
            # [ResLSTMCell(self.dim_rnn, self.dim_rnn) for _ in range(self.decoder_depth)]
            [nn.LSTMCell(self.dim_rnn, self.dim_rnn) for _ in range(self.decoder_depth)]
        )

    def initial_state(self, batch_size, memory_size, dtype, device):
        # type: (int, int, torch.dtype, torch.device) -> Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]
        w_0 = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=dtype, device=device),
                torch.zeros((batch_size, memory_size - 1), dtype=dtype, device=device),
            ],
            dim=1,
        )  # Initial attention weights B x L
        h_dec_0 = [
            (
                torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device),
                torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device),
            )
            for _ in range(self.decoder_depth)
        ]  # Initial decoder RNN cell states B x D_rnn
        return w_0, h_dec_0

    def forward(self, x, dec_state, memory, mmask):
        # type: (Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]], Tensor, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]]
        # x:    B x r x D_mel
        # w:    B x L
        # memory: B x L x D_ctx
        w, h_dec = dec_state[0], dec_state[1]

        x = self.pre_net(x.flatten(1, 2))  # B x D_pre

        ctx_att = torch.bmm(w.unsqueeze(1), memory).squeeze(1)  # B x D_ctx

        x_dec = torch.cat((x, ctx_att), dim=1)  # B x (D_pre+D_ctx)
        for idx, rnn in enumerate(self.decoder_rnn_list):
            h_dec[idx] = rnn(x_dec, h_dec[idx])  # B x D_rnn
            x_dec = h_dec[idx][0] + x_dec

        # x_att = torch.cat((x_dec, ctx_att), dim=1)
        # h_att = self.attention_rnn(x_att, h_att)  # B x D_att
        # x_att = self.attention_fc(x_dec)
        x_att = x_dec
        w = self.attention_module(x_att, w, memory, mmask)  # B x L

        dec_state = w, h_dec
        return x_dec, dec_state


class Decoder(nn.Module):
    def __init__(self, decoder_cell, max_steps=200, stop_threshold=-2):
        super().__init__()
        self.decoder_cell = decoder_cell
        self.max_steps = max_steps
        self.stop_threshold = stop_threshold
        self.r = self.decoder_cell.r
        self.dim_mel = self.decoder_cell.dim_mel
        self.fc_mel = nn.Linear(decoder_cell.dim_rnn, self.r * self.dim_mel)
        self.fc_stop = nn.Linear(decoder_cell.dim_rnn, self.r)

    def forward(self, memory, mmask, x):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
        # memory: B x L x D_enc
        # x:      B x T x D_mel
        B = memory.shape[0]  # batch size
        L = memory.shape[1]  # text length
        dtype = memory.dtype
        device = memory.device

        state_t = self.decoder_cell.initial_state(B, L, dtype, device)

        # GO frame B x r x D_mel
        y_t = torch.zeros((B, self.r, self.dim_mel), dtype=dtype, device=device)
        x_split = None if x is None else x.split(self.r, dim=1)

        y, s, w = [], [], []
        step = 0
        while True:
            # y_t = y_t[:, -1, :].unsqueeze(1)
            d_t, state_t = self.decoder_cell(y_t, state_t, memory, None)

            w_t = state_t[0]
            s_t = self.fc_stop(d_t).unsqueeze(2)  # B x r x 1
            y_t = self.fc_mel(d_t).view(-1, self.r, self.dim_mel)  # B x r x D_mel

            y.append(y_t)
            s.append(s_t)
            w.append(w_t)
            step += 1

            if x_split is not None:
                if step >= len(x_split):
                    break
                # Force teacher inputs
                if torch.rand(1) > 0.1:
                    y_t = x_split[step - 1]  # B x r x D_mel
            else:
                if torch.all(s_t < self.stop_threshold) or step > self.max_steps:
                    break

        y = torch.cat(y, dim=1)  # B x T x D_mel
        s = torch.cat(s, dim=1)  # B x T x 1
        w = torch.stack(w, dim=1)  # B x T x L

        return y, s, w
