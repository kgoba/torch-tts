import torch
import torch.nn as nn
from modules import PreNet, ResGRUCell, ContentGeneralAttention


class DecoderCell(nn.Module):
    def __init__(self, dim_enc, dim_mel, r, dim_pre=128, dim_att=128):
        super().__init__()
        self.dim_mel = dim_mel
        self.dim_pre = dim_pre
        self.dim_att = dim_att
        self.r = r
        self.decoder_depth = 2
        self.dim_rnn = 2 * dim_att
        self.attention_module = ContentGeneralAttention(dim_enc, dim_att)
        self.pre_net = PreNet(self.r * dim_mel, dim_pre)
        self.attention_rnn = nn.GRUCell(dim_pre, dim_att)
        self.decoder_rnn_list = nn.ModuleList([ResGRUCell(self.dim_rnn) for _ in range(self.decoder_depth)])
        self.fc_mel = nn.Linear(self.dim_rnn, r * dim_mel)
        self.fc_stop = nn.Linear(self.dim_rnn, 1)

    def initial_state(self, batch_size, memory_size, dtype, device):
        h_att_0 = torch.zeros((batch_size, self.dim_att), dtype=dtype, device=device)
        h_dec_0 = [
            torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device)
            for _ in range(self.decoder_depth)
        ]
        w_0 = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=dtype, device=device),
                torch.zeros((batch_size, memory_size - 1), dtype=dtype, device=device),
            ],
            dim=1,
        )  # Initial attention weights B x L
        return w_0, h_att_0, h_dec_0

    def forward(self, x, w, h_att, h_dec, encoder_outputs):
        # x:    B x r x D_mel
        # w:    B x L
        # h_att:B x D_att
        # h_dec:B x D_dec
        x = self.pre_net(x.flatten(1, 2))  # B x D_pre

        h_att = self.attention_rnn(x, h_att)  # B x D_att
        att, w = self.attention_module(h_att, w, encoder_outputs)  # B x D_att

        x = torch.cat((h_att, att), dim=1)  # B x (D_att+D_att)

        for idx, rnn in enumerate(self.decoder_rnn_list):
            x, h_dec[idx] = rnn(x, h_dec[idx])  # B x D_rnn

        s = self.fc_stop(x)  # B x 1
        out = self.fc_mel(x).view(-1, self.r, self.dim_mel)  # B x r x D_mel

        return s, out, w, h_att, h_dec


class Decoder(nn.Module):
    def __init__(self, decoder_cell):
        super().__init__()
        self.decoder_cell = decoder_cell
        self.r = self.decoder_cell.r
        self.stop_threshold = 0.9

    def forward(self, encoder_outputs, x):
        # encoder_outputs:  B x L x D_enc
        # x:                B x T x D_mel
        B = encoder_outputs.shape[0]  # batch size
        L = encoder_outputs.shape[1]  # text length
        dtype = encoder_outputs.dtype
        device = encoder_outputs.device

        w_t, h_att_t, h_dec_t = self.decoder_cell.initial_state(B, L, dtype, device)

        y_t = torch.zeros(
            (B, self.r, self.decoder_cell.dim_mel), dtype=dtype, device=device
        )  # GO frame B x r x D_mel
        x_split = None if x is None else x.split(self.r, dim=1)

        y, w = [], []
        idx = 0
        while True:
            s_t, y_t, w_t, h_att_t, h_dec_t = self.decoder_cell(
                y_t, w_t, h_att_t, h_dec_t, encoder_outputs
            )

            y.append(y_t)
            w.append(w_t)
            if x_split is not None:
                # Force teacher inputs
                y_t = x_split[idx]  # B x r x D_mel
            idx += 1
            if x_split is not None and idx >= len(x_split):
                break
            elif torch.all(s_t > self.stop_threshold):
                break

        y = torch.cat(y, dim=1)  # B x T x D_mel
        w = torch.stack(w, dim=1)  # B x T x L

        return y, w
