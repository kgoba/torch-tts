import torch
import torch.nn as nn
from modules import PreNet, ResGRUCell, ContentGeneralAttention


class DecoderCell(nn.Module):
    def __init__(self, dim_enc, dim_mel, r):
        super().__init__()
        self.dim_mel = dim_mel
        self.r = r
        decoder_depth = 2
        dim_pre = 128
        dim_att = 128
        dim_rnn = 2*dim_att
        self.attention_module = ContentGeneralAttention(dim_enc, dim_att)
        self.pre_net = PreNet(self.r * dim_mel, dim_pre)
        self.attention_rnn = nn.GRUCell(dim_pre, dim_att)
        self.decoder_rnns = [ResGRUCell(dim_rnn) for _ in range(decoder_depth)]
        self.fc = nn.Linear(dim_rnn, r*dim_mel)

    def forward(self, x, w, h_att, h_dec, encoder_outputs):
        # x: B x r x D_mel
        x = self.pre_net(x.flatten(1, 2)) # B x D_pre

        h_att = self.attention_rnn(x, h_att)                        # B x D_att
        att, w = self.attention_module(h_att, w, encoder_outputs)   # B x D_att
        x = torch.stack((h_att, att), dim=1)                        # B x (D_att+D_att)

        for idx, rnn in enumerate(self.decoder_rnns):
            x, h_dec[idx] = rnn(x, h_dec[idx])      # B x D_rnn
            
        x = self.fc(x)                              # B x (r*D_mel)
        x = x.view(-1, self.r, self.dim_mel)        # B x r x D_mel
        return x, w, h_att, h_dec


class DecoderTeacher(nn.Module):
    def __init__(self, decoder_cell):
        super().__init__()
        self.decoder_cell = decoder_cell
        self.r = self.decoder_cell.r

    def forward(self, x, encoder_outputs):
        # x: B x T x D_mel
        B = x.shape(0)

        h_att_t = torch.zeros((B, self.decoder_cell.dim_att), dtype=x.dtype, layout=x.layout, device=x.device)
        h_dec_t = [torch.zeros((B, self.decoder_cell.dim_rnn), dtype=x.dtype, layout=x.layout, device=x.device) for _ in range(self.decoder_cell.decoder_depth)]

        y_t = torch.zeros((B, self.r, self.decoder_cell.dim_mel), dtype=x.dtype, layout=x.layout, device=x.device) # GO frame
        w_t = torch.stack(
                    torch.ones((B, 1), dtype=x.dtype, layout=x.layout, device=x.device),
                    torch.zeros((B, encoder_outputs.shape(1) - 1), dtype=x.dtype, layout=x.layout, device=x.device),
                    dim=1)  # Initial attention weights
        y, w = [], []
        for x_t in x.tensor_split(self.r, dim=1):
            y_t, w_t, h_att_t, h_dec_t = self.decoder_cell(y_t, w_t, h_att_t, h_dec_t, encoder_outputs)
            y.append(y_t)
            w.append(w_t)
            # Force teacher inputs
            y_t = x_t # B x r x D_mel
        
        y = torch.stack(y, dim=1) # B x T x D_mel
        w = torch.stack(w, dim=1) # B x T x L_enc
        return y, w
