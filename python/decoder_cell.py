import torch
import torch.nn as nn
from modules.activations import isru
from modules.modules import PreNet, HighwayLayer, GRUCellFixed
from modules.rnn import LSTMZoneoutCell, ResGRUCell
from modules.attention import ContentMarkovAttention, StepwiseMonotonicAttention


def initial_weights(batch_size, memory_size):
    w_0 = torch.cat(
        [
            torch.ones((batch_size, 1), requires_grad=False),
            torch.zeros((batch_size, memory_size - 1), requires_grad=False),
        ],
        dim=1,
    )  # Initial attention weights B x L
    return w_0


class Taco1DecoderCell(nn.Module):
    def __init__(self, dim_ctx, dim_mel, r, dim_pre=128, dim_att=256, num_layers=2, p_zoneout=0.1):
        super().__init__()
        self.dim_rnn = dim_att + dim_ctx

        self.pre_net = PreNet(r * dim_mel, dim_pre, p_dropout=0.5, always_dropout=True)
        # self.attention_module = ContentGeneralAttention(dim_ctx, dim_att)
        self.attention_module = ContentMarkovAttention(dim_ctx, dim_att)
        self.attention_rnn = GRUCellFixed(dim_pre + dim_ctx, dim_att, p_zoneout=0.1)
        self.decoder_rnn_list = nn.ModuleList(
            [ResGRUCell(self.dim_rnn, p_zoneout=p_zoneout) for _ in range(num_layers)]
        )

    def initial_state(self, batch_size, memory_size, dtype, device):
        w_0 = initial_weights(batch_size, memory_size, dtype, device)
        h_att_0 = torch.zeros((batch_size, self.dim_att), dtype=dtype, device=device)
        h_dec_0 = [
            torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device)
            for _ in range(self.decoder_rnn_list)
        ]
        return w_0, h_att_0, h_dec_0

    def forward(self, x, dec_state, memory, mmask):
        # x:    B x r x D_mel
        # w:    B x L
        # h_att:B x D_att
        # h_dec:B x D_dec
        w, h_att, h_dec = dec_state

        x_pre = self.pre_net(x.flatten(1, 2))  # B x D_pre

        ctx_att = torch.bmm(w.unsqueeze(1), memory).squeeze(1)  # B x D_ctx
        x_att = torch.cat((ctx_att, x_pre), dim=1)
        h_att = self.attention_rnn(x_att, h_att)  # B x D_att
        w = self.attention_module(h_att, w, memory)  # B x D_enc

        x_dec = torch.cat((h_att, ctx_att), dim=1)  # B x (D_att+D_ctx)
        for idx, rnn in enumerate(self.decoder_rnn_list):
            x_dec, h_dec[idx] = rnn(x_dec, h_dec[idx])  # B x D_rnn

        dec_state = w, h_att, h_dec
        return x_dec, dec_state


class Taco2DecoderCell(nn.Module):
    def __init__(self, dim_ctx, dim_mel, r, dim_rnn, dim_pre=128, dim_att=128, p_zoneout=0.1):
        super().__init__()
        self.dim_output = dim_rnn[-1] + dim_ctx + dim_pre

        self.pre_net = PreNet(dim_mel, dim_pre, always_dropout=True, p_dropout=0.5)
        # self.attention_module = ContentMarkovAttention(dim_ctx, dim_att)
        self.attention_module = StepwiseMonotonicAttention(dim_ctx, dim_rnn[0] + dim_rnn[1])
        # self.attention_fc = nn.Linear(dim_rnn[0], dim_att)

        rnn_dims = [dim_pre] + dim_rnn
        rnn_dims_zipped = zip(rnn_dims[:-1], rnn_dims[1:])
        self.decoder_rnn_list = nn.ModuleList(
            [
                LSTMZoneoutCell(dim_in + dim_ctx, dim_hidden, p_zoneout=p_zoneout)
                for dim_in, dim_hidden in rnn_dims_zipped
            ]
        )
        self.initial_decoder_h = nn.ParameterList(
            [torch.zeros(1, dim_hidden) for dim_hidden in dim_rnn]
        )
        self.initial_decoder_c = nn.ParameterList(
            [torch.zeros(1, dim_hidden) for dim_hidden in dim_rnn]
        )

    def initial_state(self, batch_size, memory_size, dtype, device):
        # type: (int, int, torch.dtype, torch.device) -> Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]
        w_0 = initial_weights(batch_size, memory_size).to(dtype=dtype, device=device)
        h_dec_0 = [
            (
                h.to(dtype=dtype, device=device).expand(batch_size, -1),
                c.to(dtype=dtype, device=device).expand(batch_size, -1),
            )
            for h, c in zip(self.initial_decoder_h, self.initial_decoder_c)
        ]
        return w_0, h_dec_0

    def forward(self, x, dec_state, memory, mmask):
        # type: (Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]], Tensor, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]]
        # x:    B x r x D_mel
        # w:    B x L
        # memory: B x L x D_ctx
        w, h_dec = dec_state[0], dec_state[1]

        x_pre = self.pre_net(x.flatten(1, 2))  # B x D_pre

        # sharpen attention weights
        # w_adj = torch.pow(w, 1)
        # w_adj = w_adj / (1e-6 + w_adj.sum(dim=1).unsqueeze(1))
        ctx_att = torch.bmm(w.unsqueeze(1), memory).squeeze(1)  # B x D_ctx

        x_dec = x_pre
        for idx, rnn in enumerate(self.decoder_rnn_list):
            x_dec = torch.cat((x_dec, ctx_att), dim=1)
            h_dec[idx] = rnn(x_dec, h_dec[idx])  # B x D_rnn
            # x_dec = torch.cat((h_dec[idx][0], ctx_att), dim=1)
            x_dec = h_dec[idx][0]

        x_att = torch.cat([h_dec[-1][0], h_dec[0][0]], dim=1) # isru(self.attention_fc(h_dec[0][0]))
        w = self.attention_module(x_att, w, memory, mmask)  # B x L

        x_dec = torch.cat((x_dec, ctx_att, torch.zeros_like(x_pre)), dim=1)
        # x_dec = torch.cat((x_dec, x_pre), dim=1)

        dec_state = w, h_dec
        return x_dec, ctx_att, dec_state


class DecoderCell(nn.Module):
    def __init__(self, dim_ctx, dim_mel, r, dim_pre=128, dim_att=128, num_layers=1, p_zoneout=0.1):
        super().__init__()
        self.dim_rnn = dim_pre + dim_ctx

        self.pre_net = PreNet(r * dim_mel, dim_pre, always_dropout=True)
        # self.attention_module = ContentMarkovAttention(dim_ctx, dim_att)
        self.attention_module = ContentMarkovAttention(dim_ctx, self.dim_rnn)
        # self.attention_rnn = nn.LSTMCell(self.dim_rnn + dim_ctx, dim_att)
        # self.attention_fc = nn.Linear(self.dim_rnn, dim_att, bias=False)
        self.highway_list = nn.ModuleList(
            [HighwayLayer(self.dim_rnn, self.dim_rnn, activation=isru) for _ in range(3)]
        )
        self.decoder_rnn_list = nn.ModuleList(
            [
                LSTMZoneoutCell(self.dim_rnn, self.dim_rnn, p_zoneout=p_zoneout)
                for _ in range(num_layers)
            ]
        )

    def initial_state(self, batch_size, memory_size, dtype, device):
        # type: (int, int, torch.dtype, torch.device) -> Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]
        w_0 = initial_weights(batch_size, memory_size, dtype, device)
        h_dec_0 = [
            (
                torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device),
                torch.zeros((batch_size, self.dim_rnn), dtype=dtype, device=device),
            )
            for _ in self.decoder_rnn_list
        ]  # Initial decoder RNN cell states B x D_rnn
        return w_0, h_dec_0

    def forward(self, x, dec_state, memory, mmask):
        # type: (Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]], Tensor, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]]
        # x:    B x r x D_mel
        # w:    B x L
        # memory: B x L x D_ctx
        w, h_dec = dec_state[0], dec_state[1]

        x_pre = self.pre_net(x.flatten(1, 2))  # B x D_pre

        ctx_att = torch.bmm(w.unsqueeze(1), memory).squeeze(1)  # B x D_ctx

        x_dec = torch.cat((x_pre, ctx_att), dim=1)  # B x (D_pre+D_ctx)
        for idx, rnn in enumerate(self.decoder_rnn_list):
            h_dec[idx] = rnn(x_dec, h_dec[idx])  # B x D_rnn
            x_dec = h_dec[idx][0]  # + x_dec

        # x_att = torch.cat((x_dec, ctx_att), dim=1)
        # h_att = self.attention_rnn(x_att, h_att)  # B x D_att
        # x_att = self.attention_fc(x_dec)
        x_att = x_dec
        w = self.attention_module(x_att, w, memory, mmask)  # B x L

        for idx, hw_layer in enumerate(self.highway_list):
            x_dec = hw_layer(x_dec)

        dec_state = w, h_dec
        return x_dec, dec_state
