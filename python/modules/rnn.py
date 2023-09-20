import torch
import torch.nn as nn


def reverse_padded(x, x_lengths):
    x_rev_list = [x_[:x_l].flipud() for x_, x_l in zip(x, x_lengths)]
    return torch.nn.utils.rnn.pad_sequence(x_rev_list, batch_first=True)


class ResGRUCell(nn.GRUCell):
    def __init__(self, input_size, bias=True, p_zoneout=None):
        super().__init__(input_size, input_size, bias=bias, p_zoneout=p_zoneout)

    def forward(self, x, h):
        h = super().forward(x, h)
        return x + h, h


class LSTMZoneoutCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True, p_zoneout=None):
        super().__init__(input_size, hidden_size, bias=bias)
        self.p_zoneout = p_zoneout

    def forward(self, x, hidden):
        h, c = super().forward(x, hidden)
        if self.training:
            if self.p_zoneout:
                zoneout_h = torch.rand(self.hidden_size, device=h.device) < self.p_zoneout
                zoneout_c = torch.rand(self.hidden_size, device=c.device) < self.p_zoneout
                h = torch.where(zoneout_h, hidden[0], h)
                c = torch.where(zoneout_c, hidden[1], c)
        else:
            h = self.p_zoneout * hidden[0] + (1.0 - self.p_zoneout) * h
            c = self.p_zoneout * hidden[1] + (1.0 - self.p_zoneout) * c
        return h, c


class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        # self.reset_parameters()

    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        ifo_gates = (
            torch.mm(input, self.weight_ii.t())
            + self.bias_ii
            + torch.mm(hx, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(cx, self.weight_ic.t())
            + self.bias_ic
        )
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)

        cellgate = torch.mm(hx, self.weight_hh.t()) + self.bias_hh

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        if self.input_size == self.hidden_size:
            hy = outgate * (ry + input)
        else:
            hy = outgate * (ry + torch.mm(input, self.weight_ir.t()))
        return hy, cy


class BiDiLSTMSplit(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.rnn_f = nn.LSTM(input_size, hidden_size, batch_first=True, bias=bias)
        self.rnn_b = nn.LSTM(input_size, hidden_size, batch_first=True, bias=bias)

    def forward(self, x, x_lengths, h0, c0):
        # x: [B, T, input_size]
        T = x.shape[1]
        idx = torch.arange(0, T, device=x_lengths.device)
        mask = idx.unsqueeze(0) >= x_lengths.unsqueeze(1)

        f_h0, b_h0 = torch.chunk(h0, 2, dim=-1)
        f_c0, b_c0 = torch.chunk(c0, 2, dim=-1)
        x_f, (f_hn, _) = self.rnn_f(x, (f_h0, f_c0))  # B x T x D_rnn
        x_b, (b_hn, _) = self.rnn_b(reverse_padded(x, x_lengths), (b_h0, b_c0))  # B x T x D_rnn
        x = torch.cat((x_f, reverse_padded(x_b, x_lengths)), dim=2)
        x.masked_fill_(mask.unsqueeze(2), value=0)
        return x, torch.cat([f_hn, b_hn], dim=2)


class BiDiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, batch_first=True, bias=bias, bidirectional=True
        )

    def forward(self, x, x_lengths, h0, c0):
        x = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h0 = torch.cat(torch.chunk(h0, 2, dim=-1), dim=0)
        c0 = torch.cat(torch.chunk(c0, 2, dim=-1), dim=0)
        x, (h, _) = self.rnn(x, (h0, c0))  # B x T x D_rnn
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x, h
