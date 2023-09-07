import torch
import torch.nn as nn
from mps_fixes.mps_fixes import Conv1dFix, GRUCellFixed


class ResGRUCell(GRUCellFixed):
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
            h = self.p_zoneout * hidden[0] + (1. - self.p_zoneout) * h
            c = self.p_zoneout * hidden[1] + (1. - self.p_zoneout) * c
            pass
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