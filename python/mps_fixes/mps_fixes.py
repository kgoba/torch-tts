import torch
import torch.nn as nn
import math


class Conv1dFix(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros([out_channels, in_channels, kernel_size]))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros([out_channels]))
            nn.init.normal_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, self.padding))  # N, C, L
        x = torch.cat([x.roll(n - self.padding, dims=2) for n in range(self.kernel_size)], dim=1)
        x = x[:, :, self.padding : -self.padding]
        x = torch.matmul(self.weight.view(self.out_channels, -1), x)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(-1)
        return x


class GRUCellFixed(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, p_zoneout=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.p_zoneout = p_zoneout
        scale = math.sqrt(1 / hidden_size)
        self.weight_ii = nn.Parameter(scale * torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(scale * torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(scale * torch.randn(3 * hidden_size))
        self.bias_hh = nn.Parameter(scale * torch.randn(3 * hidden_size))

    def forward(self, input, hidden):
        a = torch.mm(input, self.weight_ii.mT) + self.bias_ii
        b = torch.mm(hidden, self.weight_hh.mT) + self.bias_hh
        r_i, z_i, n_i = torch.split(a, self.hidden_size, dim=-1)
        r_h, z_h, n_h = torch.split(b, self.hidden_size, dim=-1)
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + r * n_h)
        h = (1 - z) * n + z * hidden
        if self.p_zoneout and self.training:
            zoneout = torch.rand(self.hidden_size, device=hidden.device) < self.p_zoneout
            return torch.where(zoneout, hidden, h)
        else:
            return h
