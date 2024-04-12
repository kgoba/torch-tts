import torch
import torch.nn as nn
from mps_fixes.mps_fixes import Conv1dFix, GRUCellFixed
from modules.activations import ISRU, isru


class Transposition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mT


class PreNet(nn.Module):
    """
    FC-256-ReLU → Dropout(0.5) → FC-128-ReLU → Dropout(0.5)
    """

    def __init__(
        self,
        dim_input,
        dim_output,
        dim_hidden=256,
        activation=nn.functional.relu,
        p_dropout=0.5,
        always_dropout=False,
    ):
        super().__init__()
        self.activation = activation
        self.p_dropout = p_dropout
        self.always_dropout = always_dropout
        self.layers = nn.ModuleList(
            [nn.Linear(dim_input, dim_hidden), nn.Linear(dim_hidden, dim_output)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = torch.dropout(x, self.p_dropout, self.always_dropout or self.training)
        return x


class HighwayLayer(nn.Module):
    def __init__(self, dim_input, activation=nn.functional.relu):
        super().__init__()
        self.H = nn.Linear(dim_input, dim_input)
        self.T = nn.Linear(dim_input, dim_input)
        self.activation = activation

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        y = self.activation(self.H(x))
        return (y * t) + (x * (1 - t))


class CBHG(nn.Module):
    """
    Encoder CBHG:
    * Conv1D bank: K=16, conv-k-128-ReLU
    * Max pooling: stride=1, width=2
    * Conv1D projections: conv-3-128-ReLU → conv-3-128-Linear
    * Highway net: 4 layers of FC-128-ReLU
    * Bidirectional GRU: 128 cells
    """

    def __init__(
        self,
        dim_input,
        dim_output,
        dim_conv_hidden=128,
        dim_proj_hidden=128,
        dim_highway=128,
        K=16,
    ):
        super().__init__()
        self.conv_bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        dim_input,
                        dim_conv_hidden,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim_conv_hidden),
                )
                for k in range(1, 1 + K, 2)
            ]
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_proj = nn.Sequential(
            nn.Conv1d(
                len(self.conv_bank) * dim_conv_hidden,
                dim_proj_hidden,
                padding=1,
                kernel_size=3,
                bias=False,
            ),
            nn.BatchNorm1d(dim_proj_hidden),
            nn.ReLU(),
            nn.Conv1d(dim_proj_hidden, dim_input, padding=1, kernel_size=3),
        )
        self.highway = nn.Sequential(
            (
                nn.Linear(dim_input, dim_highway, bias=False)
                if dim_input != dim_highway
                else nn.Identity()
            ),
            HighwayLayer(dim_highway),
            HighwayLayer(dim_highway),
            HighwayLayer(dim_highway),
            HighwayLayer(dim_highway),
        )
        dim_rnn = dim_output // 2
        self.rnn = nn.GRU(
            input_size=dim_highway,
            hidden_size=dim_rnn,
            bidirectional=True,
            batch_first=True,
        )
        # self.fc = nn.Linear(2 * dim_rnn, dim_output)
        self.rnn.flatten_parameters()

    def forward(self, x):
        x_residual = x  # B x T x D_in
        x = x.transpose(1, 2)  # B x D_in x T
        x = [conv(x) for conv in self.conv_bank]  # [B x D_conv_h x T]
        x = torch.cat(x, dim=1)  # B x K*D_conv_h x T
        x = self.max_pool(x)  # B x K*D_in x T
        x = self.conv_proj(x).transpose(1, 2)  # B x T x D_in
        x = x + x_residual  # B x T x D_in
        x = self.highway(x)  # B x T x D_highway
        x, _ = self.rnn(x)  # B x T x (2*D_rnn)
        # x = self.fc(x)  # B x T x D_out
        return x


class Taco1PostNet(nn.Module):
    def __init__(self, dim_mel, dim_stft):
        super().__init__()
        dim_rnn = 256
        self.cbhg = nn.Sequential(
            CBHG(dim_mel, dim_rnn, dim_conv_hidden=64, dim_proj_hidden=128, K=8),
            nn.Linear(dim_rnn, dim_stft),
        )
        self.direct = nn.Linear(dim_mel, dim_stft, bias=False)

    def forward(self, x):
        return self.cbhg(x) + self.direct(x)


class MelPostnet(nn.Module):
    def __init__(self, dim_mel, dim_hidden=512, kernel_size=5, num_layers=3):
        super().__init__()

        padding = (kernel_size - 1) // 2
        conv_dims = [dim_mel] + [dim_hidden for _ in range(num_layers)]
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        ch_in,
                        ch_out,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm1d(ch_out),
                )
                for ch_in, ch_out in zip(conv_dims[:-1], conv_dims[1:])
            ]
        )
        self.fc_out = nn.Linear(dim_hidden, dim_mel, bias=False)

    def forward(self, x):
        x_conv = x.mT
        for layer in self.conv:
            x_conv = isru(layer(x_conv))
            x_conv = nn.functional.dropout(x_conv, p=0.1, training=self.training)

        return x + self.fc_out(x_conv.mT)


class MelPostnet2(nn.Module):
    def __init__(self, dim_in, dim_hidden=128, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    Transposition(),
                    Conv1dFix(dim_in, dim_hidden, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm1d(dim_hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    Conv1dFix(
                        dim_hidden, dim_hidden, kernel_size=5, padding=2, bias=False
                    ),
                    nn.BatchNorm1d(dim_hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    Conv1dFix(dim_hidden, dim_in, kernel_size=5, padding=2, bias=False),
                    Transposition(),
                    # nn.Linear(dim_hidden, dim_in),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv1 = Conv1dFix(dim_in, dim_in, kernel_size=3, padding=1, bias=False)
        self.conv2 = Conv1dFix(dim_in, dim_in, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # B x T x D_in
        y1 = self.conv1(x.mT).mT
        y2 = self.conv2(x.mT).mT
        return torch.flatten(torch.stack([y1, y2], dim=2), 1, 2)
