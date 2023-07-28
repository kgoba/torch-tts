import torch
import torch.nn as nn


class PreNet(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=256):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
            nn.Tanh(),
            nn.Dropout(),
        )

    def forward(self, x):
        return self.pre_net(x)


class HighwayLayer(nn.Module):
    def __init__(self, dim_input, dim_output, activation=nn.functional.relu):
        super().__init__()
        self.H = nn.Linear(dim_input, dim_output)
        self.T = nn.Linear(dim_input, dim_output)
        self.activation = activation

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        y = self.activation(self.H(x))
        return (y * t) + (x * (1 - t))


class CBHG(nn.Module):
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
                        dim_input, dim_conv_hidden, kernel_size=k, padding=k // 2, bias=False
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
            nn.ReLU(),
            nn.BatchNorm1d(dim_proj_hidden),
            nn.Conv1d(dim_proj_hidden, dim_input, padding=1, kernel_size=3),
        )
        self.highway = nn.Sequential(
            nn.Linear(dim_input, dim_highway, bias=False)
            if dim_input != dim_highway
            else nn.Identity(),
            HighwayLayer(dim_highway, dim_highway, nn.functional.relu),
            HighwayLayer(dim_highway, dim_highway, nn.functional.relu),
            HighwayLayer(dim_highway, dim_highway, nn.functional.relu),
            HighwayLayer(dim_highway, dim_highway, nn.functional.relu),
        )
        dim_rnn = dim_output // 2
        self.rnn = nn.GRU(
            input_size=dim_highway, hidden_size=dim_rnn, bidirectional=True, batch_first=True
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


class ResGRUCell(nn.GRUCell):
    def __init__(self, input_size, bias=True):
        super().__init__(input_size, input_size, bias=bias)

    def forward(self, x, h):
        h = super().forward(x, h)
        return x + h, h


class ResLSTMCell(nn.Module):
# class ResLSTMCell(nn.RNNCellBase):
    # def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
    #              device=None, dtype=None) -> None:
    #     factory_kwargs = {'device': device, 'dtype': dtype}
    #     super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def __init__(self, input_size, hidden_size):
        super(ResLSTMCell, self).__init__()
        # self.register_buffer("input_size", torch.Tensor([input_size]))
        # self.register_buffer("hidden_size", torch.Tensor([hidden_size]))
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
        # self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

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
        return hy, (hy, cy)


class PostNet(nn.Module):
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


class ContentAttention(nn.Module):
    """Based on Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
    https://arxiv.org/abs/1508.04025v5
    """

    def __init__(self):
        super().__init__()

    def forward(self, score, context):
        # score:   B x L
        # context: B x L x D_ctx
        w = torch.softmax(score, dim=1)  # B x L
        c = torch.bmm(w.unsqueeze(1), context)  # (B x 1 x L) x (B x L x D_ctx) -> (B x 1 x D_ctx)
        c = c.squeeze(1)  # B x D_ctx
        return c, w  # B x D_ctx, B x L


class ContentConcatAttention(ContentAttention):
    def __init__(self, dim_context, dim_input, dim_hidden):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim_context + dim_input, dim_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1, bias=False),
        )

    def forward(self, x, w, context):
        # x: B x D_input
        x = x.unsqueeze(1)  # B x 1 x D_input
        score = self.score_net(torch.cat((context, x), dim=2)).squeeze(2)  # B x L
        return super().forward(score, context)


class ContentGeneralAttention(ContentAttention):
    def __init__(self, dim_context, dim_input):
        super().__init__()
        self.score_net = nn.Linear(dim_input, dim_context)

    def forward(self, x, w, context):
        # x: B x D_input
        # context: B x L x D_ctx
        x = self.score_net(x).unsqueeze(2)  # B x D_ctx x 1
        score = torch.bmm(context, x).squeeze(2)  # B x L
        return super().forward(score, context)


class ContentMarkovAttention(nn.Module):
    def __init__(self, dim_context, dim_input):
        super().__init__()
        self.prob_scores = nn.Linear(dim_input, 3 * dim_context, bias=False)

    def forward(self, x, w, context, cmask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tensor
        # x: B x D_input
        # w: B x L
        # context: B x L x D_ctx
        # cmask: B x L
        x = self.prob_scores(x).tanh()
        x = x.view(x.shape[0], -1, 3)  # B x D_ctx x 3
        x = torch.bmm(context, x)  # B x L x 3

        # mask each probability according to context length per batch item
        if cmask is not None:
            cmask_extended = torch.stack(
                [
                    torch.ones_like(cmask),
                    cmask & cmask.roll(-1, dims=1),
                    cmask & cmask.roll(-2, dims=1),
                ],
                dim=2,
            )
            x[~cmask_extended] = -1e12
        else:
            x[:, -1:, 1] = -1e12
            x[:, -2:, 2] = -1e12
        x = x.softmax(dim=2)

        wp = w.unsqueeze(2) * x  # B x L x 3
        # w = wp[:, :, 0] + wp[:, :, 1].roll(1, dims=1) + wp[:, :, 2].roll(2, dims=1)
        w = wp[:, :, 0]  # B x L
        w[:, 1:] += wp[:, :-1, 1]
        w[:, 2:] += wp[:, :-2, 2]

        return w
