import torch
import torch.nn as nn


class PreNet(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(dim_input, dim_input),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim_input, dim_output),
            nn.ReLU(),
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
        dim_rnn=128,
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
                    nn.BatchNorm1d(dim_conv_hidden),
                    nn.ReLU(),
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
            # nn.BatchNorm1d(dim_input),
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
        self.rnn = nn.GRU(
            input_size=dim_highway, hidden_size=dim_rnn, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(2 * dim_rnn, dim_output)
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
        x = self.fc(x)  # B x T x D_out
        return x


class ResGRUCell(nn.GRUCell):
    def __init__(self, input_size, bias=True):
        super().__init__(input_size, input_size, bias=bias)

    def forward(self, x, h):
        h = super().forward(x, h)
        return x + h, h


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
        # score:   B x L_ctx
        # context: B x L_ctx x D_ctx
        w = torch.softmax(score, dim=1)  # B x L_ctx
        c = torch.bmm(
            w.unsqueeze(1), context
        )  # (B x 1 x L_ctx) x (B x L_ctx x D_ctx) -> (B x 1 x D_ctx)
        c = c.squeeze(1)  # B x D_ctx
        return c, w  # B x D_ctx, B x L_ctx


class ContentConcatAttention(ContentAttention):
    def __init__(self, dim_context, dim_input, dim_hidden):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim_context + dim_input, dim_hidden, bias=False),
            nn.Tanh(),
            torch.nn.Linear(dim_hidden, 1, bias=False),
        )

    def forward(self, x, w, context):
        # x: B x D_input
        x = x.unsqueeze(1)  # B x 1 x D_input
        score = self.score_net(torch.cat((context, x), dim=2)).squeeze(2)  # B x L_ctx
        return super().forward(score, context)


class ContentGeneralAttention(ContentAttention):
    def __init__(self, dim_context, dim_input):
        super().__init__()
        self.score_net = nn.Linear(dim_input, dim_context)

    def forward(self, x, w, context):
        # x: B x D_input
        # context: B x L_ctx x D_ctx
        x = self.score_net(x).unsqueeze(2)  # B x D_ctx x 1
        score = torch.bmm(context, x).squeeze(2)  # B x L_ctx
        return super().forward(score, context)
