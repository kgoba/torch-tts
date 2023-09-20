import torch
import torch.nn as nn
from modules.activations import isru_sigmoid

# class ContentAttention(nn.Module):
#     """Based on Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
#     https://arxiv.org/abs/1508.04025v5
#     """

#     def __init__(self):
#         super().__init__()

#     def forward(self, score, context):
#         # score:   B x L
#         # context: B x L x D_ctx
#         w = torch.softmax(score, dim=1)  # B x L
#         c = torch.bmm(w.unsqueeze(1), context)  # (B x 1 x L) x (B x L x D_ctx) -> (B x 1 x D_ctx)
#         c = c.squeeze(1)  # B x D_ctx
#         return w  # B x D_ctx, B x L


class ContentConcatAttention(nn.Module):
    def __init__(self, dim_context, dim_input, dim_hidden):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim_context + dim_input, dim_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1, bias=False),
        )

    def forward(self, x, w, context, cmask=None):
        # x: B x D_input
        x = x.unsqueeze(1)  # B x 1 x D_input
        score = self.score_net(torch.cat((context, x), dim=2)).squeeze(2)  # B x L
        w = torch.softmax(score, dim=1)  # B x L
        return w


class ContentGeneralAttention(nn.Module):
    def __init__(self, dim_context, dim_input):
        super().__init__()
        self.score_net = nn.Linear(dim_input, dim_context)

    def forward(self, x, w, context, cmask=None):
        # x: B x D_input
        # context: B x L x D_ctx
        x = self.score_net(x).unsqueeze(2)  # B x D_ctx x 1
        score = torch.bmm(context, x).squeeze(2)  # B x L
        w = torch.softmax(score, dim=1)  # B x L
        return w


class ContentMarkovAttention(nn.Module):
    def __init__(self, dim_context, dim_input, num_probs=3):
        super().__init__()
        self.num_probs = num_probs
        self.fc_query = nn.Linear(dim_input, num_probs * dim_context, bias=False)
        # self.fc_query = nn.Linear(dim_input, dim_context, bias=False)
        self.v = nn.Linear(dim_context, 3, bias=False)
        self.w = nn.Linear(dim_input, 3, bias=False)

    def forward(self, x, w, context, cmask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tensor
        # x: B x D_input
        # w: B x L
        # context: B x L x D_ctx
        # cmask: B x L
        q = self.fc_query(x)
        q = q.view(q.shape[0], -1, self.num_probs)  # B x D_ctx x 3
        e = torch.bmm(context, q)  # B x L x 3
        # e = self.v((context + q.unsqueeze(1)).tanh())
        # e = e + self.v(context) - self.w(x).unsqueeze(1)

        # mask each probability according to context length per batch item
        if cmask is not None:
            cmask_inv = ~cmask
            cmask_extended = torch.stack(
                [cmask_inv.roll(-n, dims=1) for n in range(self.num_probs)],
                dim=2,
            )
            e.masked_fill_(cmask_extended, -1e12)

        for n in range(1, self.num_probs):
            e[:, -n:, n] = -1e12
        p = e.softmax(dim=2)

        wp = w.unsqueeze(2) * p  # B x L x 3
        # w = wp[:, :, 0] + wp[:, :, 1].roll(1, dims=1) + wp[:, :, 2].roll(2, dims=1)
        w = wp[:, :, 0]  # B x L
        for n in range(1, self.num_probs):
            w[:, n:] += wp[:, :-n, n]

        return w


class StepwiseMonotonicAttention(nn.Module):
    def __init__(self, dim_context, dim_input, sigmoid_noise=1.0):
        super().__init__()
        self.sigmoid_noise = sigmoid_noise
        self.query_layer = nn.Linear(dim_input, dim_context, bias=False)
        self.bias = nn.Parameter(torch.Tensor([1.0]))
        # self.v = nn.Linear(dim_context, 1, bias=False)

    def forward(self, x, w, context, cmask=None):
        q = self.query_layer(x)  # B x D_ctx
        # e = self.v(torch.tanh(q.unsqueeze(1) + context))
        e = torch.bmm(context, q.unsqueeze(2))
        e = e.squeeze(2)
        # e = e + self.bias

        if self.training:
            e += self.sigmoid_noise * torch.randn_like(e)

        # if cmask is not None:
        #     e.masked_fill_(~cmask, 1e12)

        e[:, -1] = 1e12
        p0 = isru_sigmoid(e)  # e.sigmoid()
        w0 = w * p0
        w1 = w * (1 - p0)

        w = w0
        w[:, 1:] += w1[:, :-1]

        return w


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.d_gain = 1 / (key_dim**0.5)

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, key_mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(
            torch.split(querys, split_size, dim=2), dim=0
        )  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(
            torch.split(values, split_size, dim=2), dim=0
        )  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = self.d_gain * torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        if key_mask != None:
            scores.masked_fill_(~(key_mask.unsqueeze(0).unsqueeze(2)), -1e6)
        scores = nn.functional.softmax(scores, dim=3)
        if not self.training:
            print(
                f"MHA score: max={scores.max().item():.3f} T_q={query.size(1)} T_k={key.size(1)} N={key.size(0)}"
            )

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
