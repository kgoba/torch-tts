import torch
import torch.nn as nn
from modules import PreNet, CBHG


class Encoder(nn.Module):
    def __init__(self, alphabet_size):
        super().__init__()
        dim_emb = 256
        dim_pre = 128
        dim_rnn = 128
        dim_out = 128
        self.emb = nn.Embedding(alphabet_size, dim_emb, padding_idx=0)
        self.pre_net = PreNet(dim_emb, dim_pre)
        self.cbhg = CBHG(dim_pre, dim_out, dim_rnn=dim_rnn)

    def forward(self, x):
        # x = B x T
        x = self.emb(x)  # B x T x D_emb
        x = self.pre_net(x)  # B x T x D_pre
        x = self.cbhg(x)  # B x T x D_rnn
        return x
