import torch
import torch.nn as nn
from modules.modules import PreNet, CBHG
from modules.activations import ISRLU
from modules.attention import MultiHeadAttention
from modules.rnn import reverse_padded, BiDiLSTM, BiDiLSTMSplit
from mps_fixes.mps_fixes import Conv1dFix
from data.util import lengths_to_mask

class Encoder(nn.Module):
    def __init__(self, alphabet_size, dim_out=256, dim_emb=256):
        super().__init__()
        dim_pre = 128
        self.emb = nn.Embedding(alphabet_size, dim_emb, padding_idx=0)
        self.pre_net = PreNet(dim_emb, dim_pre)
        self.cbhg = CBHG(dim_pre, dim_out)

    def forward(self, x, xmask):
        # x = B x T
        x = self.emb(x)  # B x T x D_emb
        x = self.pre_net(x)  # B x T x D_pre
        x = self.cbhg(x)  # B x T x D_rnn
        return x


class Encoder2(nn.Module):
    """
    The encoder converts a character sequence into a hidden feature
    representation which the decoder consumes to predict a spectrogram.
    Input characters are represented using a learned 512-dimensional
    character embedding, which are passed through a stack of 3 convolu-
    tional layers each containing 512 filters with shape 5 × 1, i.e., where
    each filter spans 5 characters, followed by batch normalization [18]
    and ReLU activations. As in Tacotron, these convolutional layers
    model longer-term context (e.g., N-grams) in the input character
    sequence. The output of the final convolutional layer is passed into a
    single bi-directional [19] LSTM [20] layer containing 512 units (256
    in each direction) to generate the encoded features.
    """

    def __init__(self, alphabet_size, dim_out=512, dim_emb=512):
        super().__init__()

        self.dim_out = dim_out
        self.dim_emb = dim_emb
        self.emb = nn.Embedding(alphabet_size, dim_emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(dim_emb, dim_emb, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(dim_emb),
            ISRLU(),
            nn.Conv1d(dim_emb, dim_emb, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(dim_emb),
            ISRLU(),
            nn.Conv1d(dim_emb, dim_emb, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(dim_emb, affine=False),
            ISRLU(),
        )
        # self.ln = nn.LayerNorm([self.dim_emb])

        self.rnn = BiDiLSTMSplit(dim_emb * 2, dim_out // 2, bias=False)
        self.rnn_h0 = nn.Parameter(torch.zeros(1, 1, dim_out))
        self.rnn_c0 = nn.Parameter(torch.zeros(1, 1, dim_out))

        # self.mha = MultiHeadAttention(dim_out, dim_out, dim_out, num_heads=8)

    def forward(self, x, x_lengths):
        # x = B x T
        x = self.emb(x)  # B x T x D_emb
        xc = self.conv(x.mT).mT
        # x = xc + x
        x = torch.cat((xc, x), dim=2)
        x = nn.functional.dropout(x, p=0.1, training=self.training)

        B = x.shape[0]
        h0 = self.rnn_h0.expand(-1, B, -1)
        c0 = self.rnn_c0.expand(-1, B, -1)
        x, hn = self.rnn(x, x_lengths, h0, c0)

        # x_mask = lengths_to_mask(x_lengths)
        # x = x + self.mha(x, x, x_mask)
        return x
