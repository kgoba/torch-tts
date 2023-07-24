import torch
import torch.nn as nn
from modules import PreNet, CBHG


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
    '''
    The encoder converts a character sequence into a hidden feature
    representation which the decoder consumes to predict a spectrogram.
    Input characters are represented using a learned 512-dimensional
    character embedding, which are passed through a stack of 3 convolu-
    tional layers each containing 512 filters with shape 5 × 1, i.e., where
    each filter spans 5 characters, followed by batch normalization [18]
    and ReLU activations. As in Tacotron, these convolutional layers
    model longer-term context (e.g., N -grams) in the input character
    sequence. The output of the final convolutional layer is passed into a
    single bi-directional [19] LSTM [20] layer containing 512 units (256
    in each direction) to generate the encoded features.
    '''
    def __init__(self, alphabet_size, dim_out=128, dim_emb=256):
        super().__init__()

        self.emb = nn.Embedding(alphabet_size, dim_emb, padding_idx=0)
        self.conv = nn.Sequential()
        for _ in range(3):
            self.conv.extend(
                [
                    nn.Conv1d(
                        dim_emb,
                        dim_emb,
                        kernel_size=5,
                        padding=2
                    ),
                    nn.BatchNorm1d(dim_emb),
                    nn.ReLU(),
                ]
            )
        self.rnn = nn.LSTM(dim_emb, dim_out // 2, batch_first=True, bidirectional=True)

    def forward(self, x, xmask):
        # x = B x T
        x = self.emb(x)  # B x T x D_emb
        xc = self.conv(x.mT).mT
        x = xc + x
        x, _ = self.rnn(x)  # B x T x D_rnn
        return x
