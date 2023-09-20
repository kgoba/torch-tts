import torch
import torch.nn as nn

from modules.attention import MultiHeadAttention


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]

    The reference encoder is made up of a convolutional stack, followed by an RNN.
    It takes as input a log-mel spectrogram, which is first passed to a stack of
    six 2-D convolutional layers with 3×3 kernel, 2×2 stride, batch normalization and
    ReLU activation function. We use 32, 32, 64, 64, 128 and 128 output channels for
    the 6 convolutional layers, respectively. The resulting output tensor is then
    shaped back to 3 dimensions (preserving the output time resolution) and fed to
    a single-layer 128-unit unidirectional GRU. The last GRU state serves as
    the reference embedding, which is then fed as input to the style token layer.
    """

    def __init__(self, num_mels=80, dim_out=128, ref_enc_filters=[32, 32, 64, 64, 128, 128]):
        super().__init__()

        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)]
        )

        out_channels = self.calculate_channels(num_mels, 3, 2, 1, K)
        self.gru = nn.LSTM(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=dim_out,
            batch_first=True,
        )
        # self.num_mels = num_mels

    def forward(self, inputs, input_lengths=None):
        out = inputs.unsqueeze(1)
        # out = inputs.view(inputs.size(0), 1, -1, self.num_mels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = nn.functional.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            input_lengths = (input_lengths / 2 ** len(self.convs)).long()
            input_lengths = torch.clip(input_lengths, min=1)
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out[0].squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(self, dim_query=128, num_tokens=10, dim_emb=256, num_heads=4):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(num_tokens, dim_emb // num_heads))
        self.attention = MultiHeadAttention(
            query_dim=dim_query,
            key_dim=dim_emb // num_heads,
            num_units=dim_emb,
            num_heads=num_heads,
        )

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        """
        Similarly, the text encoder state uses a tanh activation; we found that applying a
        tanh activation to GSTs before applying attention led to greater token diversity.
        """
        keys = (
            torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        )  # [N, num_tokens, token_embedding_size//num_heads]
        style_embed = self.attention(query, keys)
        # print(self.embed)

        return style_embed


class GST(nn.Module):
    def __init__(self, num_mels=80, dim_emb=256, dim_enc=128, num_tokens=10, num_heads=4):
        super().__init__()
        self.encoder = ReferenceEncoder(num_mels=num_mels, dim_out=dim_enc)
        self.stl = STL(num_tokens=num_tokens, dim_emb=dim_emb, num_heads=num_heads)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed = self.stl(enc_out)

        return style_embed, {}


class VAE(nn.Module):
    def __init__(self, num_mels=80, dim_emb=256, dim_enc=128, dim_vae=16):
        super().__init__()
        self.encoder = ReferenceEncoder(num_mels=num_mels, dim_out=dim_enc)

        self.mean_linear = nn.Linear(dim_enc, dim_vae)
        self.logvar_linear = nn.Linear(dim_enc, dim_vae)
        self.fc_out = nn.Linear(dim_vae, dim_emb, bias=False)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        z_mean = self.mean_linear(enc_out)
        z_logvar = self.logvar_linear(enc_out)
        eps = torch.randn_like(z_mean)
        z = eps * torch.exp(0.5 * z_logvar) + z_mean

        kl_loss = -(1 + z_logvar - z_mean * z_mean - z_logvar.exp()) / 2
        # kl_loss = torch.Tensor([float(0)])
        x = torch.tanh(self.fc_out(z).unsqueeze(1))
        if not self.training:
            # print(x)
            print(f"VST mean(std)={torch.exp(0.5 * z_logvar).mean().item()}")
            print(f"VST mean(mean)={z_mean.mean().item()}")
            print(f"VST rms(mean)={z_mean.pow(2).mean().sqrt().item()}")

        return x, {"kl": kl_loss}


class GST_VAE(nn.Module):
    def __init__(
        self, num_mels=80, dim_emb=256, dim_enc=128, num_tokens=10, num_heads=4, dim_vae=32
    ):
        super().__init__()
        self.encoder = ReferenceEncoder(num_mels=num_mels, dim_out=dim_enc)
        self.stl = STL(num_tokens=num_tokens, dim_emb=dim_emb, num_heads=num_heads)

        self.mean_linear = nn.Linear(dim_emb, dim_vae)
        self.logvar_linear = nn.Linear(dim_emb, dim_vae)
        self.fc_out = nn.Linear(dim_vae, dim_emb, bias=False)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed = self.stl(enc_out)

        z_mean = self.mean_linear(style_embed)
        z_logvar = self.logvar_linear(style_embed)
        eps = torch.randn_like(z_mean)
        z = eps * torch.exp(0.5 * z_logvar) + z_mean

        kl_loss = -(1 + z_logvar - z_mean * z_mean - z_logvar.exp()) / 2
        x = self.fc_out(z)
        return x, {"kl": kl_loss}
