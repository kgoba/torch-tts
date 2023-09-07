import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class VAE(nn.Module):
    def __init__(self, dim_input, dim_vae):
        super().__init__()
        self.fc_mu = nn.Linear(dim_input, dim_vae)
        self.fc_logvar = nn.Linear(dim_input, dim_vae)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        sigma = torch.exp(logvar / 2)
        # z_sampled = torch.normal(mu, sigma)
        z_sampled = mu + sigma * torch.randn_like(mu)
        # z_sampled = mu
        # print(mu, sigma, z_sampled)
        # print(mu[0], sigma[0])
        kl_loss = -(1 + logvar - mu*mu - logvar.exp()) / 2
        return z_sampled, kl_loss


class ReferenceEncoder(nn.Module):
    def __init__(self, dim_mel, dim_rnn=256, dim_conv=[512, 512], kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        conv_dims = [dim_mel] + dim_conv
        self.conv_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(dim_out),
                )
                for dim_in, dim_out in zip(conv_dims[:-1], conv_dims[1:])
            ]
        )
        self.rnn = nn.LSTM(dim_conv[-1], dim_rnn, batch_first=True)

    def forward(self, x, x_lengths=None):
        # x: [B, T, dim_mel]
        x = x.mT  # [B, dim_mel, T]
        for conv in self.conv_layer:
            x = conv(x).relu()  # [B, dim_conv, T]
        x = x.mT  # [B, T, dim_conv]

        x, (hn, cn) = self.rnn(x)  # [B, T, dim_rnn]
        # x = x.mean(axis=1)  # [B, dim_rnn]
        return x # , hn.squeeze(0)


class ReferenceEncoderVAE(nn.Module):
    def __init__(self, dim_mel, dim_out=16, dim_rnn=256, dim_conv=[512, 512], kernel_size=3):
        super().__init__()
        self.dim_out = dim_out
        self.encoder = ReferenceEncoder(
            dim_mel, dim_rnn=dim_rnn, dim_conv=dim_conv, kernel_size=kernel_size
        )
        self.vae = VAE(dim_rnn, dim_out)

    def forward(self, x, x_lengths=None):
        x = self.encoder(x, x_lengths)
        x = torch.index_select(x, dim=1, index=x_lengths - 1)
        x = x[:, 0, :]
        # x = x.mean(axis=1)
        x, kl_loss = self.vae(x)
        return x, kl_loss
