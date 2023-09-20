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
        kl_loss = -(1 + logvar - mu * mu - logvar.exp()) / 2
        return z_sampled, kl_loss
