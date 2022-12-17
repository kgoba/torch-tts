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
    def __init__(self, encoder_mu, encoder_logvar, decoder):
        super().__init__()
        self.encoder_mu = encoder_mu
        self.encoder_logvar = encoder_logvar
        self.decoder = decoder
        # self.loss_fn = nn.MSELoss()

    def forward(self, x):
        mu, logvar = self.encoder_mu(x), self.encoder_logvar(x)
        sigma = torch.exp(logvar / 2) # == sqrt(e^log_var)
        z_sampled = torch.distributions.distribution.Normal(mu, sigma).rsample()
        x_hat = self.decoder(z_sampled)

        # rec_loss = self.loss_fn(x_hat, x)
        # kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return x_hat, mu, logvar, z_sampled # rec_loss + kl_loss

