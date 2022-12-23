import pytorch_lightning as pl
import torch
from torch import nn


class VAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, beta: float = 1.0, opt: type[torch.optim.Optimizer] = torch.optim.Adam):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.opt = opt
        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, sigma, dist, z = self.forward_encoder(x)
        x_hat = self.decoder(z)

        return dict(x_hat=x_hat, mu=mu, sigma=sigma, z=z, posterior=dist)

    def forward_encoder(self, x):
        mu, sigma = self.encoder(x).chunk(2, dim=1)
        sigma = torch.exp(sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        return mu, sigma, dist, z

    def training_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = self.calc_loss(x, result['x_hat'], result['posterior'])
        if self.global_step % 100 == 0:
            self.log('loss/train_loss', loss, on_step=True, on_epoch=False, 
                      prog_bar=True, logger=True)
        return loss

    def calc_loss(self, x, x_hat, post: torch.distributions.Normal):
        recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='none').mean(dim=0).sum()
        kl_loss = torch.distributions.kl_divergence(post, self.prior).mean(dim=0).sum()
        loss = recon_loss + self.beta * kl_loss
        return dict(loss=loss, recon_loss=recon_loss, kl_loss=kl_loss)

    def validation_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = self.calc_loss(x, result['x_hat'], result['posterior'])
        self.log('loss/valid_loss', loss, on_step=True, on_epoch=True, 
                  prog_bar=True, logger=True)
        return loss