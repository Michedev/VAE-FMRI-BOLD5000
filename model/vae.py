import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class VAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_size: int,
                 beta: float = 1.0, opt: type[torch.optim.Optimizer] = torch.optim.Adam,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self._opt = opt
        self.latent_size = latent_size
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
        recon_loss = torch.nn.functional.mse_loss(
            x_hat, x, reduction='none').mean(dim=0).sum()
        kl_loss = torch.distributions.kl_divergence(
            post, self.prior).mean(dim=0).sum()
        loss = recon_loss + self.beta * kl_loss
        return dict(loss=loss, recon_loss=recon_loss, kl_loss=kl_loss)

    def validation_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = self.calc_loss(x, result['x_hat'], result['posterior'])
        self.log('loss/valid_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self._opt(self.parameters())


class VAEImgDecoder(pl.LightningModule):

    def __init__(roi_encoder: nn.Module, img_decoder: nn.Module, latent_size: int,
                 opt: type[torch.optim.Optimizer] = torch.optim.Adam):
        super().__init__()
        self.encoder = roi_encoder
        self.encoder.eval()
        self.decoder = img_decoder
        self._opt = opt
        self.latent_size = latent_size

    def forward(self, x):
        mu, sigma, dist, z = self.forward_encoder(x)
        x_hat = self.decoder(z)

        return dict(x_hat=x_hat, mu=mu, sigma=sigma, z=z, posterior=dist)

    @torch.no_grad()
    def forward_encoder(self, x):
        mu, sigma = self.encoder(x).chunk(2, dim=1)
        sigma = torch.exp(sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        return mu, sigma, dist, z

    def training_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            target=x, input=result['x_hat'])
        if self.global_step % 100 == 0:
            self.log('loss/train_loss', loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            target=x, input=result['x_hat'])
        self.log('loss/valid_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self._opt(self.parameters())

class ROIDecoder(pl.LightningModule):
    """
    Decode ROI vector get from all the brain areas to the viewed image
    """

    def __init__(self, decoder: nn.Module, opt: type[torch.optim.Optimizer] = torch.optim.Adam):
        super().__init__()
        self.decoder = decoder
        self._opt = opt

    def forward(self, z):
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            target=x, input=result['x_hat'])
        if self.global_step % 100 == 0:
            self.log('loss/train_loss', loss, on_step=True, on_epoch=False,
                     prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['roi']
        result = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            target=x, input=result['x_hat'])
        self.log('loss/valid_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self._opt(self.parameters())