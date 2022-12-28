import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid


class VAEImgDecoder(pl.LightningModule):

    def __init__(self, roi_encoder: nn.Module, img_decoder: nn.Module, latent_size: int,
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
        x_roi = batch['roi']
        x_img = batch['img']
        result = self.forward(x_roi)
        loss = F.binary_cross_entropy_with_logits(
            target=x_img, input=result['x_hat'], reduction='none').mean(dim=0).sum()
        if self.global_step % 100 == 0:
            x_xhat = torch.stack((x_img[:3], result['x_hat'][:3].sigmoid()), dim=1).detach().cpu()
            x_xhat = torch.flatten(x_xhat, start_dim=0, end_dim=1)
            x_xhat = make_grid(x_xhat, nrow=3, normalize=True)
            self.logger.experiment.add_image('x_xhat', x_xhat, self.global_step)
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
        return self._opt(self.decoder.parameters())