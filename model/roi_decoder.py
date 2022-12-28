import pytorch_lightning as pl
import torch
from torch.nn import functional as F

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