import hydra
import pkg_resources
from omegaconf import DictConfig, OmegaConf
from path import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import omegaconf
import os
from utils.paths import CODE_MODEL

from .dataset.roi import ROIDataset
from math import prod


OmegaConf.register_new_resolver("num_features", lambda user: ROIDataset.calc_num_features(user)[0], use_cache=True)
OmegaConf.register_new_resolver('intprod', lambda *x: int(prod(float(el) for el in x)), use_cache=False)

@hydra.main('config', 'train.yaml')
def train(config: DictConfig):
    ckpt = None
    pl.seed_everything(config.seed)
    if config.ckpt is not None:
        ckpt = Path(config.ckpt)
        assert ckpt.exists() and ckpt.isdir(), f"Checkpoint {ckpt} does not exist or is not a directory"
        config = OmegaConf.load(ckpt / 'config.yaml')
        os.chdir(ckpt.abspath())
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    train_dataset: Dataset = hydra.utils.instantiate(config.dataset)

    CODE_MODEL.copytree('model')  # copy source code of model under experiment directory

    model.save_hyperparameters(OmegaConf.to_object(config)['model'])  # save model hyperparameters in tb

    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    
    monitor_metric = 'loss/train_loss_epoch'
    ckpt_callback = ModelCheckpoint('./', 'best',
                                    monitor=monitor_metric,
                                    auto_insert_metric_name=False, save_last=True)
    callbacks = [ckpt_callback]
    if config.early_stop:
        callbacks.append(EarlyStopping(monitor_metric, min_delta=config.min_delta,
                                       patience=config.patience))
    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         gradient_clip_val=config.gradient_clip_val,
                         gradient_clip_algorithm=config.gradient_clip_algorithm,
                         resume_from_checkpoint=ckpt)
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    train()
