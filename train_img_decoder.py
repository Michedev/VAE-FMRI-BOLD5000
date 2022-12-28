import gc
import omegaconf
from path import Path
import pytorch_lightning as pl
import hydra
from model.vae import VAEImgDecoder
from torch.utils.data import DataLoader


@hydra.main(config_path="config", config_name="train_img_decoder.yaml")
def main(config):

    checkpoint_path = Path(config.checkpoint_path)
    ckpt_config_path = checkpoint_path / "config.yaml"
    ckpt_config = omegaconf.OmegaConf.load(ckpt_config_path)

    config.model = ckpt_config.model

    dataset_config = ckpt_config.dataset
    dataset_config['_target_'] = 'dataset.roi.ROIDatasetImage'

    dataset = hydra.utils.instantiate(dataset_config)

    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(dataset, batch_size=config.batch_size, pin_memory=pin_memory)

    ckpt_model = hydra.utils.instantiate(ckpt_config.model)
    ckpt_model.load_checkpoint(checkpoint_path / 'best.ckpt')

    print('Loaded model from checkpoint', checkpoint_path / 'best.ckpt')

    img_decoder = hydra.utils.instantiate(config.decoder)
    opt = hydra.utils.instantiate(config.opt)

    vae_img = VAEImgDecoder(ckpt_model.encoder, img_decoder, ckpt_model.latent_size, opt)

    monitor_metric = 'loss/train_loss'
    storage_img_decoder = checkpoint_path / 'img_decoder'
    if not storage_img_decoder.exists():
        storage_img_decoder.mkdir()
    callbacks = pl.callbacks.ModelCheckpoint(storage_img_decoder, 'best',
                                                    monitor=monitor_metric,
                                                    auto_insert_metric_name=False, save_last=True)

    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         gradient_clip_val=config.gradient_clip_val,
                         gradient_clip_algorithm=config.gradient_clip_algorithm)
    del ckpt_model
    gc.collect()
    trainer.fit(vae_img, train_dl)
