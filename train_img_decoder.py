import gc
from omegaconf import OmegaConf
from path import Path
import pytorch_lightning as pl
import hydra
from model.vae_img_decoder import VAEImgDecoder
from torch.utils.data import DataLoader
import torch
from dataset.roi import ROIDataset
from math import prod


OmegaConf.register_new_resolver("num_features", lambda user: ROIDataset.calc_num_features(user)[0], use_cache=True)
OmegaConf.register_new_resolver('intprod', lambda *x: int(prod(float(el) for el in x)), use_cache=False)


@hydra.main(config_path="config", config_name="train_img_decoder.yaml")
def main(config):

    print('config', OmegaConf.to_yaml(config), '', sep='\n\n')

    checkpoint_path = Path(config.checkpoint_path)
    ckpt_config_path = checkpoint_path / "config.yaml"
    ckpt_config = OmegaConf.load(ckpt_config_path)

    print('loaded config from checkpoint', ckpt_config_path)
    print('ckpt config', OmegaConf.to_yaml(ckpt_config), sep='\n\n')

    config.img_decoder.input_size = ckpt_config.model.latent_size

    dataset_config = ckpt_config.dataset
    dataset_config['_target_'] = 'dataset.roi.ROIDatasetImage'

    dataset = hydra.utils.instantiate(dataset_config)

    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(dataset, batch_size=config.batch_size, pin_memory=pin_memory)

    ckpt_model = hydra.utils.instantiate(ckpt_config.model)
    ckpt_model.load_state_dict(torch.load(checkpoint_path / 'best.ckpt')['state_dict'])

    print('Loaded model from checkpoint', checkpoint_path / 'best.ckpt')

    img_decoder = hydra.utils.instantiate(config.img_decoder)
    opt = hydra.utils.instantiate(config.opt)

    vae_img = VAEImgDecoder(ckpt_model.encoder, img_decoder, ckpt_model.latent_size, opt)

    monitor_metric = 'loss/train_loss_epoch'
    storage_img_decoder = checkpoint_path / 'img_decoder'
    if not storage_img_decoder.exists():
        storage_img_decoder.mkdir()
    OmegaConf.save(config, storage_img_decoder / 'config.yaml')
    callbacks = pl.callbacks.ModelCheckpoint(storage_img_decoder, 'best',
                                                    monitor=monitor_metric,
                                                    auto_insert_metric_name=False, save_last=True)

    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         gradient_clip_val=config.gradient_clip_val,
                         gradient_clip_algorithm=config.gradient_clip_algorithm)
    del ckpt_model
    gc.collect()
    trainer.fit(vae_img, train_dl)

if __name__ == '__main__':
    main()