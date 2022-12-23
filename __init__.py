from omegaconf import OmegaConf
from .dataset.roi import ROIDataset


OmegaConf.register_new_resolver("num_features", lambda user: ROIDataset.calc_num_features(user)[0], use_cache=True)