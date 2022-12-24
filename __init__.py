from omegaconf import OmegaConf
from .dataset.roi import ROIDataset
from math import prod


OmegaConf.register_new_resolver("num_features", lambda user: ROIDataset.calc_num_features(user)[0], use_cache=True)
OmegaConf.register_new_resolver('intprod', lambda *x: int(prod(float(el) for el in x)), use_cache=False)