from path import Path

ROOT: Path = Path(__file__).parent.parent
CODE_MODEL: Path = ROOT / 'model'
CONFIG: Path = ROOT / 'config'
DATA: Path = ROOT / 'data_storage'
CONFIG_DATA: Path = CONFIG / 'dataset'
CONFIG_MODEL: Path = CONFIG / 'model'
CONFIG_MODEL_DATASET: Path = CONFIG / 'model_dataset'

SAVED_MODELS: Path = ROOT / 'saved_models'
