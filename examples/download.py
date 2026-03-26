from openpi.shared import download
from openpi.training import config as _config

model_name = "pi05_libero"

config = _config.get_config(model_name)
checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
print(f"Model checkpoint downloaded to: {checkpoint_dir}")