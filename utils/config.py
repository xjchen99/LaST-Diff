"""Shared configuration and checkpoint naming."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "CAMUS.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    return cfg


def checkpoint_paths(cfg, model_dir=None):
    directory = Path(model_dir or Path(cfg["output_dir"]) / "models")
    return {name: directory / (name + ".pth")
            for name in ("vae", "idm", "image_encoder", "vdm")}
