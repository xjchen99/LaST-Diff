"""Load trusted legacy model objects with their original module names."""
from pathlib import Path
import torch


def load_model(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: %s" % path)
    model = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Expected a complete torch module in %s" % path)
    return model.to(device=device, dtype=torch.float32)


def load_encoder(path, cfg, device):
    from model.image_feature_encoder import ImageFeatureEncoder
    model = ImageFeatureEncoder(**cfg["image_encoder"]["model"]).to(device)
    state = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model
