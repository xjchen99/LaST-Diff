"""Discover prepared CAMUS sequences without importing model libraries."""
from pathlib import Path
import numpy as np


def frame_paths(directory):
    return sorted(Path(directory).glob("timeidx=*.npy"),
                  key=lambda p: int(p.stem.split("=")[1]))


def discover_sequences(root, split="train", views=("2CH", "4CH"), frames=16,
                       require_labels=True):
    root = Path(root)
    image_root = root / "images" / split
    if not image_root.is_dir():
        raise FileNotFoundError("Missing image directory: %s" % image_root)
    sequences = []
    for patient in sorted(image_root.iterdir()):
        if not patient.is_dir():
            continue
        for view in views:
            folder = "%s_half_sequence" % view
            image_dir = patient / folder
            if not image_dir.is_dir():
                continue
            images = frame_paths(image_dir)
            expected = ["timeidx=%d.npy" % i for i in range(frames)]
            if [p.name for p in images] != expected:
                raise ValueError("%s must contain exactly %d consecutive frames." % (image_dir, frames))
            labels = []
            if require_labels:
                label_dir = root / "labels" / split / patient.name / folder
                labels = frame_paths(label_dir)
                if [p.name for p in labels] != expected:
                    raise ValueError("Missing or mismatched label frames: %s" % label_dir)
            sequences.append(("%s_%s" % (patient.name, view), images, labels))
    if not sequences:
        raise ValueError("No CAMUS sequences found in %s" % image_root)
    return sequences


def load_frame(path, size, is_label=False):
    array = np.load(path, allow_pickle=False)
    if array.shape == (1, size, size):
        array = array[0]
    if array.shape != (size, size):
        raise ValueError("Expected a %dx%d frame at %s, got %s" % (size, size, path, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("Non-finite values in %s" % path)
    if is_label and not np.isin(array, [0, 1, 2, 3]).all():
        raise ValueError("CAMUS masks must use labels 0, 1, 2, 3: %s" % path)
    return array.astype(np.int64 if is_label else np.float32)
