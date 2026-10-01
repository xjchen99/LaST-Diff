"""CAMUS loaders for spatial training, temporal training, and inference."""
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.sequences import discover_sequences, load_frame


class CAMUSDataset(Dataset):
    def __init__(self, root, split="train", size=256, frames=16,
                 views=("2CH", "4CH"), video=False, random_flip=False,
                 require_labels=True):
        self.sequences = discover_sequences(root, split, views, frames, require_labels)
        self.size = size
        self.video = video
        self.random_flip = random_flip
        self.require_labels = require_labels
        self.items = []
        for case, images, labels in self.sequences:
            if video:
                self.items.append((case, images, labels))
            else:
                for i, path in enumerate(images):
                    self.items.append((case, [path], [labels[i]] if labels else []))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        case, images, labels = self.items[index]
        image = torch.from_numpy(np.stack([load_frame(p, self.size) for p in images])[:, None])
        mask = None
        if self.require_labels:
            mask = torch.from_numpy(np.stack([load_frame(p, self.size, True) for p in labels])[:, None])
        if self.random_flip:
            for axis in (-1, -2):
                if np.random.rand() < 0.5:
                    image = image.flip([axis])
                    if mask is not None:
                        mask = mask.flip([axis])
        if not self.video:
            image = image[0]
            mask = mask[0] if mask is not None else None
        return (image, mask) if self.require_labels else (image, case)
