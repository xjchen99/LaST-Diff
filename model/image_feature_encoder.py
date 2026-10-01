"""Deterministic image encoder with the original checkpoint parameter layout."""
from torch import nn
from diffusers.models.autoencoders.vae import Encoder


class ImageFeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, down_block_types,
                 block_out_channels, norm_num_groups, up_block_types=None,
                 layers_per_block=None, **kwargs):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block if layers_per_block is not None else 1,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )

    def forward(self, x):
        return self.encoder(x)
