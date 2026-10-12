"""The pixel-graph-time CNN: a graph-time CNN on 0.75 km nodes, with a 375 m pixel encoder in
front and a pixel decoder behind.

The graph reasons at node level, where a directed shift operator means something; the pixel
layers decide which pixels inside a node burn, which a node score cannot express.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .gtcnn import GraphTimeCNN


class PixelGraphTimeCNN(nn.Module):
    def __init__(self, node_channels: int, pixel_channels: int, patch: int = 2, pixel_hidden: int = 16, decoder_hidden: int = 32, **gtcnn_kwargs):
        super().__init__()
        self.patch = patch
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(pixel_channels, pixel_hidden, 3, padding=1), nn.GELU(), nn.Conv2d(pixel_hidden, pixel_hidden, 3, padding=1), nn.GELU()
        )
        self.gtcnn = GraphTimeCNN(node_channels + pixel_hidden, **gtcnn_kwargs)
        hidden = self.gtcnn.encoder.out_channels
        self.decoder = nn.Sequential(nn.Conv2d(hidden + pixel_hidden, decoder_hidden, 3, padding=1), nn.GELU(), nn.Conv2d(decoder_hidden, 1, 1))

    def forward(self, features: torch.Tensor, valid: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        """`features` [B, T, C, H, W], `valid` [B, H, W], `pixels` [B, T, Cp, H * patch, W * patch].
        Returns next-day logits per pixel, [B, H * patch, W * patch]."""

        batch, days = features.shape[:2]
        pixel_states = self.pixel_encoder(pixels.flatten(0, 1))
        pooled = F.avg_pool2d(pixel_states, self.patch).unflatten(0, (batch, days))
        nodes = self.gtcnn(torch.cat([features, pooled], dim=2), valid)
        upsampled = F.interpolate(nodes, scale_factor=self.patch, mode="nearest")
        last_day = pixel_states.unflatten(0, (batch, days))[:, -1]
        return self.decoder(torch.cat([upsampled, last_day], dim=1)).squeeze(1)
