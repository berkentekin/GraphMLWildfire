"""Non-graph baselines on the benchmark node grid. Every model maps features [B, T, C, H, W] to logits [B, H, W]."""

import torch
from torch import nn

from ..data.patches import CHANNEL_INDEX


class NodeMLP(nn.Module):
    """Classify every node from its own window of features. `layers=0` is logistic regression."""

    def __init__(self, in_channels: int, window: int, hidden: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks, width = [], in_channels * window
        for _ in range(layers):
            blocks += [nn.Conv2d(width, hidden, 1), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        self.net = nn.Sequential(*blocks, nn.Conv2d(width, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.flatten(1, 2)).squeeze(1)


class NodeLSTM(nn.Module):
    """Run an LSTM over every node's own sequence of daily features and classify it from the last
    hidden state. Nodes never exchange information, so the graph is not used."""

    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1, chunk: int = 2**16):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv2d(hidden, 1, 1)
        # Nodes per LSTM call: full test images hold far more nodes than cuDNN accepts in one batch.
        self.chunk = chunk

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, window, channels, height, width = features.shape
        sequences = features.permute(0, 3, 4, 1, 2).reshape(-1, window, channels)
        last = torch.cat([self.lstm(part)[0][:, -1] for part in sequences.split(self.chunk)])
        state = last.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        return self.head(self.dropout(state)).squeeze(1)


class Persistence(nn.Module):
    """Tomorrow burns where today burns: the logit of the node's burning fraction today."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.logit(features[:, -1, CHANNEL_INDEX["fire_frac"]], eps=1e-4)


def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class ConvNet(nn.Module):
    """Small U-Net over the node grid with the days of the window stacked as channels."""

    def __init__(self, in_channels: int, window: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down1 = _conv_block(in_channels * window, hidden, dropout)
        self.down2 = _conv_block(hidden, hidden * 2, dropout)
        self.bottom = _conv_block(hidden * 2, hidden * 4, dropout)
        self.up2 = _conv_block(hidden * 6, hidden * 2, dropout)
        self.up1 = _conv_block(hidden * 3, hidden, dropout)
        self.head = nn.Conv2d(hidden, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(features.flatten(1, 2))
        skip2 = self.down2(self.pool(skip1))
        state = self.bottom(self.pool(skip2))
        state = nn.functional.interpolate(state, size=skip2.shape[-2:], mode="nearest")
        state = self.up2(torch.cat([state, skip2], dim=1))
        state = nn.functional.interpolate(state, size=skip1.shape[-2:], mode="nearest")
        state = self.up1(torch.cat([state, skip1], dim=1))
        return self.head(state).squeeze(1)
