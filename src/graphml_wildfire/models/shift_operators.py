"""Shift operators for `LearnedGraphVAR`: modules that map the node features of one day
[N, C, H, W] and the edge validity [N, K, H, W] to stencil weights [N, heads, K, H, W]
(see `graphs.stencil`).

`EdgeAttention` is taken unchanged from the pixel-graph-time CNN of Marcin Jarosz
(branch `marcin/pixel-graph-time-final`, `models/gtcnn.py`).
"""

import torch
from torch import nn

from ..graphs import neighbour_views, stencil_offsets


class UniformShift(nn.Module):
    """Equal weights over the valid neighbours of every node: the 8-neighbour grid of
    `GraphVAR`, with every row scaled to sum to 1."""

    def __init__(self, hops: int = 1):
        super().__init__()
        self.offsets = stencil_offsets(hops)

    def forward(self, state: torch.Tensor, pair_valid: torch.Tensor) -> torch.Tensor:
        valid = pair_valid.float()
        return (valid / valid.sum(dim=1, keepdim=True).clamp_min(1)).unsqueeze(1)


class EdgeAttention(nn.Module):
    """The learned shift operator: GATv2-style attention over the stencil.

    For node i, neighbour j at offset k and head h,
        e_ij = a_h . LeakyReLU(W_t x_i + W_s x_j + b_k),
        S[i, j] = softmax_j(e_ij)  (over the valid neighbours of i),
    so every row of S is a probability distribution over the 8 neighbours, chosen by the state of
    the receiving node, the state of the sender and a learned prior per direction (b_k).
    """

    def __init__(self, channels: int, heads: int, offsets: list[tuple[int, int]], hidden: int = 8):
        super().__init__()
        self.heads, self.hidden, self.offsets = heads, hidden, offsets
        self.target = nn.Conv2d(channels, heads * hidden, 1)
        self.source = nn.Conv2d(channels, heads * hidden, 1, bias=False)
        self.direction = nn.Parameter(torch.zeros(len(offsets), heads * hidden))
        self.score = nn.Parameter(torch.randn(heads, hidden) * hidden**-0.5)

    def forward(self, state: torch.Tensor, pair_valid: torch.Tensor) -> torch.Tensor:
        """Stencil weights [N, heads, K, H, W] from node states [N, C, H, W] and `pair_valid` [N, K, H, W]."""

        count, _, height, width = state.shape
        target = self.target(state)
        logits = []
        for index, source in enumerate(neighbour_views(self.source(state), self.offsets)):
            hidden = target + source + self.direction[index].to(state.dtype)[None, :, None, None]
            hidden = nn.functional.leaky_relu(hidden, 0.2).reshape(count, self.heads, self.hidden, height, width)
            logits.append(torch.einsum("nrahw,ra->nrhw", hidden, self.score.to(state.dtype)))
        logits = torch.stack(logits, dim=2)
        valid = pair_valid.unsqueeze(1)
        logits = logits.masked_fill(valid == 0, -1e4)
        return torch.softmax(logits.float(), dim=2) * valid
