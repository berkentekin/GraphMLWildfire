"""Graph-time convolutional network on a product graph (Isufi & Mazzola, 2021).

By default the spatial shift is learned attention over the 8-neighbour stencil,
recomputed for every node, day and layer. Ablations fix it to uniform neighbour
weights or remove neighbour messages while retaining same-node temporal shifts.

Time is a directed path graph (day t-1 -> day t), so information only flows forward in time.
"""

import torch
from torch import nn

from ..graphs import apply_shift, neighbour_views, pair_validity, stencil_offsets

GRAPH_MODES = ("learned", "uniform", "none")


class NodeNorm(nn.Module):
    """Layer normalisation over the channels of each node, independent of the graph size."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state - state.mean(dim=1, keepdim=True)
        scale = state.float().pow(2).mean(dim=1, keepdim=True).add(1e-5).rsqrt().to(state.dtype)
        return state * scale * self.weight + self.bias


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


class GraphTimeLayer(nn.Module):
    """Graph filter X + GELU(NodeNorm(sum_k H_k S_x^k X)) on the parametric product graph

        S_x = s0 I + s1 (I_T x S_t) + s2 (S_T x I) + s3 (S_T x S_{t-1}),

    where S_t is the attention operator of day t, S_T shifts every node signal one day forward,
    and the four weights s = softmax(s_global + W_gate x_i) are chosen per node from its state.
    """

    def __init__(self, channels: int, taps: int, heads: int, offsets: list[tuple[int, int]], dropout: float,
                 graph_mode: str = "learned"):
        super().__init__()
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"Unknown graph_mode: {graph_mode}")
        if heads < 1 or channels % heads:
            raise ValueError("channels must be divisible by a positive number of heads")
        self.graph_mode, self.heads = graph_mode, heads
        self.taps, self.offsets = taps, offsets
        self.mix = nn.Parameter(torch.tensor([0.0, 1.0, 1.0, 1.0]))
        # Per-node correction to the mix; starts at zero so the global mix is the prior.
        self.gate = nn.Conv2d(channels, 4, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.attention = EdgeAttention(channels, heads, offsets) if graph_mode == "learned" else None
        self.filter = nn.Conv2d(channels * (taps + 1), channels, 1)
        self.norm = NodeNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def product_shift(self, state: torch.Tensor, weights: torch.Tensor | None, days: int) -> torch.Tensor:
        """S_x X for node signals [B * days, C, H, W]."""

        logits = self.mix[None, :, None, None] + self.gate(state).float()
        if self.graph_mode == "none":
            active = logits.new_tensor([True, False, days > 1, False], dtype=torch.bool)
            logits = logits.masked_fill(~active[None, :, None, None], -torch.inf)
        mix = torch.softmax(logits, dim=1).to(state.dtype)
        out = mix[:, 0:1] * state
        spatial = None
        if self.graph_mode != "none":
            if weights is None:
                raise ValueError("Spatial graph modes require stencil weights")
            spatial = apply_shift(state, weights.to(state.dtype), self.offsets)
            out = out + mix[:, 1:2] * spatial
        if days > 1:
            def yesterday(values: torch.Tensor) -> torch.Tensor:
                values = values.unflatten(0, (-1, days))
                return torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1).flatten(0, 1)
            out = out + mix[:, 2:3] * yesterday(state)
            if spatial is not None:
                out = out + mix[:, 3:4] * yesterday(spatial)
        return out

    def forward(self, state: torch.Tensor, pair_valid: torch.Tensor, days: int) -> torch.Tensor:
        weights = None
        if self.attention is not None:
            weights = self.attention(state, pair_valid)
        elif self.graph_mode == "uniform":
            uniform = pair_valid / pair_valid.sum(dim=1, keepdim=True).clamp_min(1)
            weights = uniform[:, None].expand(-1, self.heads, -1, -1, -1)
        powers = [state]
        for _ in range(self.taps):
            powers.append(self.product_shift(powers[-1], weights, days))
        return state + self.dropout(nn.functional.gelu(self.norm(self.filter(torch.cat(powers, dim=1)))))


class GraphTimeCNN(nn.Module):
    """Node states of the last day of a window of daily node features."""

    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 4, taps: int = 3, heads: int = 4,
                 dropout: float = 0.1, graph_mode: str = "learned"):
        super().__init__()
        self.offsets = stencil_offsets(hops=1)
        self.encoder = nn.Conv2d(in_channels, hidden, 1)
        self.layers = nn.ModuleList(GraphTimeLayer(hidden, taps, heads, self.offsets, dropout, graph_mode) for _ in range(layers))

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """`features` [B, T, C, H, W], `valid` [B, H, W] (1 on real nodes). Returns [B, hidden, H, W]."""

        batch, days = features.shape[:2]
        with torch.autocast("cuda", enabled=False):
            pair_valid = pair_validity(valid[:, None].expand(-1, days, -1, -1).flatten(0, 1).float(), self.offsets)
        state = self.encoder(features.flatten(0, 1))
        for layer in self.layers:
            state = layer(state, pair_valid, days)
        return state.unflatten(0, (batch, days))[:, -1]
