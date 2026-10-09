"""Graph-time convolutional network on a product graph (Sabbaqi & Isufi, 2023).

The spatial shift operator S may change every day; time is a directed path graph, so
information only flows from earlier days to later ones.
"""

import torch
from torch import nn

from ..graphs import apply_shift, edge_geometry, grid_weights, neighbour_views, physics_weights, stencil_offsets


GRAPHS = ("grid", "physics", "physics_avg", "physics_learned", "learned", "attention")


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


class EdgeWeightNet(nn.Module):
    """Learned shift operators: one weight per edge from wind, slope and edge direction."""

    def __init__(self, heads: int, edges: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, hidden, 1), nn.GELU(), nn.Conv2d(hidden, heads, 1))
        self.direction_bias = nn.Parameter(torch.zeros(heads, edges))

    def forward(self, geometry: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = geometry["valid"]
        count, edges, height, width = valid.shape
        features = torch.stack(
            [
                geometry["speed"] * geometry["alignment"] / 5.0,
                geometry["alignment"],
                geometry["rise"] * 10.0,
                geometry["speed"] / 5.0,
            ],
            dim=1,
        )
        logits = self.net(features.reshape(count, 4, edges * height, width))
        logits = logits.reshape(count, -1, edges, height, width) + self.direction_bias[None, :, :, None, None]
        logits = logits.masked_fill(valid.unsqueeze(1) == 0, -1e4)
        return torch.softmax(logits.float(), dim=2) * valid.unsqueeze(1)


class EdgeAttention(nn.Module):
    """GATv2-style attention over the stencil, conditioned on node states and edge geometry."""

    def __init__(self, channels: int, heads: int, offsets: list[tuple[int, int]], hidden: int = 8):
        super().__init__()
        self.heads, self.hidden, self.offsets = heads, hidden, offsets
        self.target = nn.Conv2d(channels, heads * hidden, 1)
        self.source = nn.Conv2d(channels, heads * hidden, 1, bias=False)
        self.edge = nn.Parameter(torch.zeros(len(offsets), heads * hidden, 3))
        self.direction = nn.Parameter(torch.zeros(len(offsets), heads * hidden))
        self.score = nn.Parameter(torch.randn(heads, hidden) * hidden**-0.5)

    def forward(self, state: torch.Tensor, geometry: dict[str, torch.Tensor]) -> torch.Tensor:
        count, _, height, width = state.shape
        target = self.target(state)
        sources = neighbour_views(self.source(state), self.offsets)
        edge_features = torch.stack(
            [geometry["speed"] * geometry["alignment"] / 5.0, geometry["alignment"], geometry["rise"] * 10.0], dim=-1
        )
        logits = []
        for index, source in enumerate(sources):
            edge_term = torch.einsum("nhwe,ce->nchw", edge_features[:, index].to(state.dtype), self.edge[index].to(state.dtype))
            hidden = target + source + edge_term + self.direction[index].to(state.dtype)[None, :, None, None]
            hidden = nn.functional.leaky_relu(hidden, 0.2).reshape(count, self.heads, self.hidden, height, width)
            logits.append(torch.einsum("nrahw,ra->nrhw", hidden, self.score.to(state.dtype)))
        logits = torch.stack(logits, dim=2)
        valid = geometry["valid"].unsqueeze(1)
        logits = logits.masked_fill(valid == 0, -1e4)
        return torch.softmax(logits.float(), dim=2) * valid


class GraphTimeLayer(nn.Module):
    """Graph filter sum_k H_k S_x^k X on the parametric product graph S_x.

    S_x = s00 I + s01 (I_T x S) + s10 (S_T x I) + s11 (S_T x S) with learned s >= 0 that
    sum to one, where S_T shifts every node signal one day forward.
    """

    def __init__(self, channels: int, taps: int, heads: int, offsets: list[tuple[int, int]], attention: bool, dropout: float):
        super().__init__()
        self.taps, self.offsets = taps, offsets
        self.mix = nn.Parameter(torch.tensor([0.0, 1.0, 1.0, 1.0]))
        self.attention = EdgeAttention(channels, heads, offsets) if attention else None
        self.filter = nn.Conv2d(channels * (taps + 1), channels, 1)
        self.norm = NodeNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def product_shift(self, state: torch.Tensor, weights: torch.Tensor, days: int) -> torch.Tensor:
        mix = torch.softmax(self.mix, dim=0)
        spatial = apply_shift(state, weights.to(state.dtype), self.offsets)
        out = mix[0] * state + mix[1] * spatial
        if days > 1:
            delayed = (mix[2] * state + mix[3] * spatial).unflatten(0, (-1, days))
            delayed = torch.cat([torch.zeros_like(delayed[:, :1]), delayed[:, :-1]], dim=1)
            out = out + delayed.flatten(0, 1)
        return out

    def forward(self, state: torch.Tensor, weights: torch.Tensor | None, geometry: dict[str, torch.Tensor], days: int) -> torch.Tensor:
        if self.attention is not None:
            weights = self.attention(state, geometry)
        powers = [state]
        for _ in range(self.taps):
            powers.append(self.product_shift(powers[-1], weights, days))
        update = self.filter(torch.cat(powers, dim=1))
        return state + self.dropout(nn.functional.gelu(self.norm(update)))


class GraphTimeCNN(nn.Module):
    """Predict next-day fire per node from a window of daily node features."""

    def __init__(
        self,
        in_channels: int,
        graph: str = "grid",
        hidden: int = 64,
        layers: int = 4,
        taps: int = 3,
        heads: int = 4,
        hops: int = 1,
        dropout: float = 0.1,
        patch_metres: float = 1500.0,
        wind_gain: float = 0.3,
        slope_gain: float = 3.0,
    ):
        super().__init__()
        if graph not in GRAPHS:
            raise ValueError(f"Unknown graph: {graph}")
        self.graph, self.patch_metres = graph, patch_metres
        self.offsets = stencil_offsets(hops)
        learned_heads = heads if graph in ("learned", "attention") else 1
        gains = torch.tensor([wind_gain, slope_gain])
        if graph == "physics_learned":
            self.gains = nn.Parameter(gains)
        else:
            self.register_buffer("gains", gains)
        self.edge_net = EdgeWeightNet(heads, len(self.offsets)) if graph == "learned" else None
        self.encoder = nn.Conv2d(in_channels, hidden, 1)
        self.layers = nn.ModuleList(
            GraphTimeLayer(hidden, taps, learned_heads, self.offsets, graph == "attention", dropout) for _ in range(layers)
        )
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def shift_weights(self, geometry: dict[str, torch.Tensor], days: int) -> torch.Tensor | None:
        """Stencil weights [N, heads, K, H, W] of the spatial shift operator of every day."""

        if self.graph == "grid":
            return grid_weights(geometry["valid"])
        if self.graph in ("physics", "physics_learned"):
            return physics_weights(geometry, self.gains[0], self.gains[1])
        if self.graph == "physics_avg":
            weights = physics_weights(geometry, self.gains[0], self.gains[1]).unflatten(0, (-1, days))
            return weights.mean(dim=1, keepdim=True).expand_as(weights).flatten(0, 1)
        if self.graph == "learned":
            return self.edge_net(geometry)
        return None

    def node_states(self, features: torch.Tensor, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        """Node states of the last day, [B, hidden, H, W], for `features` [B, T, C, H, W] and
        `fields` of [B, T, H, W] maps."""

        batch, days = features.shape[:2]
        with torch.autocast("cuda", enabled=False):
            geometry = edge_geometry(
                **{name: values.flatten(0, 1).float() for name, values in fields.items()},
                offsets=self.offsets,
                patch_metres=self.patch_metres,
            )
            weights = self.shift_weights(geometry, days)
        state = self.encoder(features.flatten(0, 1))
        for layer in self.layers:
            state = layer(state, weights, geometry, days)
        return state.unflatten(0, (batch, days))[:, -1]

    def forward(self, features: torch.Tensor, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits [B, H, W] of every node for the next day."""

        return self.head(self.node_states(features, fields)).squeeze(1)
