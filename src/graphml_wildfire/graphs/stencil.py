"""Shift operators on the node grid, stored per node over the 8-neighbour stencil.

Every node (r, c) has one incoming edge per stencil offset. ``weights[n, h, k, r, c]`` is the
weight of the directed edge from the neighbour at ``(r, c) + offsets[k]`` into ``(r, c)``: the
entry S[i, j] of shift operator ``h`` of graph ``n``. This is the ELL sparse format of S, so
S x costs one multiply-add per offset and no dense matrix is ever built.
"""

import torch
import torch.nn.functional as F


def stencil_offsets(hops: int = 1) -> list[tuple[int, int]]:
    """Offsets (d_row, d_col) of all neighbours within `hops` steps (Chebyshev distance)."""

    span = range(-hops, hops + 1)
    return [(d_row, d_col) for d_row in span for d_col in span if (d_row, d_col) != (0, 0)]


def neighbour_views(values: torch.Tensor, offsets: list[tuple[int, int]]) -> list[torch.Tensor]:
    """For each offset, the value at the neighbour of every node ([..., H, W], zero outside)."""

    hops = max(max(abs(d_row), abs(d_col)) for d_row, d_col in offsets)
    height, width = values.shape[-2:]
    padded = F.pad(values, (hops, hops, hops, hops))
    return [
        padded[..., hops + d_row : hops + d_row + height, hops + d_col : hops + d_col + width]
        for d_row, d_col in offsets
    ]


def pair_validity(valid: torch.Tensor, offsets: list[tuple[int, int]]) -> torch.Tensor:
    """[N, K, H, W]: whether both end points of the edge neighbour_k -> node are real nodes."""

    return torch.stack([valid * view for view in neighbour_views(valid, offsets)], dim=1)


class _StencilShift(torch.autograd.Function):
    """S x with a hand-written backward pass, so no per-offset intermediate is kept."""

    @staticmethod
    def forward(ctx, signal, weights, offsets):
        out = torch.zeros_like(signal)
        for index, view in enumerate(neighbour_views(signal, offsets)):
            out.addcmul_(weights[:, :, index, None], view)
        ctx.save_for_backward(signal, weights)
        ctx.offsets = offsets
        return out

    @staticmethod
    def backward(ctx, grad):
        signal, weights = ctx.saved_tensors
        offsets = ctx.offsets
        hops = max(max(abs(d_row), abs(d_col)) for d_row, d_col in offsets)
        height, width = signal.shape[-2:]
        grad_signal = grad_weights = None
        if ctx.needs_input_grad[0]:
            # Transpose of S: every node hands its gradient back to its neighbours.
            padded = F.pad(torch.zeros_like(signal), (hops, hops, hops, hops))
            for index, (d_row, d_col) in enumerate(offsets):
                target = padded[..., hops + d_row : hops + d_row + height, hops + d_col : hops + d_col + width]
                target.addcmul_(weights[:, :, index, None], grad)
            grad_signal = padded[..., hops : hops + height, hops : hops + width]
        if ctx.needs_input_grad[1]:
            grad_weights = torch.stack([(grad * view).sum(dim=2) for view in neighbour_views(signal, offsets)], dim=2)
        return grad_signal, grad_weights, None


def apply_shift(signal: torch.Tensor, weights: torch.Tensor, offsets: list[tuple[int, int]]) -> torch.Tensor:
    """Compute S x. `signal` is [N, F, H, W]; `weights` is [N, heads, K, H, W].

    With several heads the F channels are split into equal groups, one shift operator each.
    """

    count, channels, height, width = signal.shape
    heads = weights.shape[1]
    grouped = signal.reshape(count, heads, channels // heads, height, width)
    return _StencilShift.apply(grouped, weights, offsets).reshape(count, channels, height, width)
