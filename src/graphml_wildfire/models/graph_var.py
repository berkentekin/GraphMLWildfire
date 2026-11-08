"""Graph VAR forecasting, adapted from the graph polynomial VAR (GP-VAR) of Isufi et al. (2019).

The GP-VAR, eq. (13) of the article, models a signal on one fixed undirected graph as

    x[t] = -sum_{p=1..P} sum_{l=0..L_p} psi[l, p] * L^l x[t-p] + noise,

with L the graph Laplacian, and finds the scalars psi by minimising the prediction MSE, eq. (21).
This module keeps that recursion and the least-squares fit, and predicts

    x[t+1] = sum_{p=1..window} sum_{k=0..hops} psi[p-1, k] * S_t^k x[t+1-p].

It departs from the article as follows. The powers are of a shift operator S instead of the
Laplacian, and S_t is the operator of day t, so the graph may be directed and change from day to
day. All lags share the polynomial order `hops`, and psi has the opposite sign. The
window * (hops + 1) scalars psi are shared by all nodes, days and fires, and minimise the squared
one-step error summed over the training samples; the article poses the same objective through
autocorrelation matrices, eq. (20), which need a single fixed graph. The signal mean is not
subtracted.
"""

import numpy as np
import torch
from scipy import sparse
from torch import nn

from ..data.patches import CHANNEL_INDEX
from ..graphs import apply_shift, pair_validity


class GraphVAR:
    def __init__(self, window: int, hops: int):
        if window < 1 or hops < 0:
            raise ValueError("window must be at least 1 and hops at least 0")
        self.window = window
        self.hops = hops
        self.coefficients = None  # psi: [window, hops + 1]

    @property
    def parameters(self) -> int:
        return self.window * (self.hops + 1)

    def regressors(self, shift: sparse.sparray, history: np.ndarray) -> np.ndarray:
        """Return S^k x[t+1-p] for every node as [nodes, window * (hops + 1)], ordered like psi.

        `shift` is the matrix S[target, source] of the latest day (see `graph.to_sparse`) and
        `history` holds the signals of the last `window` days, oldest first: [window, nodes].
        """

        if history.shape != (self.window, shift.shape[0]):
            raise ValueError("history must have shape [window, nodes]")
        taps = [history[::-1].T.astype(np.float64)]
        for _ in range(self.hops):
            taps.append(shift @ taps[-1])
        return np.stack(taps, axis=2).reshape(shift.shape[0], self.parameters)

    def fit(self, regressors: np.ndarray, targets: np.ndarray) -> "GraphVAR":
        """Least-squares fit of psi to the next-day signal of every training node: [samples]."""

        regressors = regressors.astype(np.float64)
        gram = regressors.T @ regressors
        moment = regressors.T @ targets.astype(np.float64)
        # lstsq gives the minimum-norm solution when regressors are collinear.
        self.coefficients = np.linalg.lstsq(gram, moment)[0].reshape(self.window, self.hops + 1)
        return self

    def predict(self, regressors: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit must be called before predict")
        return regressors @ self.coefficients.reshape(-1)

    def forecast(self, shift: sparse.sparray, history: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction of the signal for every node: [nodes]."""

        return self.predict(self.regressors(shift, history))


class LearnedGraphVAR(nn.Module):
    """Graph VAR with a separate shift operator for every day of the window:

        x[t+1] = sum_{p=1..window} sum_{k=0..hops} psi[p-1, k] * S_{t+1-p}^k x[t+1-p].

    `GraphVAR` filters every lag with the operator of the latest day; here each lag is filtered
    with the operator of its own day. The operators come from `shift`, so psi has no closed form
    and is trained together with the weights of `shift` by gradient descent.

    `shift` is a module with stencil `offsets` (see `graphs.stencil_offsets`) that maps the node
    features of one day [N, C, H, W] and the edge validity [N, K, H, W] to the weights of its
    operator [N, 1, K, H, W]. The signal x is the burning fraction, and so is the output.
    """

    def __init__(self, shift: nn.Module, window: int, hops: int):
        super().__init__()
        if window < 1 or hops < 0:
            raise ValueError("window must be at least 1 and hops at least 0")
        self.shift = shift
        self.window = window
        self.hops = hops
        # psi: [window, hops + 1], latest day first. Starts as persistence.
        coefficients = torch.zeros(window, hops + 1)
        coefficients[0, 0] = 1.0
        self.coefficients = nn.Parameter(coefficients)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Predicted burning fraction [B, H, W] from `features` [B, T, C, H, W], oldest day first,
        and `valid` [B, H, W] (1 on real nodes)."""

        batch, days = features.shape[:2]
        if days != self.window:
            raise ValueError(f"features must hold {self.window} days, got {days}")
        offsets = self.shift.offsets
        # Full precision: bfloat16 would round the predicted fractions into ties that lower AP.
        with torch.autocast(features.device.type, enabled=False):
            day_features = features.float().flatten(0, 1)
            day_valid = valid[:, None].expand(-1, days, -1, -1).flatten(0, 1).float()
            weights = self.shift(day_features, pair_validity(day_valid, offsets))
            if weights.shape[1] != 1:
                raise ValueError("shift must return one operator per day")
            taps = [day_features[:, CHANNEL_INDEX["fire_frac"], None]]
            for _ in range(self.hops):
                taps.append(apply_shift(taps[-1], weights, offsets))
            # S^k x of every day: [B, window, hops + 1, H, W], latest day first like psi.
            filtered = torch.cat(taps, dim=1).unflatten(0, (batch, days)).flip(1)
            return torch.einsum("bpkhw,pk->bhw", filtered, self.coefficients)
