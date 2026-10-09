"""Graph VAR forecasting: the graph polynomial VAR (GP-VAR) of Isufi et al. (2019), eq. (13).

Tomorrow's fire signal is a sum of polynomial graph filters applied to the last `window` days,

    x[t+1] = sum_{p=1..window} sum_{k=0..hops} psi[p-1, k] * S_t^k x[t+1-p],

where S_t is the shift operator of day t, so the graph may change from day to day. The
window * (hops + 1) scalars psi are shared by all nodes, days and fires, and minimise the squared
one-step prediction error on the training samples.
"""

import numpy as np
from scipy import sparse


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
