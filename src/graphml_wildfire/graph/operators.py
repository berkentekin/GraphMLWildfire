import dataclasses

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .structures import ShiftOperator


def to_sparse(shift: ShiftOperator) -> sparse.csr_array:
    """Assemble the sparse matrix S with S[target, source] = edge_weight."""

    sources, targets = shift.edge_index
    return sparse.csr_array(
        (shift.edge_weight.astype(np.float64), (targets, sources)),
        shape=(shift.number_of_nodes, shift.number_of_nodes),
    )


def spectral_norm(shift: ShiftOperator) -> float:
    """Largest singular value of S, which is |lambda_max| when S is symmetric."""

    matrix = to_sparse(shift)
    if shift.number_of_nodes <= 64:
        return float(np.linalg.norm(matrix.toarray(), 2))
    # A fixed start vector keeps ARPACK deterministic.
    start = np.ones(shift.number_of_nodes)
    return float(svds(matrix, k=1, v0=start, return_singular_vectors=False)[0])


def normalize_shift(shift: ShiftOperator) -> ShiftOperator:
    """Scale the edge weights so that S has unit spectral norm."""

    norm = spectral_norm(shift)
    if norm == 0:
        raise ValueError("Cannot normalize a shift operator without nonzero edges")
    return dataclasses.replace(shift, edge_weight=shift.edge_weight / norm)
