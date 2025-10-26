"""Mathematical utilities supporting the reporting analyzer module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse

from propflow.snapshots.types import Jacobians


def binary_thresholds(cost: np.ndarray) -> tuple[float, float]:
    """Compute the neutrality thresholds ``(Θ0, Θ1)`` for a binary factor."""
    matrix = np.asarray(cost, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("Binary threshold computation requires a 2x2 cost table")

    diff0 = matrix[0, 0] - matrix[0, 1]
    diff1 = matrix[1, 0] - matrix[1, 1]
    theta0 = max(diff0, diff1)
    theta1 = max(-diff0, -diff1)
    return float(theta0), float(theta1)


def check_binary_neutral(delta_q: float, theta0: float, theta1: float) -> tuple[bool, int | None]:
    """Check whether ``ΔQ`` certifies a neutral factor step in the binary case."""
    if delta_q >= theta0:
        return True, 0
    return (True, 1) if delta_q <= -theta1 else (False, None)


def multilabel_gaps(cost: np.ndarray) -> tuple[Dict[int, np.ndarray], Callable[[int], np.ndarray]]:
    """Compute row-wise gap certificates for the multi-label case."""
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cost table for multi-label gaps must be square")
    size = matrix.shape[0]
    gaps: Dict[int, np.ndarray] = {}
    for label in range(size):
        row = matrix[label]
        base = row[label]
        gap_vector = row - base
        gap_vector[label] = 0.0
        gaps[label] = gap_vector

    def selector_block(winner: int) -> np.ndarray:
        if winner < 0 or winner >= size:
            raise ValueError("Winner index out of range")
        S = np.eye(size)
        row = S[winner: winner + 1, :]
        return S - np.ones((size, 1)) @ row

    return gaps, selector_block


def scfg_scaling(alpha: float, structure):
    """Scale thresholds or gap certificates under soft-constrained factor gains."""
    if alpha < 0:
        raise ValueError("Scaling factor alpha must be non-negative")

    if isinstance(structure, tuple) and len(structure) == 2:
        return tuple(alpha * float(value) for value in structure)
    if isinstance(structure, Mapping):
        return {key: np.asarray(value, dtype=float) * alpha for key, value in structure.items()}
    raise TypeError("Unsupported structure for SCFG scaling")


def sibling_reinforcement_lower_bound(u_sibling: float, reinforcement_terms: Sequence[float], kappa: float) -> float:
    """Return the lower bound supplied by the sibling reinforcement condition."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    if not reinforcement_terms:
        return max(0.0, -u_sibling)
    term = max(float(value) for value in reinforcement_terms)
    return max(0.0, kappa * term - float(u_sibling))


def jacobian_blocks_APB(jacobians: Jacobians, *, as_sparse: bool | None = None) -> tuple:
    """Return the ``(A, P, B)`` blocks from a :class:`~propflow.snapshots.types.Jacobians` object."""
    if as_sparse is None:
        return jacobians.A, jacobians.P, jacobians.B

    def _convert(matrix):
        if as_sparse:
            return matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
        return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)

    return _convert(jacobians.A), _convert(jacobians.P), _convert(jacobians.B)


__all__ = [
    "binary_thresholds",
    "check_binary_neutral",
    "multilabel_gaps",
    "scfg_scaling",
    "sibling_reinforcement_lower_bound",
    "jacobian_blocks_APB",
]
