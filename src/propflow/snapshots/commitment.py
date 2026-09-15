"""Exact cost-dependent commitment thresholds for pairwise incoming messages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommitmentState:
    """Best common row, its worst inequality slack, and boundary classification."""

    row: np.ndarray
    slack: np.ndarray
    strict: np.ndarray
    weak: np.ndarray


def row_thresholds(costs: np.ndarray) -> np.ndarray:
    """Return omega[a,b] = max_v(C[a,v] - C[b,v]) with sender on the row axis.

    Leading batch axes are preserved. Pass the actual clone cost table, not the
    unsplit cost, unless incoming messages have also been scaled accordingly.
    """
    table = np.asarray(costs, dtype=float)
    if table.ndim < 2 or min(table.shape[-2:]) < 2 or not np.isfinite(table).all():
        raise ValueError("finite pairwise tables with at least two labels are required")
    return (table[..., :, None, :] - table[..., None, :, :]).max(axis=-1)


def evaluate_commitment(
    messages: np.ndarray, thresholds: np.ndarray, tolerance: float = 1e-6
) -> CommitmentState:
    """Test every candidate row against every rival and receiver value.

    A candidate a has slack min_{b != a}(Q[b]-Q[a]-omega[a,b]). A positive
    maximum slack means one row is the unique minimizer for all receiver labels.
    Weak commitment includes threshold ties within tolerance. The candidate
    need not minimize Q alone. This is a per-update property, not time stability.
    """
    q = np.asarray(messages, dtype=float)
    omega = np.asarray(thresholds, dtype=float)
    if q.ndim < 1 or q.shape[-1] < 2 or omega.shape[-2:] != (q.shape[-1],) * 2:
        raise ValueError("message domain and threshold matrix must agree")
    if not np.isfinite(q).all() or not np.isfinite(omega).all():
        raise ValueError("non-finite messages or thresholds")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    margins = q[..., None, :] - q[..., :, None] - omega
    indices = np.arange(q.shape[-1])
    margins[..., indices, indices] = np.inf
    by_row = margins.min(axis=-1)
    row = by_row.argmax(axis=-1)
    slack = np.take_along_axis(by_row, row[..., None], axis=-1)[..., 0]
    return CommitmentState(row, slack, slack > tolerance, slack >= -tolerance)
