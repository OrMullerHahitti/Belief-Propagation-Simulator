"""Linear coefficient measurements and operational stability checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def applied_coefficients(
    damped: np.ndarray, attention: np.ndarray, source_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return old-message damping, attention, and actual source coefficients.

    DABP first averages attention heads, then mixes that shared new message
    using damping heads and averages again. Consequently the actual source
    coefficient is count * mean(attention) * mean(new-message weight), NOT
    mean(attention * new-message weight). Inputs may include a time axis.
    """
    if damped.shape[-2] != 2 or damped.shape[-1] != attention.shape[-1]:
        raise ValueError("incompatible damping and attention head dimensions")
    targets = damped.shape[-3]
    source_target = np.asarray(source_target, dtype=np.int64)
    if len(source_target) != attention.shape[-2]:
        raise ValueError("attention provenance length mismatch")
    if np.any(source_target < 0) or np.any(source_target >= targets):
        raise ValueError("source target outside damping rows")
    if not np.isfinite(damped).all() or not np.isfinite(attention).all():
        raise ValueError("non-finite learned weights")
    old = damped[..., 1, :].mean(axis=-1)
    new = damped[..., 0, :].mean(axis=-1)
    share = attention.mean(axis=-1)
    count = np.bincount(source_target, minlength=targets)
    coefficient = share * count[source_target] * new[..., source_target]
    return old, share, coefficient


def split_balance(a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
    """Return the split-adjusted coefficient share, not a new cost-table split."""
    if not 0 < p < 1:
        raise ValueError("split ratio must lie between 0 and 1")
    denominator = p * a + (1 - p) * b
    if np.any(denominator <= 0):
        raise ValueError("split balance requires positive combined influence")
    return p * a / denominator


def paired_sources(metadata: dict) -> list[dict]:
    """Pair halves only when BOTH feed the same variable-to-factor update."""
    groups: dict[tuple[int, str], dict[int, int]] = {}
    for row, (function, target) in enumerate(
        zip(metadata["src_fn_idxes"], metadata["src_trg_idxes"])
    ):
        key = (target, metadata["fn_factor_names"][function])
        half = metadata["fn_half"][function]
        slot = groups.setdefault(key, {})
        if half in slot:
            raise ValueError("duplicate split half in one destination")
        slot[half] = row
    return [
        {"target": target, "factor": factor, "rows": [slot[0], slot[1]]}
        for (target, factor), slot in groups.items()
        if set(slot) == {0, 1}
    ]


def node_changes(series: np.ndarray, owners: np.ndarray, count: int) -> list[dict]:
    """Summarize absolute movement before averaging, so changes cannot cancel."""
    if series.ndim != 2 or series.shape[0] == 0 or series.shape[1] != len(owners):
        raise ValueError("expected nonempty [iteration, coefficient] trajectories")
    movement = np.abs(np.diff(series, axis=0))
    result = []
    for node in range(count):
        columns = owners == node
        if not columns.any():
            raise ValueError(f"variable {node} has no measured coefficients")
        result.append(
            {
                "initial": float(series[0, columns].mean()),
                "final": float(series[-1, columns].mean()),
                "movement": float(movement[:, columns].sum(axis=0).mean()),
                "jump": float(movement[:, columns].max(initial=0)),
            }
        )
    return result


def node_split_changes(
    coefficients: np.ndarray,
    pairs: list[dict],
    target_owners: np.ndarray,
    count: int,
    split: float,
) -> list[dict | None]:
    """Select each node's pair with the largest split departure, in percent points.

    Initial and final values belong to the same pair. Selection uses the largest
    absolute departure from that pair's first iteration anywhere in its run;
    an excursion that returns to its initial value is therefore retained.
    Nodes without a common destination for both halves return None.
    """
    if coefficients.ndim != 2 or coefficients.shape[0] == 0:
        raise ValueError("expected nonempty [iteration, coefficient] trajectories")
    result: list[dict | None] = [None] * count
    if not pairs:
        return result
    rows = np.asarray([pair["rows"] for pair in pairs], dtype=np.int64)
    owners = np.asarray(target_owners)[[pair["target"] for pair in pairs]]
    shares = 100 * split_balance(
        coefficients[:, rows[:, 0]], coefficients[:, rows[:, 1]], split
    )
    departure = np.abs(shares - shares[0])
    peak = departure.max(axis=0)
    for node in range(count):
        indices = np.flatnonzero(owners == node)
        if not len(indices):
            continue
        index = int(indices[np.argmax(peak[indices])])
        result[node] = {
            "pair_index": index,
            "pair_count": int(len(indices)),
            "initial": float(shares[0, index]),
            "final": float(shares[-1, index]),
            "departure": float(peak[index]),
            "peak_iteration": int(np.argmax(departure[:, index])) + 1,
        }
    return result


@dataclass(frozen=True)
class Stability:
    """Observed stability over a complete window, not a convergence proof."""

    assignments: bool
    weights: bool
    range_max: float | None

    @property
    def converged(self) -> bool:
        return self.assignments and self.weights


class StabilityWindow:
    """Require unchanged assignments and bounded full-window coefficient range."""

    def __init__(self, length: int, tolerance: float):
        if length < 2 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid stability window")
        self.length = length
        self.tolerance = tolerance
        self.values: deque = deque(maxlen=length)
        self.assignment: np.ndarray | None = None
        self.unchanged = 0

    def update(self, assignment: np.ndarray, values: np.ndarray) -> Stability:
        if not np.isfinite(values).all():
            raise ValueError("non-finite coefficients cannot establish stability")
        same_assignment = np.array_equal(assignment, self.assignment)
        self.unchanged = self.unchanged + 1 if same_assignment else 1
        self.assignment = assignment.copy()
        self.values.append(values.copy())
        if len(self.values) < self.length:
            return Stability(False, False, None)
        span = float(np.ptp(np.stack(self.values), axis=0).max())
        return Stability(self.unchanged >= self.length, span < self.tolerance, span)
