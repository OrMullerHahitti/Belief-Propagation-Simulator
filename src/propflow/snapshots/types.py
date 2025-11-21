"""Simplified snapshot data structures.

This module defines the minimal data containers used by the lightweight
snapshot system. Each snapshot is self-contained and stores the runtime
state required for downstream analysis (Jacobians, cycle metrics,
visualisation tooling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Jacobians:
    """Holds Jacobian-related matrices and associated metadata."""

    idxQ: Dict[Tuple[str, str, int], int]
    idxR: Dict[Tuple[str, str, int], int]
    A: csr_matrix
    P: csr_matrix
    B: csr_matrix
    block_norms: Optional[Dict[str, float]] = None


@dataclass
class CycleMetrics:
    """Compact summary of cycle analysis for a given step."""

    num_cycles: int
    aligned_hops_total: int
    has_certified_contraction: bool
    details: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_cycles": self.num_cycles,
            "aligned_hops_total": self.aligned_hops_total,
            "has_certified_contraction": self.has_certified_contraction,
            "details": self.details,
        }


@dataclass
class EngineSnapshot:
    """Self-contained snapshot captured at a single BP step.

    Attributes:
        step: Current iteration step. (int)
        lambda_: Damping factor used in this step. (float)
        dom: Variable domains.
        N_var: Neighbouring factors for each variable.
        N_fac: Neighbouring variables for each factor. (2 for binary factors)
        Q: Outgoing Q messages.
        R: Outgoing R messages.
        unary: Unary beliefs for each variable.
        beliefs: Marginal beliefs for each variable.
        assignments: Current MAP assignments for each variable.
        global_cost: Current global cost (if computed).
        metadata: Additional metadata captured at this step.
        cost_tables: Cost tables for each factor.
        cost_labels: Labels for cost tables.
        jacobians: Jacobian matrices and metadata (if computed).
        cycles: Cycle metrics (if computed).
        winners: Winning assignments per factor (if applicable).
        min_idx: Indices of minimum costs per message (if applicable).
        bct_metadata: Additional BCT-related metadata.
        captured_at: Timestamp when the snapshot was captured.
    """

    step: int
    lambda_: float
    dom: Dict[str, List[str]]
    N_var: Dict[str, List[str]]
    N_fac: Dict[str, List[str]]
    Q: Dict[Tuple[str, str], np.ndarray]
    R: Dict[Tuple[str, str], np.ndarray]
    unary: Dict[str, np.ndarray] = field(default_factory=dict)
    beliefs: Dict[str, np.ndarray] = field(default_factory=dict)
    assignments: Dict[str, int] = field(default_factory=dict)
    global_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost_tables: Dict[str, np.ndarray] = field(default_factory=dict)
    cost_labels: Dict[str, List[str]] = field(default_factory=dict)
    jacobians: Optional[Jacobians] = None
    cycles: Optional[CycleMetrics] = None
    winners: Optional[Dict[Tuple[str, str, str], Dict[str, str]]] = None
    min_idx: Optional[Dict[Tuple[str, str], int]] = None
    bct_metadata: Dict[str, Any] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:  # type: ignore[override]
        # Wrap message stores with pretty printers without changing dict behaviour.
        self.Q = _PrettyMessageDict(self.Q)
        self.R = _PrettyMessageDict(self.R)


class _PrettyMessageDict(dict):
    """Dict subclass that renders message payloads in a readable, aligned form."""

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return self._format()

    __str__ = __repr__

    def _format(self) -> str:
        if not self:
            return "{}"
        lines: List[str] = []
        for (src, dst), values in self._sorted_items():
            arr_str = np.array2string(
                np.asarray(values),
                precision=3,
                separator=", ",
                suppress_small=True,
            )
            lines.append(f"{src} -> {dst}: {arr_str}")
        return "{\n  " + "\n  ".join(lines) + "\n}"

    def _sorted_items(self) -> Iterable[Tuple[Tuple[str, str], Any]]:
        def _key(item: Tuple[Tuple[str, str], Any]) -> Tuple[str, str]:
            (src, dst), _ = item
            return (str(src), str(dst))

        return sorted(super().items(), key=_key)
    
__all__ = [
    "EngineSnapshot",
    "Jacobians",
    "CycleMetrics",
]
