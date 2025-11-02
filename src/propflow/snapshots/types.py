"""Simplified snapshot data structures.

This module defines the minimal data containers used by the lightweight
snapshot system. Each snapshot is self-contained and stores the runtime
state required for downstream analysis (Jacobians, cycle metrics,
visualisation tooling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
    """Self-contained snapshot captured at a single BP step."""

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

    @property
    def data(self) -> "EngineSnapshot":
        """Compatibility accessor for legacy code that references ``record.data``."""
        return self


# Backwards compatibility aliases -------------------------------------------------
SnapshotRecord = EngineSnapshot

__all__ = [
    "EngineSnapshot",
    "SnapshotRecord",
    "Jacobians",
    "CycleMetrics",
]
