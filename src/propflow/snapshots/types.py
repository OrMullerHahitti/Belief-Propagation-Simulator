from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.sparse import csr_matrix


"""
Example SnapshotsConfig (with line-by-line comments):

# SnapshotsConfig(
#     compute_jacobians=True,          # Build A, P, B and slot indices (idxQ/idxR)
#     compute_block_norms=True,         # Compute block norms and an upper bound for ||M||_inf
#     compute_cycles=True,              # Analyze cycles on the slot graph
#     include_detailed_cycles=False,    # When True, include per-cycle aligned hop details
#     compute_numeric_cycle_gain=False, # Estimate Infinity-norm numeric gain per cycle
#     max_cycle_len=12,                 # Maximum simple-cycle length to enumerate
#     retain_last=25,                   # Keep last N snapshots in memory (None = unlimited)
# )
"""


@dataclass
class SnapshotsConfig:
    """
    Configuration for per-step snapshot capture and analysis.

    Pass an instance of this class to `BPEngine(..., snapshots_config=SnapshotsConfig(...))`
    and snapshots will be captured automatically each step.
    """

    compute_jacobians: bool = True
    compute_block_norms: bool = True
    compute_cycles: bool = True
    include_detailed_cycles: bool = False
    compute_numeric_cycle_gain: bool = False
    max_cycle_len: int = 12
    retain_last: Optional[int] = 25
    # Optional persistence controls
    save_each_step: bool = False
    save_dir: Optional[str] = None


@dataclass
class SnapshotData:
    """
    Lightweight, immutable view of a single step.

    Q and R map 2-tuples (node, neighbor) to 1D arrays of shape (|domain|,)
    that are already min-normalized where applicable for Q.
    """

    step: int
    lambda_: float
    dom: Dict[str, List[str]]
    N_var: Dict[str, List[str]]
    N_fac: Dict[str, List[str]]
    Q: Dict[Tuple[str, str], np.ndarray]
    R: Dict[Tuple[str, str], np.ndarray]
    # Optional: raw cost table-backed callables keyed by factor name
    cost: Dict[str, Any] = field(default_factory=dict)
    # Optional unaries (not required for Jacobian construction here)
    unary: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Jacobians:
    """
    Jacobian-related artifacts for a snapshot.
    """

    idxQ: Dict[Tuple[str, str, int], int]
    idxR: Dict[Tuple[str, str, int], int]
    A: csr_matrix
    P: csr_matrix
    B: csr_matrix
    block_norms: Optional[Dict[str, float]] = None


@dataclass
class CycleMetrics:
    """
    Compact summary of cycle analysis for a step.
    """

    num_cycles: int
    aligned_hops_total: int
    has_certified_contraction: bool
    # Keep details optional and small by default
    details: Optional[List[Dict[str, Any]]] = None


@dataclass
class SnapshotRecord:
    """
    Full record stored per step: raw snapshot data plus computed artifacts.
    """

    data: SnapshotData
    jacobians: Optional[Jacobians] = None
    cycles: Optional[CycleMetrics] = None
    # Keep useful intermediates for downstream uses
    winners: Optional[Dict[Tuple[str, str, str], Dict[str, str]]] = None
    min_idx: Optional[Dict[Tuple[str, str], int]] = None
