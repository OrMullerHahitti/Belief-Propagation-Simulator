"""Type Definitions for the Snapshot System.

This module contains the `dataclasses` that define the structure for
snapshot configuration, captured data, and analysis artifacts. These types
provide a standardized way to manage and interact with the state of a
simulation at different points in time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class SnapshotsConfig:
    """Configuration for per-step snapshot capture and analysis.

    An instance of this class is passed to the engine to enable and configure
    the snapshotting feature.

    Attributes:
        compute_jacobians: If True, computes the Jacobian matrices (A, P, B).
        compute_block_norms: If True, computes the infinity norms of Jacobian blocks.
        compute_cycles: If True, analyzes cycles in the message-passing graph.
        include_detailed_cycles: If True, includes detailed per-cycle metrics.
        compute_numeric_cycle_gain: If True, estimates the numeric gain for each cycle.
        max_cycle_len: The maximum length of simple cycles to enumerate.
        retain_last: The number of recent snapshots to keep in memory. `None`
            means no limit.
        save_each_step: If True, saves each snapshot to disk automatically.
        save_dir: The directory to save snapshots to if `save_each_step` is True.
    """
    compute_jacobians: bool = True
    compute_block_norms: bool = True
    compute_cycles: bool = True
    include_detailed_cycles: bool = False
    compute_numeric_cycle_gain: bool = False
    max_cycle_len: int = 12
    retain_last: Optional[int] = 25
    save_each_step: bool = False
    save_dir: Optional[str] = None


@dataclass
class SnapshotData:
    """A lightweight, immutable view of the simulation state at a single step.

    This dataclass holds the essential information captured from the engine at
    a specific moment, forming the basis for any further analysis.

    Attributes:
        step: The simulation step index.
        lambda_: The damping factor (lambda) used in the simulation at this step.
        dom: A dictionary mapping variable names to their domain labels.
        N_var: A dictionary mapping variable names to their factor neighbors.
        N_fac: A dictionary mapping factor names to their variable neighbors.
        Q: A dictionary mapping (variable, factor) pairs to the Q-message array.
        R: A dictionary mapping (factor, variable) pairs to the R-message array.
        cost: A dictionary mapping factor names to their cost function accessors.
        unary: A dictionary mapping variable names to their unary potential arrays.
    """
    step: int
    lambda_: float
    dom: Dict[str, List[str]]
    N_var: Dict[str, List[str]]
    N_fac: Dict[str, List[str]]
    Q: Dict[Tuple[str, str], np.ndarray]
    R: Dict[Tuple[str, str], np.ndarray]
    cost: Dict[str, Any] = field(default_factory=dict)
    unary: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Jacobians:
    """Holds the Jacobian-related matrices and artifacts for a snapshot.

    These matrices represent the linearized dynamics of the belief propagation
    update rules and are used for convergence analysis.

    Attributes:
        idxQ: A mapping from (variable, factor, domain_index) to a matrix index.
        idxR: A mapping from (factor, variable, domain_index) to a matrix index.
        A: The sparse matrix representing R -> Q message dependencies.
        P: The sparse projection matrix for the min-sum operator.
        B: The sparse matrix representing Q -> R message dependencies.
        block_norms: A dictionary of computed infinity norms for Jacobian blocks.
    """
    idxQ: Dict[Tuple[str, str, int], int]
    idxR: Dict[Tuple[str, str, int], int]
    A: csr_matrix
    P: csr_matrix
    B: csr_matrix
    block_norms: Optional[Dict[str, float]] = None


@dataclass
class CycleMetrics:
    """A compact summary of cycle analysis for a given step.

    Attributes:
        num_cycles: The total number of simple cycles found.
        aligned_hops_total: The number of cycles containing an "aligned hop,"
            which is relevant for contraction analysis.
        has_certified_contraction: A boolean indicating if a contraction
            condition is met based on block norms.
        details: An optional list of dictionaries with per-cycle information.
    """
    num_cycles: int
    aligned_hops_total: int
    has_certified_contraction: bool
    details: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the cycle metrics."""
        return {
            "num_cycles": self.num_cycles,
            "aligned_hops_total": self.aligned_hops_total,
            "has_certified_contraction": self.has_certified_contraction,
            "details": self.details,
        }


@dataclass
class SnapshotRecord:
    """The full record for a single step, containing raw data and computed artifacts.

    Attributes:
        data: The raw `SnapshotData` captured for the step.
        jacobians: The computed `Jacobians` for the step, if requested.
        cycles: The computed `CycleMetrics` for the step, if requested.
        winners: A dictionary containing pre-computed "winning" assignments for
            message calculations.
        min_idx: A dictionary containing the index of the minimum value for
            each Q-message.
    """
    data: SnapshotData
    jacobians: Optional[Jacobians] = None
    cycles: Optional[CycleMetrics] = None
    winners: Optional[Dict[Tuple[str, str, str], Dict[str, str]]] = None
    min_idx: Optional[Dict[Tuple[str, str], int]] = None
