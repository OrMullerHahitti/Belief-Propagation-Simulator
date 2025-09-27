"""Snapshot Builders and Adapters.

This module provides utility functions for translating the state of a simulation
engine at a specific step into a `SnapshotData` object. This standardized
snapshot format is designed to be consumed by downstream analysis components,
such as those for calculating Jacobians or analyzing message cycles.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np

from ..bp.engine_components import Step
from ..core.agents import VariableAgent, FactorAgent
from .types import SnapshotData


def _labels_for_domain(domain_size: int) -> List[str]:
    """Generates string labels for a given domain size."""
    return [str(i) for i in range(int(domain_size))]


def _normalize_min_zero(arr: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array by subtracting its minimum value."""
    if arr.size == 0:
        return arr
    m = float(np.min(arr))
    return arr - m


def extract_qr_from_step(
    step: Step,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """Extracts and formats Q and R messages from a `Step` object.

    Args:
        step: The `Step` object containing captured messages for a single
            simulation step.

    Returns:
        A tuple containing two dictionaries:
        - Q: A mapping from (variable_name, factor_name) to the message array.
        - R: A mapping from (factor_name, variable_name) to the message array.
    """
    Q: Dict[Tuple[str, str], np.ndarray] = {}
    R: Dict[Tuple[str, str], np.ndarray] = {}

    # Variable -> Factor messages (Q)
    for var_name, msgs in step.q_messages.items():
        for msg in msgs:
            key = (var_name, getattr(msg.recipient, "name", str(msg.recipient)))
            data = np.asarray(getattr(msg, "data", np.array([])), dtype=float)
            Q[key] = _normalize_min_zero(data)

    # Factor -> Variable messages (R)
    for fac_name, msgs in step.r_messages.items():
        for msg in msgs:
            key = (fac_name, getattr(msg.recipient, "name", str(msg.recipient)))
            data = np.asarray(getattr(msg, "data", np.array([])), dtype=float)
            R[key] = data

    return Q, R


def build_snapshot_from_engine(step_idx: int, step: Step, engine: Any) -> SnapshotData:
    """Constructs a `SnapshotData` instance from the current state of an engine.

    This function captures a comprehensive view of the simulation at a specific
    step, including graph topology, variable domains, messages, damping factor,
    and cost functions.

    Args:
        step_idx: The index of the simulation step being captured.
        step: The `Step` object containing the message data for this step.
        engine: The simulation engine instance, from which graph structure
            and other parameters are extracted.

    Returns:
        A `SnapshotData` object populated with the state of the simulation.
    """
    variables: List[VariableAgent] = list(engine.var_nodes)
    factors: List[FactorAgent] = list(engine.factor_nodes)

    # Extract domains and neighborhoods from the graph
    dom: Dict[str, List[str]] = {
        v.name: _labels_for_domain(int(getattr(v, "domain", 0))) for v in variables
    }
    N_var: Dict[str, List[str]] = {
        v.name: [getattr(n, "name", str(n)) for n in engine.graph.G.neighbors(v)]
        for v in variables
    }
    N_fac: Dict[str, List[str]] = {
        f.name: [getattr(n, "name", str(n)) for n in engine.graph.G.neighbors(f)]
        for f in factors
    }

    # Extract messages from the step
    Q, R = extract_qr_from_step(step)

    # Infer damping factor and create unary potentials
    lambda_val = float(getattr(engine, "damping_factor", 0.0))
    unary: Dict[str, np.ndarray] = {
        v.name: np.zeros(int(getattr(v, "domain", 0))) for v in variables
    }

    # Create callable accessors for factor cost tables
    cost: Dict[str, Any] = {}
    for f in factors:
        table = getattr(f, "cost_table", None)
        conn = getattr(f, "connection_number", {})
        if table is None or not conn:
            continue

        var_by_dim = sorted(conn.items(), key=lambda kv: kv[1])
        var_order = [name for name, _ in var_by_dim]

        def make_cost_fn(ct: np.ndarray, order: List[str]):
            def _cost(assign: Dict[str, str]) -> float:
                try:
                    idx: List[int] = [int(assign[v]) for v in order]
                    return float(ct[tuple(idx)])
                except Exception:
                    return 0.0
            return _cost

        cost[f.name] = make_cost_fn(np.asarray(table), var_order)

    return SnapshotData(
        step=step_idx,
        lambda_=lambda_val,
        dom=dom,
        N_var=N_var,
        N_fac=N_fac,
        Q=Q,
        R=R,
        cost=cost,
        unary=unary,
    )
