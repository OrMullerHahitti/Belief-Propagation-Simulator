"""Snapshot Builders and Adapters.

This module provides utility functions for translating the state of a simulation
engine at a specific step into a `SnapshotData` object. This standardized
snapshot format is designed to be consumed by downstream analysis components,
such as those for calculating Jacobians or analyzing message cycles.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

import numpy as np

from ..bp.engine_components import Step
from ..core.agents import VariableAgent, FactorAgent
from .types import SnapshotData


def _labels_for_domain(domain_size: int) -> List[str]:
    """Generates string labels for a given domain size."""
    return [str(i) for i in range(domain_size)]


def _normalize_min_zero(arr: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array by subtracting its minimum value."""
    if arr.size == 0:
        return arr
    m = float(np.min(arr))
    return arr - m


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy/scalar types to Python builtins.

    For non-serializable objects (e.g., Computator instances), converts to string representation.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    # Fallback for non-serializable objects: convert to string representation
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(type(obj).__name__)


def _belief_scalar(value: Any) -> float:
    """Convert a belief-like value into a scalar representative."""
    if isinstance(value, np.ndarray):
        return float(np.min(value)) if value.size else 0.0
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
        return float(np.min(arr)) if arr.size else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _assignment_scalar(value: Any) -> int:
    """Coerce assignment-like value to an int."""
    try:
        return int(value.item()) if isinstance(value, np.integer) else int(value)
    except Exception:
        return 0


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

    # Create callable accessors for factor cost tables and persist arrays
    cost: Dict[str, Any] = {}
    cost_tables: Dict[str, np.ndarray] = {}
    cost_labels: Dict[str, List[str]] = {}
    for f in factors:
        table = getattr(f, "cost_table", None)
        conn = getattr(f, "connection_number", {})
        if table is None or not conn:
            continue

        arr = np.asarray(table, dtype=float).copy()
        var_by_dim = sorted(conn.items(), key=lambda kv: kv[1])
        var_order = [name for name, _ in var_by_dim]

        cost_tables[f.name] = arr
        cost_labels[f.name] = var_order

        def make_cost_fn(ct: np.ndarray, order: List[str]):
            def _cost(assign: Dict[str, str]) -> float:
                try:
                    idx: List[int] = [int(assign[v]) for v in order]
                    return float(ct[tuple(idx)])
                except Exception:
                    return 0.0
            return _cost

        cost[f.name] = make_cost_fn(arr, var_order)

    # Runtime beliefs/assignments/cost sourced from history when available.
    beliefs: Dict[str, float] = {}
    assignments: Dict[str, int] = {}
    global_cost = None
    metadata: Dict[str, Any] = {
        "engine": engine.__class__.__name__,
        "graph_diameter": getattr(engine, "graph_diameter", None),
        "num_variables": len(variables),
        "num_factors": len(factors),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    computator = getattr(engine, "computator", None)
    metadata["computator"] = getattr(computator, "__class__.__name__", None)

    history = getattr(engine, "history", None)
    if history is not None:
        metadata["history_config"] = _to_builtin(getattr(history, "config", {}))
        metadata["history_name"] = getattr(history, "name", None)
        metadata["use_bct_history"] = bool(getattr(history, "use_bct_history", False))

        if getattr(history, "use_bct_history", False):
            beliefs = {
                str(var): _belief_scalar(val)
                for var, val in getattr(history, "step_beliefs", {}).get(step_idx, {}).items()
            }
            assignments = {
                str(var): _assignment_scalar(val)
                for var, val in getattr(history, "step_assignments", {}).get(step_idx, {}).items()
            }
            step_costs = getattr(history, "step_costs", [])
            if 0 <= step_idx < len(step_costs):
                try:
                    global_cost = float(step_costs[step_idx])
                    metadata["global_cost_source"] = "history.step_costs"
                except Exception:
                    global_cost = None

    if not beliefs and hasattr(engine, "get_beliefs"):
        try:
            engine_beliefs = getattr(engine, "get_beliefs")()
            beliefs = {str(k): _belief_scalar(v) for k, v in engine_beliefs.items()}
            metadata.setdefault("global_cost_source", "engine.get_beliefs")
        except Exception:
            beliefs = {}

    if not assignments and hasattr(engine, "assignments"):
        try:
            assignments = {
                str(k): _assignment_scalar(v)
                for k, v in getattr(engine, "assignments", {}).items()
            }
        except Exception:
            assignments = {}

    if global_cost is None and hasattr(engine, "calculate_global_cost"):
        try:
            global_cost = float(engine.calculate_global_cost())
            metadata["global_cost_source"] = "engine.calculate_global_cost"
        except Exception:
            metadata.setdefault("global_cost_source", "unavailable")

    convergence_monitor = getattr(engine, "convergence_monitor", None)
    if convergence_monitor is not None:
        try:
            summary = convergence_monitor.get_convergence_summary()
            if summary:
                metadata["convergence_summary"] = _to_builtin(summary)
        except Exception:
            pass

    performance_monitor = getattr(engine, "performance_monitor", None)
    if performance_monitor is not None:
        try:
            perf_summary = performance_monitor.get_summary()
            if perf_summary:
                metadata["performance_summary"] = _to_builtin(perf_summary)
        except Exception:
            pass

    metadata = _to_builtin(metadata)

    return SnapshotData(
        step=step_idx,
        lambda_=lambda_val,
        dom=dom,
        N_var=N_var,
        N_fac=N_fac,
        Q=Q,
        R=R,
        cost=cost,
        cost_tables=cost_tables,
        cost_labels=cost_labels,
        unary=unary,
        beliefs=beliefs,
        assignments=assignments,
        global_cost=global_cost,
        metadata=metadata,
    )
