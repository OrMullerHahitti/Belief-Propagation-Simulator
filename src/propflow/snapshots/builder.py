"""Utilities for constructing `EngineSnapshot` instances."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..bp.engine_components import Step
from ..core.agents import FactorAgent, VariableAgent
from .types import EngineSnapshot


def _labels_for_domain(domain_size: int) -> List[str]:
    return [str(i) for i in range(domain_size)]


def _normalize_min_zero(arr: np.ndarray) -> np.ndarray:
    return arr if arr.size == 0 else arr - float(np.min(arr))


def extract_qr_from_step(
    step: Step,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    Q: Dict[Tuple[str, str], np.ndarray] = {}
    R: Dict[Tuple[str, str], np.ndarray] = {}

    for var_name, messages in step.q_messages.items():
        for msg in messages:
            recipient = getattr(msg.recipient, "name", str(msg.recipient))
            data = np.asarray(getattr(msg, "data", np.array([])), dtype=float)
            Q[(var_name, recipient)] = _normalize_min_zero(data)

    for fac_name, messages in step.r_messages.items():
        for msg in messages:
            recipient = getattr(msg.recipient, "name", str(msg.recipient))
            data = np.asarray(getattr(msg, "data", np.array([])), dtype=float)
            R[(fac_name, recipient)] = data

    return Q, R


def _extract_cost_tables(
    factors: List[FactorAgent],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    tables: Dict[str, np.ndarray] = {}
    labels: Dict[str, List[str]] = {}
    for factor in factors:
        table = getattr(factor, "cost_table", None)
        conn = getattr(factor, "connection_number", {})
        if table is None or not conn:
            continue
        arr = np.asarray(table, dtype=float).copy()
        ordering = [var for var, _ in sorted(conn.items(), key=lambda item: item[1])]
        tables[factor.name] = arr
        labels[factor.name] = ordering
    return tables, labels


def build_snapshot_from_engine(
    step_idx: int, step: Step, engine: Any
) -> EngineSnapshot:
    variables: List[VariableAgent] = list(engine.var_nodes)
    factors: List[FactorAgent] = list(engine.factor_nodes)

    graph = getattr(engine.graph, "G", None)

    def _neighbour_names(node: Any) -> List[str]:
        if graph is None:
            return []
        try:
            neighbours = graph.neighbors(node)
        except Exception:
            return []
        return [getattr(neighbour, "name", str(neighbour)) for neighbour in neighbours]

    dom: Dict[str, List[str]] = {
        var.name: _labels_for_domain(int(getattr(var, "domain", 0)))
        for var in variables
    }
    N_var: Dict[str, List[str]] = {var.name: _neighbour_names(var) for var in variables}
    N_fac: Dict[str, List[str]] = {fac.name: _neighbour_names(fac) for fac in factors}

    Q, R = extract_qr_from_step(step)

    lambda_val = float(getattr(engine, "damping_factor", 0.0))
    unary: Dict[str, np.ndarray] = {
        var.name: np.zeros(int(getattr(var, "domain", 0))) for var in variables
    }

    cost_tables, cost_labels = _extract_cost_tables(factors)

    beliefs: Dict[str, np.ndarray] = {}
    if hasattr(engine, "get_beliefs"):
        try:
            for name, belief in engine.get_beliefs().items():
                if belief is None:
                    continue
                beliefs[name] = np.asarray(belief, dtype=float)
        except Exception:
            beliefs = {}

    assignments: Dict[str, int] = {}
    try:
        for name, value in engine.assignments.items():
            assignments[name] = int(value)
    except Exception:
        assignments = {}

    global_cost = None
    try:
        global_cost = float(engine.calculate_global_cost())
    except Exception:
        global_cost = None

    metadata = {
        "engine": engine.__class__.__name__,
        "graph_diameter": getattr(engine, "graph_diameter", None),
        "num_variables": len(variables),
        "num_factors": len(factors),
        "message_counts": {"Q": len(Q), "R": len(R)},
        "use_bct_history": bool(getattr(engine, "use_bct_history", False)),
    }

    return EngineSnapshot(
        step=step_idx,
        lambda_=lambda_val,
        dom=dom,
        N_var=N_var,
        N_fac=N_fac,
        Q=Q,
        R=R,
        unary=unary,
        beliefs=beliefs,
        assignments=assignments,
        global_cost=global_cost,
        metadata=metadata,
        cost_tables=cost_tables,
        cost_labels=cost_labels,
    )


__all__ = ["build_snapshot_from_engine"]
