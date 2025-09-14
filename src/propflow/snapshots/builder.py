from __future__ import annotations

"""
Snapshot builders and adapters.

These utilities translate engine/step state into a minimal, analysis-ready
SnapshotData that downstream components (Jacobians, cycles) can consume.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from ..bp.engine_components import Step
from ..core.agents import VariableAgent, FactorAgent
from .types import SnapshotData


def _labels_for_domain(domain_size: int) -> List[str]:
    return [str(i) for i in range(int(domain_size))]


def _normalize_min_zero(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    m = float(np.min(arr))
    return arr - m


def extract_qr_from_step(
    step: Step,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """
    Build Q and R mappings from the Step's captured message objects.

    Q: (u, f) -> ndarray(|X_u|,)
    R: (f, v) -> ndarray(|X_v|,)
    """
    Q: Dict[Tuple[str, str], np.ndarray] = {}
    R: Dict[Tuple[str, str], np.ndarray] = {}

    # Variable -> Factor messages captured in step.q_messages
    for var_name, msgs in step.q_messages.items():
        for msg in msgs:
            # Expect msg.recipient is FactorAgent and msg.data is (|domain_u|,)
            key = (var_name, getattr(msg.recipient, "name", str(msg.recipient)))
            data = np.asarray(getattr(msg, "data", np.array([])), dtype=float)
            Q[key] = _normalize_min_zero(data)

    # Factor -> Variable messages captured in step.r_messages
    for fac_name, msgs in step.r_messages.items():
        for msg in msgs:
            key = (fac_name, getattr(msg.recipient, "name", str(msg.recipient)))
            data = np.asarray(getattr(msg, "data", np.array([])), dtype=float)
            R[key] = data

    return Q, R


def build_snapshot_from_engine(step_idx: int, step: Step, engine) -> SnapshotData:
    """
    Construct a SnapshotData instance from the engine and the captured Step.

    - Uses current graph topology for N_var/N_fac and variable domains
    - Uses Step-captured messages for Q/R
    - Infers lambda (damping) when present, otherwise 0.0
    - Wraps factor cost tables as callable accessors when available
    """
    # Domains and labels
    dom: Dict[str, List[str]] = {}
    variables: List[VariableAgent] = list(engine.var_nodes)
    factors: List[FactorAgent] = list(engine.factor_nodes)

    for v in variables:
        dom[v.name] = _labels_for_domain(int(getattr(v, "domain", 0)))

    # Neighborhoods
    N_var: Dict[str, List[str]] = {}
    N_fac: Dict[str, List[str]] = {}

    for v in variables:
        neighbors = [n for n in engine.graph.G.neighbors(v)]
        N_var[v.name] = [getattr(f, "name", str(f)) for f in neighbors]

    for f in factors:
        neighbors = [n for n in engine.graph.G.neighbors(f)]
        N_fac[f.name] = [getattr(v, "name", str(v)) for v in neighbors]

    # Messages from this step
    Q, R = extract_qr_from_step(step)

    # Damping (lambda)
    lambda_val = float(getattr(engine, "damping_factor", 0.0))

    # Optional unary (zeros by default)
    unary: Dict[str, np.ndarray] = {
        v.name: np.zeros(int(getattr(v, "domain", 0))) for v in variables
    }

    # Factor cost mapping
    cost: Dict[str, Any] = {}
    for f in factors:
        table = getattr(f, "cost_table", None)
        conn = getattr(f, "connection_number", {})
        if table is None or not conn:
            continue

        # Closure capturing cost table and variable order
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
