import numpy as np
from typing import List, Dict, Tuple
from propflow.core.agents import VariableAgent, FactorAgent
from propflow.bp.factor_graph import FactorGraph
from propflow.bp.engine_base import BPEngine
from propflow.bp.engines import QRDampingEngine
from propflow.snapshots.types import EngineSnapshot
from propflow.bp.computators import (
    MinSumComputator,
    MaxSumComputator,
    SumProductComputator,
)

from .models import GraphSpec, SnapshotJSON

# maps engine_type string to computator class
COMPUTATOR_MAP = {
    "min_sum": MinSumComputator,
    "max_sum": MaxSumComputator,
    "sum_product": SumProductComputator,
}

def build_graph(spec: GraphSpec) -> FactorGraph:
    """Constructs a FactorGraph from the GraphSpec."""
    var_agents = [VariableAgent(name=v.name, domain=v.domain_size) for v in spec.variables]
    var_map = {v.name: v for v in var_agents}

    factor_agents = []
    edges = {}

    # create factors from spec
    for f_spec in spec.factors:
        for v_name in f_spec.neighbors:
            if v_name not in var_map:
                raise ValueError(f"Factor {f_spec.name} refers to unknown variable {v_name}")

        # compute_R in propflow assumes the factor inbox index matches the
        # cost-table dimension. the inbox is keyed by sender name so it ends
        # up alphabetical. if the user's neighbors aren't alphabetical we
        # reorder them and transpose the cost table so dim i maps to the
        # i-th sorted neighbor. this preserves the user's intended semantics
        # without touching the core library.
        neighbor_names = list(f_spec.neighbors)
        perm = sorted(range(len(neighbor_names)), key=lambda i: neighbor_names[i])
        sorted_neighbors = [neighbor_names[i] for i in perm]
        cost_matrix = np.array(f_spec.cost_table, dtype=float)
        if perm != list(range(len(neighbor_names))):
            cost_matrix = np.transpose(cost_matrix, axes=perm)

        factor = FactorAgent.create_from_cost_table(name=f_spec.name, cost_table=cost_matrix)
        connected_vars = [var_map[v_name] for v_name in sorted_neighbors]

        factor_agents.append(factor)
        edges[factor] = connected_vars

    # create unary factors for variables with unary costs
    for v_spec in spec.variables:
        if v_spec.unary_cost is not None:
            unary_cost = np.array(v_spec.unary_cost, dtype=float)
            unary_factor = FactorAgent.create_from_cost_table(
                name=f"u_{v_spec.name}",
                cost_table=unary_cost
            )
            factor_agents.append(unary_factor)
            edges[unary_factor] = [var_map[v_spec.name]]

    return FactorGraph(var_agents, factor_agents, edges)

def serialize_snapshot(snapshot: EngineSnapshot) -> SnapshotJSON:
    """Converts EngineSnapshot to JSON-friendly dict."""
    
    def format_msg(msg_dict: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, List[float]]:
        out = {}
        for (src, dst), arr in msg_dict.items():
            key = f"{src}->{dst}"
            out[key] = arr.tolist() if isinstance(arr, np.ndarray) else list(arr)
        return out

    # Convert cost tables to list of lists (or list for unary)
    serializable_tables = {}
    for k, v in snapshot.cost_tables.items():
        if isinstance(v, np.ndarray):
             serializable_tables[k] = v.tolist()
        else:
             serializable_tables[k] = v

    return SnapshotJSON(
        step=snapshot.step,
        dom=snapshot.dom,
        Q=format_msg(snapshot.Q),
        R=format_msg(snapshot.R),
        assignments=snapshot.assignments,
        global_cost=snapshot.global_cost,
        cost_tables=serializable_tables,
        cost_labels=snapshot.cost_labels
    )

def run_simulation(spec: GraphSpec) -> List[SnapshotJSON]:
    """Runs the BP simulation and returns list of snapshots."""
    graph = build_graph(spec)

    # select computator based on engine_type
    op_mode = spec.config.engine_type.lower()
    computator_cls = COMPUTATOR_MAP.get(op_mode, MinSumComputator)
    computator = computator_cls()

    q_damping = float(spec.config.damping or 0.0)
    r_damping = float(getattr(spec.config, "r_damping", 0.0) or 0.0)

    # Use a combined damping engine if either damping factor is enabled.
    if q_damping > 0 or r_damping > 0:
        engine = QRDampingEngine(
            graph,
            computator=computator,
            q_damping_factor=q_damping,
            r_damping_factor=r_damping,
        )
    else:
        engine = BPEngine(graph, computator=computator)

    # run through the full engine lifecycle so cycle-level events
    # (normalize_inbox, convergence checks) fire. the prior manual step
    # loop skipped these and left the web sim without normalization or
    # early termination.
    engine.run(max_iter=spec.config.max_iters)
    return [serialize_snapshot(snap) for snap in engine.snapshots]
