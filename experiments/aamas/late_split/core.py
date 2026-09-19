"""Experiment-local native execution, portable checkpoints, and menu merging."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.aaai.code.engines import DampedMidRunSplitEngine
from experiments.aaai.code.merge import (
    branch_and_bound,
    mgm1_binary_merge,
    score_assignment,
)
from experiments.aaai.code.problems import capture_original
from propflow import FactorAgent, FactorGraph, MinSumComputator, VariableAgent
from propflow.core.components import Message
from propflow.policies.convergance import ConvergenceConfig


@dataclass(frozen=True)
class Config:
    """Approved observation window, continuation horizon, and search limits."""

    prefix_steps: int = 1000
    post_steps: int = 1000
    tail_steps: int = 100
    damping: float = 0.9
    split: float = 0.5
    bb_seconds: float = 300.0
    mgm_rounds: int = 10000

    def __post_init__(self) -> None:
        if self.prefix_steps < 1 or not 4 <= self.tail_steps <= self.post_steps:
            raise ValueError(
                "require prefix_steps >= 1 and 4 <= tail_steps <= post_steps"
            )
        if not 0 <= self.damping < 1 or not 0 < self.split < 1:
            raise ValueError("invalid damping or split weight")
        if not np.isfinite(self.bb_seconds) or self.bb_seconds <= 0:
            raise ValueError("bb_seconds must be positive and finite")
        if self.mgm_rounds < 1:
            raise ValueError("mgm_rounds must be positive")


@dataclass
class TraceSnapshot:
    """Small per-update snapshot; full messages are kept only at checkpoints."""

    step: int
    assignments: dict[str, int]
    global_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceSnapshots:
    """Capture the assignment before the native cycle-normalization event."""

    def capture_step(self, step_index: int, step: Any, engine: Any) -> TraceSnapshot:
        return TraceSnapshot(
            step_index, {name: int(x) for name, x in engine.assignments.items()}
        )


class ReleasedDampingSplitEngine(DampedMidRunSplitEngine):
    """Use ordinary native DMS until splitting, then native undamped updates."""

    def step(self, i: int = 0):
        if i >= self.split_at_iter:
            self.damping_factor = 0.0
        return super().step(i)

    def post_var_compute(self, var) -> None:
        if not self._split_applied:
            super().post_var_compute(var)


def make_engine(graph: FactorGraph, config: Config, split_at: int):
    """Construct the approved native transfer-mode engine without early stopping."""
    return ReleasedDampingSplitEngine(
        graph,
        computator=MinSumComputator(),
        damping_factor=config.damping,
        split_factor=config.split,
        split_at_iter=split_at,
        transfer_mode="transfer",
        normalize_messages=True,
        anytime=False,
        snapshot_manager=TraceSnapshots(),
    )


def advance(engine, iteration: int) -> TraceSnapshot:
    """Run exactly one native update and its original cycle events."""
    engine.step(iteration)
    snapshot = engine.get_snapshot(iteration)
    if snapshot is None or not np.isfinite(snapshot.global_cost):
        raise RuntimeError(f"missing/nonfinite cost at iteration {iteration}")
    try:
        engine._handle_cycle_events(iteration)
    except StopIteration:
        pass
    return snapshot


def json_value(value):
    """Convert numeric evidence to strict JSON without losing float precision."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    """Write one complete result atomically; reject nonfinite numeric evidence."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _input_spec(graph: FactorGraph) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = {}
    factors = []
    for i, factor in enumerate(graph.factors):
        key = f"table_{i}"
        arrays[key] = np.array(factor.cost_table, copy=True)
        axes = sorted(factor.connection_number, key=factor.connection_number.get)
        factors.append({"name": factor.name, "axes": axes, "key": key})
    spec = {
        "variables": [(v.name, v.domain) for v in graph.variables],
        "factors": factors,
    }
    return spec, arrays


def input_fingerprint(graph: FactorGraph) -> str:
    """Hash tables, dtypes, graph order, and semantic factor-axis order."""
    spec, arrays = _input_spec(graph)
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for table in arrays.values():
        h.update(str(table.dtype).encode())
        h.update(str(table.shape).encode())
        h.update(table.tobytes())
    return h.hexdigest()


def save_input(graph: FactorGraph, path: Path) -> None:
    """Save exact original tables, preserving dtypes and insertion/axis order."""
    spec, arrays = _input_spec(graph)
    np.savez_compressed(path, spec=json.dumps(spec), **arrays)


def load_input(path: Path) -> FactorGraph:
    """Rebuild a native graph from the saved tables, without random regeneration."""
    with np.load(path, allow_pickle=False) as archive:
        spec = json.loads(str(archive["spec"]))
        variables = [VariableAgent(name, domain) for name, domain in spec["variables"]]
        by_name = {v.name: v for v in variables}
        factors, edges = [], {}
        for row in spec["factors"]:
            factor = FactorAgent.create_from_cost_table(
                row["name"], archive[row["key"]]
            )
            factors.append(factor)
            edges[factor] = [by_name[name] for name in row["axes"]]
    return FactorGraph(variables, factors, edges)


def _pack_messages(messages) -> list[dict]:
    return [
        {"sender": m.sender.name, "recipient": m.recipient.name, "data": m.data.copy()}
        for m in messages
    ]


@dataclass
class Checkpoint:
    """Full dynamic state after cycle events, for the fixed unsplit native engine."""

    next_iteration: int
    input_sha256: str
    graph_diameter: int
    damping: float
    last_cost: float
    nodes: dict[str, dict]
    monitor: dict

    @classmethod
    def capture(cls, engine, next_iteration: int, fingerprint: str) -> Checkpoint:
        """Copy mailboxes, all retained damping history, monitor and update phase."""
        if engine._split_applied:
            raise ValueError("only unsplit checkpoints are supported")
        nodes = {}
        for node in engine.graph.G.nodes():
            nodes[node.name] = {
                "incoming": _pack_messages(node.mailer.inbox),
                "outgoing": _pack_messages(node.mailer.outbox),
                "history": [_pack_messages(messages) for messages in node._history],
                "max_history": node._max_history,
            }
        monitor = deepcopy(vars(engine.convergence_monitor))
        monitor["config"] = asdict(engine.convergence_monitor.config)
        return cls(
            next_iteration,
            fingerprint,
            engine.graph_diameter,
            engine.damping_factor,
            engine._last_cost,
            nodes,
            monitor,
        )

    def save(self, path: Path) -> None:
        """Save a portable numeric checkpoint; no pickle or executable objects."""
        with gzip.open(path, "wt") as stream:
            json.dump(asdict(self), stream, default=json_value, allow_nan=False)

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load a checkpoint saved by this experiment."""
        with gzip.open(path, "rt") as stream:
            return cls(**json.load(stream))

    def restore(self, engine) -> None:
        """Restore exact state into a fresh engine on the same original graph."""
        if (
            engine._split_applied
            or input_fingerprint(engine.graph) != self.input_sha256
        ):
            raise ValueError("checkpoint graph differs from the saved original input")
        if engine.graph_diameter != self.graph_diameter:
            raise ValueError("checkpoint normalization period differs")
        if engine.split_at_iter < self.next_iteration:
            raise ValueError("split precedes the checkpoint")
        by_name = {node.name: node for node in engine.graph.G.nodes()}
        if set(by_name) != set(self.nodes):
            raise ValueError("checkpoint nodes differ")

        def unpack(records):
            return [
                Message(
                    np.array(r["data"], copy=True),
                    by_name[r["sender"]],
                    by_name[r["recipient"]],
                )
                for r in records
            ]

        for name, state in self.nodes.items():
            node = by_name[name]
            node.mailer.inbox = unpack(state["incoming"])
            node.mailer.outbox = unpack(state["outgoing"])
            node._history = [unpack(messages) for messages in state["history"]]
            node._max_history = state["max_history"]
        monitor = deepcopy(self.monitor)
        monitor["config"] = ConvergenceConfig(**monitor["config"])
        if monitor["prev_beliefs"] is not None:
            monitor["prev_beliefs"] = {
                k: np.array(v, copy=True) for k, v in monitor["prev_beliefs"].items()
            }
        vars(engine.convergence_monitor).update(monitor)
        engine.damping_factor = self.damping
        engine._last_cost = self.last_cost


def trace_arrays(snapshots: list[TraceSnapshot], names: list[str]) -> dict:
    """Convert public snapshot results into compact arrays."""
    return {
        "iterations": np.array([s.step for s in snapshots], dtype=np.int64),
        "costs": np.array([s.global_cost for s in snapshots], dtype=float),
        "assignments": np.array(
            [[s.assignments[v] for v in names] for s in snapshots], dtype=np.int64
        ),
        "variable_names": np.array(names),
    }


def verify_costs(graph: FactorGraph, trace: dict) -> float:
    """Independently reconstruct every recorded cost on original ordered tables."""
    names = list(trace["variable_names"])
    assignments = trace["assignments"]
    index = {v: i for i, v in enumerate(names)}
    values = np.zeros(len(assignments))
    for factor in graph.factors:
        axes = sorted(factor.connection_number, key=factor.connection_number.get)
        values += factor.cost_table[tuple(assignments[:, index[v]] for v in axes)]
    error = float(np.max(np.abs(values - trace["costs"])))
    if not np.all(np.isfinite(values)) or error > 1e-8:
        raise RuntimeError(f"original-cost reconstruction failed: {error}")
    return error


def tail_kind(assignments: np.ndarray, window: int) -> str:
    """Classify exact decoded-assignment repetition; no message-cycle claim."""
    if len(assignments) < window or window < 4:
        raise ValueError("insufficient tail")
    tail = assignments[-window:]
    if np.all(tail == tail[-1]):
        return "fixed"
    if np.array_equal(tail[2:], tail[:-2]):
        return "period_two"
    return "other"


def merge_tail(graph: FactorGraph, trace: dict, config: Config) -> dict:
    """MGM from both parity assignments, then capped B&B on the same menus."""
    kind = tail_kind(trace["assignments"], config.tail_steps)
    result = {"tail_kind": kind, "branch_costs": trace["costs"][-2:].tolist()}
    if kind == "other":
        return {**result, "status": "no_verified_two_cycle", "mgm": None, "bb": None}
    names, factor_vars, tables = capture_original(graph)
    order = list(trace["variable_names"])
    a, b = [dict(zip(order, map(int, row))) for row in trace["assignments"][-2:]]
    candidates = []
    for start in ["branch1", "branch2"]:
        assignment, costs, moves = mgm1_binary_merge(
            a, b, start, names, factor_vars, tables, max_rounds=config.mgm_rounds
        )
        candidates.append(
            {
                "assignment": assignment,
                "cost": score_assignment(assignment, tables, factor_vars),
                "costs": costs,
                "rounds": len(moves),
                "hit_round_cap": len(moves) == config.mgm_rounds,
            }
        )
    mgm = min(candidates, key=lambda row: row["cost"])
    domains = {v: sorted({a[v], b[v]}) for v in names}
    # condition first, as in the previous harness, to avoid needlessly loose bounds
    reduced = {
        f: tables[f][np.ix_(*[domains[v] for v in axes])]
        for f, axes in factor_vars.items()
    }
    positions = {v: list(range(len(domains[v]))) for v in names}
    warm = {v: domains[v].index(mgm["assignment"][v]) for v in names}
    cost, selected, stats = branch_and_bound(
        names,
        factor_vars,
        reduced,
        positions,
        initial_upper_bound=mgm["cost"],
        initial_assignment=warm,
        time_limit_s=config.bb_seconds,
    )
    assignment = {v: domains[v][selected[v]] for v in names}
    actual = score_assignment(assignment, tables, factor_vars)
    if abs(actual - cost) > 1e-8 or actual > mgm["cost"] + 1e-8:
        raise RuntimeError("B&B incumbent failed independent scoring")
    if mgm["cost"] > min(result["branch_costs"]) + 1e-8:
        raise RuntimeError("MGM worsened both starting branches")
    if any(assignment[v] not in domains[v] for v in names):
        raise RuntimeError("B&B assignment is outside the two-branch menus")
    return {
        **result,
        "status": "no_op" if kind == "fixed" else "merged",
        "branches": [a, b],
        "mgm_starts": candidates,
        "mgm": mgm,
        "bb": {"cost": actual, "assignment": assignment, **stats},
    }
