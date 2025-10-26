"""Core analysis utilities for parsed snapshot records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

from .math_utils import (
    binary_thresholds,
    check_binary_neutral,
    multilabel_gaps,
)
from .snapshot_parser import MessageRecord, SnapshotRecord


@dataclass(slots=True)
class _Coordinate:
    """Internal helper describing a single difference coordinate."""

    kind: str
    sender: str
    recipient: str
    label: int

    def key(self) -> tuple[str, str, str, int]:
        return (self.kind, self.sender, self.recipient, self.label)


class SnapshotAnalyzer:
    """Derive difference coordinates and structural metrics from snapshots."""

    _ZERO_TOL = 1e-9

    def __init__(
        self,
        snapshots: Sequence[SnapshotRecord],
        *,
        domain: Mapping[str, int] | None = None,
        max_cycle_len: int = 12,
    ) -> None:
        if not snapshots:
            raise ValueError("SnapshotAnalyzer requires at least one snapshot")
        self._snapshots: List[SnapshotRecord] = sorted(list(snapshots), key=lambda rec: rec.step)
        self._step_index: Dict[int, int] = {rec.step: idx for idx, rec in enumerate(self._snapshots)}
        self._max_cycle_len = max_cycle_len
        self._domain = dict(domain or self._infer_domain(self._snapshots[0]))
        self._factor_costs: Dict[str, np.ndarray] = {}
        self._dag_bound_cache: Dict[int, int | None] = {}
        self._nilpotent_cache: Dict[int, int | None] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_factor_cost(self, factor: str, table: np.ndarray) -> None:
        """Register the cost table for a factor used in neutrality checks."""
        arr = np.asarray(table, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Factor cost tables must be square")
        self._factor_costs[str(factor)] = arr

    # ------------------------------------------------------------------
    # Belief reconstruction
    # ------------------------------------------------------------------
    def beliefs_per_variable(self) -> Dict[str, List[int | None]]:
        """Return the argmin trajectory for each variable in the snapshots."""
        variables = sorted(self._domain.keys())
        series: Dict[str, List[int | None]] = {var: [] for var in variables}
        for record in self._snapshots:
            grouped = self._group_r_messages(record.messages)
            for var in variables:
                vectors = grouped.get(var)
                if vectors:
                    combined = np.sum(vectors, axis=0)
                    series[var].append(int(np.argmin(combined)))
                else:
                    series[var].append(record.assignments.get(var))
        return series

    # ------------------------------------------------------------------
    # Difference coordinates
    # ------------------------------------------------------------------
    def difference_coordinates(
        self, step_idx: int
    ) -> tuple[Dict[tuple[str, str], float | np.ndarray], Dict[tuple[str, str], float | np.ndarray]]:
        """Compute ``ΔQ`` and ``ΔR`` for the requested step."""
        record = self._snapshot_by_index(step_idx)
        delta_q: Dict[tuple[str, str], float | np.ndarray] = {}
        delta_r: Dict[tuple[str, str], float | np.ndarray] = {}

        for message in record.messages:
            values = np.asarray(message.values, dtype=float)
            if values.size == 0:
                continue
            if message.flow == "variable_to_factor":
                key = (message.sender, message.recipient)
                delta_q[key] = self._recenter(values)
            else:
                key = (message.sender, message.recipient)
                delta_r[key] = self._recenter(values)
        return delta_q, delta_r

    # ------------------------------------------------------------------
    # Jacobian construction
    # ------------------------------------------------------------------
    def jacobian(self, step_idx: int) -> np.ndarray | sparse.spmatrix:
        """Construct the Jacobian matrix in difference coordinates for a step."""
        record = self._snapshot_by_index(step_idx)
        q_arrays, r_arrays, q_coords, r_coords = self._coordinate_arrays(step_idx)

        total_dim = len(q_coords) + len(r_coords)
        matrix: np.ndarray | sparse.lil_matrix
        if total_dim < 100:
            matrix = np.zeros((total_dim, total_dim), dtype=float)
        else:
            matrix = sparse.lil_matrix((total_dim, total_dim), dtype=float)

        q_index = {coord.key(): idx for idx, coord in enumerate(q_coords)}
        r_index = {coord.key(): len(q_coords) + idx for idx, coord in enumerate(r_coords)}

        # Variable rows
        for coord in q_coords:
            row = q_index[coord.key()]
            for r_coord in r_coords:
                if r_coord.recipient != coord.sender:
                    continue
                if r_coord.sender == coord.recipient:
                    continue
                if r_coord.label != coord.label:
                    continue
                col = r_index[r_coord.key()]
                _set_entry(matrix, row, col, 1.0)

        # Factor rows
        q_messages = self._index_messages(record.messages, flow="variable_to_factor")
        for coord in r_coords:
            row = r_index[coord.key()]
            incoming = [msg for msg in q_messages.values() if msg.recipient == coord.sender and msg.sender != coord.recipient]
            for msg in incoming:
                key = (msg.sender, msg.recipient)
                array = q_arrays.get(key)
                if array is None:
                    continue
                if array.size == 1:
                    delta = float(array[0])
                    value = 0.0 if abs(delta) < self._ZERO_TOL else -float(np.sign(delta))
                    col = q_index[('Q', key[0], key[1], 0)]
                    _set_entry(matrix, row, col, value)
                else:
                    winner = int(msg.argmin_index) if msg.argmin_index is not None else int(np.argmin(msg.values))
                    _, selector = multilabel_gaps(np.eye(array.size))
                    block = selector(winner)
                    for label in range(array.size):
                        col = q_index[('Q', key[0], key[1], label)]
                        _set_entry(matrix, row, col, float(block[coord.label, label]))

        return matrix

    # ------------------------------------------------------------------
    # Dependency digraph
    # ------------------------------------------------------------------
    def dependency_digraph(self, step_idx: int) -> nx.DiGraph:
        """Construct the dependency digraph induced by the Jacobian."""
        matrix = self.jacobian(step_idx)
        _, _, q_coords, r_coords = self._coordinate_arrays(step_idx)
        coord_list = q_coords + r_coords

        graph = nx.DiGraph()
        for idx, coord in enumerate(coord_list):
            graph.add_node(
                idx,
                kind=coord.kind,
                sender=coord.sender,
                recipient=coord.recipient,
                label=coord.label,
            )

        if sparse.issparse(matrix):
            rows, cols = matrix.nonzero()
            data = matrix.data
            for r_idx, c_idx, value in zip(rows, cols, data):
                graph.add_edge(c_idx, r_idx, weight=float(value))
        else:
            nz_rows, nz_cols = np.nonzero(matrix)
            for r_idx, c_idx in zip(nz_rows, nz_cols):
                graph.add_edge(c_idx, r_idx, weight=float(matrix[r_idx, c_idx]))
        return graph

    # ------------------------------------------------------------------
    # Neutrality checks
    # ------------------------------------------------------------------
    def neutral_step_test(
        self,
        step_idx: int,
        factor: str,
        from_var: str,
        to_var: str,
    ) -> tuple[bool, int | None]:
        """Check whether the factor step is neutral for the given edge."""
        record = self._snapshot_by_index(step_idx)
        factor_key = str(factor)
        if factor_key not in self._factor_costs:
            raise KeyError(f"No cost table registered for factor '{factor}'")

        cost = self._factor_costs[factor_key]
        q_messages = self._index_messages(record.messages, flow="variable_to_factor")
        key = (str(from_var), factor_key)
        if key not in q_messages:
            raise KeyError(f"No message {from_var}->{factor} in step {step_idx}")
        message = q_messages[key]
        values = np.asarray(message.values, dtype=float)

        if cost.shape == (2, 2) and values.size == 2:
            theta0, theta1 = binary_thresholds(cost)
            delta_q = float(values[1] - values[0])
            neutral, label = check_binary_neutral(delta_q, theta0, theta1)
            return neutral, label

        gaps, _ = multilabel_gaps(cost)
        winner = int(message.argmin_index) if message.argmin_index is not None else int(np.argmin(values))
        query = values - values[winner]
        cert = gaps[winner]
        neutral = bool(np.all(query >= cert - self._ZERO_TOL))
        return neutral, winner if neutral else None

    # ------------------------------------------------------------------
    # Future extensions (implemented in subsequent steps)
    # ------------------------------------------------------------------
    def scc_greedy_neutral_cover(self, step_idx: int, *, alpha: Mapping[str, float], kappa: float = 0.0, delta: float = 1e-3):
        graph = self.dependency_digraph(step_idx).copy()
        record = self._snapshot_by_index(step_idx)
        q_messages = self._index_messages(record.messages, flow="variable_to_factor")
        cover: List[Dict[str, object]] = []

        while True:
            components = [comp for comp in nx.strongly_connected_components(graph) if len(comp) > 1 or any(graph.has_edge(node, node) for node in comp)]
            if not components:
                break
            candidate_node = None
            candidate_data: Dict[str, object] | None = None
            for comp in components:
                for node in comp:
                    data = graph.nodes[node]
                    if data.get("kind") != "R":
                        continue
                    factor = str(data["sender"])
                    to_var = str(data["recipient"])
                    from_var = self._pick_neighbor_variable(q_messages, factor, to_var)
                    if from_var is None:
                        continue
                    try:
                        neutral, label = self.neutral_step_test(step_idx, factor, from_var, to_var)
                    except KeyError:
                        continue
                    if not neutral:
                        continue
                    slack = self._neutral_slack(step_idx, factor, from_var)
                    entry = {
                        "factor": factor,
                        "from_var": from_var,
                        "to_var": to_var,
                        "slack": slack,
                        "label": label,
                    }
                    if candidate_data is None or float(slack) > float(candidate_data["slack"]):
                        candidate_node = node
                        candidate_data = entry
            if candidate_node is None or candidate_data is None:
                break
            cover.append(candidate_data)
            graph.remove_node(candidate_node)

        return cover, graph

    def nilpotent_index(self, step_idx: int) -> int | None:
        if step_idx in self._nilpotent_cache:
            return self._nilpotent_cache[step_idx]

        matrix = self.jacobian(step_idx)
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
        if dense.size == 0:
            self._dag_bound_cache[step_idx] = 0
            self._nilpotent_cache[step_idx] = 0
            return 0

        dag_graph = self.dependency_digraph(step_idx)
        dag_bound = _longest_path_length(dag_graph)
        self._dag_bound_cache[step_idx] = dag_bound

        power = dense.copy()
        for idx in range(1, dense.shape[0] + 1):
            if np.allclose(power, 0.0, atol=1e-9):
                self._nilpotent_cache[step_idx] = idx
                return idx
            power = power @ dense

        self._nilpotent_cache[step_idx] = None
        return None

    def block_norms(self, step_idx: int) -> Dict[str, float]:
        matrix = self.jacobian(step_idx)
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
        q_arrays, r_arrays, _, _ = self._coordinate_arrays(step_idx)
        q_dim = sum(arr.size for arr in q_arrays.values())
        r_dim = sum(arr.size for arr in r_arrays.values())

        if q_dim + r_dim == 0:
            return {"A": 0.0, "B": 0.0, "P": 0.0}

        A_block = dense[:q_dim, q_dim: q_dim + r_dim]
        B_block = dense[q_dim: q_dim + r_dim, :q_dim]
        P_block = dense[q_dim: q_dim + r_dim, q_dim: q_dim + r_dim]

        def _inf_norm(block: np.ndarray) -> float:
            if block.size == 0:
                return 0.0
            return float(np.max(np.sum(np.abs(block), axis=1)))

        return {
            "A": _inf_norm(A_block),
            "B": _inf_norm(B_block),
            "P": _inf_norm(P_block),
        }

    def cycle_metrics(self, step_idx: int) -> Dict[str, object]:
        graph = self.dependency_digraph(step_idx)
        cycles: List[List[int]] = []
        aligned = 0
        record = self._snapshot_by_index(step_idx)
        q_messages = self._index_messages(record.messages, flow="variable_to_factor")

        for cycle in nx.simple_cycles(graph):
            if self._max_cycle_len and len(cycle) > self._max_cycle_len:
                continue
            cycles.append(cycle)
            aligned += sum(1 for node in cycle if graph.nodes[node].get("kind") == "R")

        has_neutral = False
        for cycle in cycles:
            for node in cycle:
                data = graph.nodes[node]
                if data.get("kind") != "R":
                    continue
                factor = str(data["sender"])
                to_var = str(data["recipient"])
                from_var = self._pick_neighbor_variable(q_messages, factor, to_var)
                if from_var is None:
                    continue
                try:
                    neutral, _ = self.neutral_step_test(step_idx, factor, from_var, to_var)
                except KeyError:
                    neutral = False
                if neutral:
                    has_neutral = True
                    break
            if has_neutral:
                break

        return {
            "num_cycles": len(cycles),
            "aligned_hops": aligned,
            "has_neutral": has_neutral,
        }

    def recommend_split_ratios(self, step_idx: int) -> Dict[str, float]:
        record = self._snapshot_by_index(step_idx)
        q_messages = self._index_messages(record.messages, flow="variable_to_factor")
        ratios: Dict[str, float] = {}
        for factor, table in self._factor_costs.items():
            slacks: List[float] = []
            for (var, target), _message in q_messages.items():
                if target != factor:
                    continue
                slack = self._neutral_slack(step_idx, factor, var)
                slacks.append(slack)
            if not slacks:
                continue
            worst = max(slacks)
            suggestion = 1.0 / (1.0 + worst) if worst > 0 else 1.0
            ratios[factor] = float(max(0.0, min(1.0, suggestion)))
        return ratios

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _snapshot_by_index(self, step_idx: int) -> SnapshotRecord:
        if step_idx < 0 or step_idx >= len(self._snapshots):
            raise IndexError("step_idx out of range")
        return self._snapshots[step_idx]

    @staticmethod
    def _group_r_messages(messages: Sequence[MessageRecord]) -> Dict[str, List[np.ndarray]]:
        grouped: Dict[str, List[np.ndarray]] = {}
        for message in messages:
            if message.flow != "factor_to_variable":
                continue
            grouped.setdefault(message.recipient, []).append(np.asarray(message.values, dtype=float))
        return grouped

    @staticmethod
    def _index_messages(messages: Sequence[MessageRecord], *, flow: str) -> Dict[tuple[str, str], MessageRecord]:
        indexed: Dict[tuple[str, str], MessageRecord] = {}
        for message in messages:
            if message.flow != flow:
                continue
            indexed[(message.sender, message.recipient)] = message
        return indexed

    @staticmethod
    def _recenter(values: np.ndarray) -> float | np.ndarray:
        if values.size == 0:
            return np.array([], dtype=float)
        if values.size == 2:
            return float(values[1] - values[0])
        offset = float(np.min(values))
        return values - offset

    @staticmethod
    def _as_array(value: float | np.ndarray) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.reshape(-1)
        return np.array([float(value)], dtype=float)

    @staticmethod
    def _infer_domain(record: SnapshotRecord) -> Dict[str, int]:
        domain: Dict[str, int] = {var: int(value) + 1 for var, value in record.assignments.items() if value is not None}
        for message in record.messages:
            values = np.asarray(message.values, dtype=float)
            if values.size:
                if message.flow == "variable_to_factor":
                    domain[message.sender] = max(domain.get(message.sender, 0), values.size)
                else:
                    domain[message.recipient] = max(domain.get(message.recipient, 0), values.size)
        return domain

    def _coordinate_arrays(
        self, step_idx: int
    ) -> tuple[
        Dict[tuple[str, str], np.ndarray],
        Dict[tuple[str, str], np.ndarray],
        List[_Coordinate],
        List[_Coordinate],
    ]:
        delta_q, delta_r = self.difference_coordinates(step_idx)
        q_arrays = {key: self._as_array(value) for key, value in delta_q.items()}
        r_arrays = {key: self._as_array(value) for key, value in delta_r.items()}

        q_coords: List[_Coordinate] = []
        for (var, factor), array in q_arrays.items():
            for label in range(array.size):
                q_coords.append(_Coordinate("Q", var, factor, label))

        r_coords: List[_Coordinate] = []
        for (factor, var), array in r_arrays.items():
            for label in range(array.size):
                r_coords.append(_Coordinate("R", factor, var, label))

        return q_arrays, r_arrays, q_coords, r_coords

    def _pick_neighbor_variable(
        self,
        q_messages: Mapping[tuple[str, str], MessageRecord],
        factor: str,
        excluded_var: str,
    ) -> str | None:
        for (var, target), _message in q_messages.items():
            if target == factor and var != excluded_var:
                return str(var)
        return None

    def _neutral_slack(self, step_idx: int, factor: str, from_var: str) -> float:
        record = self._snapshot_by_index(step_idx)
        q_messages = self._index_messages(record.messages, flow="variable_to_factor")
        message = q_messages.get((from_var, factor))
        if message is None:
            return 0.0
        cost = self._factor_costs.get(factor)
        if cost is None:
            return 0.0
        values = np.asarray(message.values, dtype=float)
        if cost.shape == (2, 2) and values.size == 2:
            theta0, theta1 = binary_thresholds(cost)
            delta = float(values[1] - values[0])
            if delta >= theta0:
                return float(delta - theta0)
            if delta <= -theta1:
                return float(-theta1 - delta)
            return float(min(theta0 - delta, delta + theta1))
        gaps, _ = multilabel_gaps(cost)
        winner = int(message.argmin_index) if message.argmin_index is not None else int(np.argmin(values))
        query = values - values[winner]
        slack_vec = query - gaps[winner]
        return float(np.min(slack_vec))


def _set_entry(matrix: np.ndarray | sparse.lil_matrix, row: int, col: int, value: float) -> None:
    if sparse.issparse(matrix):
        matrix[row, col] = value
    else:
        matrix[row, col] = value


def _longest_path_length(graph: nx.DiGraph) -> int | None:
    if not nx.is_directed_acyclic_graph(graph):
        return None
    distances: Dict[int, int] = {}
    for node in nx.topological_sort(graph):
        best = 0
        for predecessor in graph.predecessors(node):
            best = max(best, distances.get(predecessor, 0) + 1)
        distances[node] = max(distances.get(node, 0), best)
    return max(distances.values(), default=0)


class AnalysisReport:
    """Aggregate convenience helpers for turning analyzer outputs into artefacts."""

    def __init__(self, analyzer: SnapshotAnalyzer) -> None:
        self._analyzer = analyzer

    def to_json(
        self,
        step_idx: int,
        *,
        include_cover: bool = True,
        compute_spectral_radius: bool = True,
    ) -> Dict[str, object]:
        analyzer = self._analyzer
        beliefs = analyzer.beliefs_per_variable()
        cover, residual = analyzer.scc_greedy_neutral_cover(step_idx, alpha={}) if include_cover else ([], analyzer.dependency_digraph(step_idx))
        nilpotent = analyzer.nilpotent_index(step_idx)
        dag_bound = analyzer._dag_bound_cache.get(step_idx)
        block_norms = analyzer.block_norms(step_idx)
        cycles = analyzer.cycle_metrics(step_idx)
        ratios = analyzer.recommend_split_ratios(step_idx)

        spectral_radius = None
        if compute_spectral_radius:
            matrix = analyzer.jacobian(step_idx)
            dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(dense)))) if dense.size else 0.0

        return {
            "step": step_idx,
            "beliefs": beliefs,
            "nilpotent_index": nilpotent,
            "longest_path_bound": dag_bound,
            "block_norms": block_norms,
            "cycle_metrics": cycles,
            "neutral_cover": cover if include_cover else None,
            "recommended_alpha": ratios,
            "spectral_radius": spectral_radius,
            "residual_nodes": residual.number_of_nodes() if include_cover else None,
        }

    def to_csv(self, base_dir: str | Path, *, step_idx: int) -> None:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        beliefs = self._analyzer.beliefs_per_variable()
        steps = range(len(next(iter(beliefs.values()), [])))

        with (base / "beliefs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["step"] + list(beliefs.keys())
            writer.writerow(header)
            for step in steps:
                row = [step]
                for var in beliefs:
                    seq = beliefs[var]
                    row.append(seq[step] if step < len(seq) else None)
                writer.writerow(row)

        summary = self.to_json(step_idx, include_cover=True, compute_spectral_radius=True)
        with (base / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for key, value in summary.items():
                writer.writerow([key, value])

    def plots(self, base_dir: str | Path, *, step_idx: int, include_graph: bool = False) -> None:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        beliefs = self._analyzer.beliefs_per_variable()
        steps = range(len(next(iter(beliefs.values()), [])))
        plt.figure(figsize=(10, 5))
        for var, series in beliefs.items():
            plt.plot(steps, series, marker="o", label=var)
        plt.xlabel("Step")
        plt.ylabel("Argmin index")
        plt.title("Belief argmin trajectories")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(base / "beliefs.png", dpi=150)
        plt.close()

        if include_graph:
            graph = self._analyzer.dependency_digraph(step_idx)
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(graph, seed=42)
            nx.draw_networkx(graph, pos=pos, node_size=200, with_labels=False, arrows=True)
            plt.savefig(base / "dependency_graph.png", dpi=150)
            plt.close()


__all__ = ["SnapshotAnalyzer", "AnalysisReport"]
