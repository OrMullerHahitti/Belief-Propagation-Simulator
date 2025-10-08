"""Core analysis utilities for parsed snapshot records."""
from __future__ import annotations

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
        self._max_cycle_len = int(max_cycle_len)
        self._domain = dict(domain or self._infer_domain(self._snapshots[0]))
        self._factor_costs: Dict[str, np.ndarray] = {}

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
        delta_q, delta_r = self.difference_coordinates(step_idx)

        q_arrays = {key: self._as_array(value) for key, value in delta_q.items()}
        r_arrays = {key: self._as_array(value) for key, value in delta_r.items()}

        q_coords: List[_Coordinate] = []
        r_coords: List[_Coordinate] = []

        for (var, factor), array in q_arrays.items():
            for label in range(array.size):
                q_coords.append(_Coordinate("Q", var, factor, label))

        for (factor, var), array in r_arrays.items():
            for label in range(array.size):
                r_coords.append(_Coordinate("R", factor, var, label))

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
        delta_q, delta_r = self.difference_coordinates(step_idx)

        q_arrays = {key: self._as_array(value) for key, value in delta_q.items()}
        r_arrays = {key: self._as_array(value) for key, value in delta_r.items()}
        q_coords = [_Coordinate("Q", var, factor, label) for (var, factor), arr in q_arrays.items() for label in range(arr.size)]
        r_coords = [_Coordinate("R", factor, var, label) for (factor, var), arr in r_arrays.items() for label in range(arr.size)]
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
        raise NotImplementedError

    def nilpotent_index(self, step_idx: int) -> int | None:
        raise NotImplementedError

    def block_norms(self, step_idx: int) -> Dict[str, float]:
        raise NotImplementedError

    def cycle_metrics(self, step_idx: int) -> Dict[str, object]:
        raise NotImplementedError

    def recommend_split_ratios(self, step_idx: int) -> Dict[str, float]:
        raise NotImplementedError

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


def _set_entry(matrix: np.ndarray | sparse.lil_matrix, row: int, col: int, value: float) -> None:
    if sparse.issparse(matrix):
        matrix[row, col] = value
    else:
        matrix[row, col] = value


__all__ = ["SnapshotAnalyzer"]
