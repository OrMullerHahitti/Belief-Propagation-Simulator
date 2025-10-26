"""Snapshot analysis utilities for belief propagation convergence dynamics.

This module provides analysis tools for examining convergence properties,
dependency structures, and cycle metrics from simulation snapshots.
"""
from __future__ import annotations

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Any

import networkx as nx
import numpy as np
from scipy import sparse

from .types import SnapshotRecord


# Tolerance constants
_ZERO_TOL = 1e-9


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
    """Analyze convergence dynamics from propflow snapshots.

    This class provides methods to derive Jacobian matrices, dependency graphs,
    cycle metrics, and other structural properties from snapshot sequences.
    """

    def __init__(
        self,
        snapshots: Sequence[SnapshotRecord],
        *,
        domain: Mapping[str, int] | None = None,
        max_cycle_len: int = 12,
    ) -> None:
        """Initialize the analyzer with a sequence of snapshots.

        Args:
            snapshots: A sequence of SnapshotRecord objects from a simulation.
            domain: Optional domain mapping (var -> domain_size). If not provided,
                will be inferred from snapshots.
            max_cycle_len: Maximum cycle length to enumerate in cycle analysis.
        """
        if not snapshots:
            raise ValueError("SnapshotAnalyzer requires at least one snapshot")
        self._snapshots: List[SnapshotRecord] = sorted(list(snapshots), key=lambda rec: rec.data.step)
        self._step_index: Dict[int, int] = {rec.data.step: idx for idx, rec in enumerate(self._snapshots)}
        self._max_cycle_len = max_cycle_len
        self._domain = dict(domain or self._infer_domain(self._snapshots[0]))
        self._dag_bound_cache: Dict[int, int | None] = {}
        self._nilpotent_cache: Dict[int, int | None] = {}

    def beliefs_per_variable(self) -> Dict[str, List[int | None]]:
        """Return the argmin trajectory for each variable across snapshots."""
        variables = sorted(self._domain.keys())
        series: Dict[str, List[int | None]] = {var: [] for var in variables}

        for rec in self._snapshots:
            data = rec.data
            # Sum R messages per variable to compute belief
            grouped: Dict[str, List[np.ndarray]] = {}
            for (f, v), r_msg in data.R.items():
                grouped.setdefault(v, []).append(np.asarray(r_msg, dtype=float))

            for var in variables:
                vectors = grouped.get(var)
                if vectors:
                    combined = np.sum(vectors, axis=0)
                    series[var].append(int(np.argmin(combined)))
                else:
                    series[var].append(data.assignments.get(var))

        return series

    def difference_coordinates(
        self, step_idx: int
    ) -> tuple[Dict[tuple[str, str], float | np.ndarray], Dict[tuple[str, str], float | np.ndarray]]:
        """Compute ΔQ and ΔR (difference coordinates) for a snapshot."""
        rec = self._snapshot_by_index(step_idx)
        data = rec.data

        delta_q: Dict[tuple[str, str], float | np.ndarray] = {}
        delta_r: Dict[tuple[str, str], float | np.ndarray] = {}

        for (u, f), q_msg in data.Q.items():
            values = np.asarray(q_msg, dtype=float)
            if values.size > 0:
                delta_q[(u, f)] = self._recenter(values)

        for (f, v), r_msg in data.R.items():
            values = np.asarray(r_msg, dtype=float)
            if values.size > 0:
                delta_r[(f, v)] = self._recenter(values)

        return delta_q, delta_r

    def jacobian(self, step_idx: int) -> np.ndarray | sparse.spmatrix:
        """Construct the Jacobian matrix in difference coordinates."""
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
        rec = self._snapshot_by_index(step_idx)
        data = rec.data
        q_messages = {(u, f): q_arr for (u, f), q_arr in data.Q.items()}

        for coord in r_coords:
            row = r_index[coord.key()]
            factor, var = coord.sender, coord.recipient

            # Incoming messages from other variables to this factor
            for (u, f), q_arr in q_messages.items():
                if f != factor or u == var:
                    continue
                array = q_arrays.get((u, f))
                if array is None:
                    continue
                if array.size == 1:
                    delta = float(array[0])
                    value = 0.0 if abs(delta) < _ZERO_TOL else -float(np.sign(delta))
                    col = q_index[('Q', u, f, 0)]
                    _set_entry(matrix, row, col, value)
                else:
                    winner = int(np.argmin(q_arr))
                    # Simplified: use identity projection
                    for label in range(array.size):
                        col = q_index[('Q', u, f, label)]
                        _set_entry(matrix, row, col, 1.0 if label == winner else 0.0)

        return matrix

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
            # Convert to COO format for safe iteration
            coo_matrix = matrix.tocoo()
            for r_idx, c_idx, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                graph.add_edge(int(c_idx), int(r_idx), weight=float(value))
        else:
            nz_rows, nz_cols = np.nonzero(matrix)
            for r_idx, c_idx in zip(nz_rows, nz_cols):
                graph.add_edge(int(c_idx), int(r_idx), weight=float(matrix[r_idx, c_idx]))

        return graph

    def cycle_metrics(self, step_idx: int) -> Dict[str, Any]:
        """Compute cycle metrics for convergence analysis."""
        graph = self.dependency_digraph(step_idx)
        cycles: List[List[int]] = []

        for cycle in nx.simple_cycles(graph):
            if self._max_cycle_len and len(cycle) > self._max_cycle_len:
                continue
            cycles.append(cycle)

        return {
            "num_cycles": len(cycles),
            "has_cycles": len(cycles) > 0,
        }

    def nilpotent_index(self, step_idx: int) -> int | None:
        """Compute the nilpotent index of the Jacobian matrix."""
        if step_idx in self._nilpotent_cache:
            return self._nilpotent_cache[step_idx]

        matrix = self.jacobian(step_idx)
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)

        if dense.size == 0:
            self._nilpotent_cache[step_idx] = 0
            return 0

        power = dense.copy()
        for idx in range(1, dense.shape[0] + 1):
            if np.allclose(power, 0.0, atol=1e-9):
                self._nilpotent_cache[step_idx] = idx
                return idx
            power = power @ dense

        self._nilpotent_cache[step_idx] = None
        return None

    def block_norms(self, step_idx: int) -> Dict[str, float]:
        """Compute infinity norms of Jacobian blocks."""
        matrix = self.jacobian(step_idx)
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
        q_arrays, r_arrays, _, _ = self._coordinate_arrays(step_idx)

        q_dim = sum(arr.size for arr in q_arrays.values())
        r_dim = sum(arr.size for arr in r_arrays.values())

        if q_dim + r_dim == 0:
            return {"A": 0.0, "B": 0.0, "P": 0.0}

        def _inf_norm(block: np.ndarray) -> float:
            if block.size == 0:
                return 0.0
            return float(np.max(np.sum(np.abs(block), axis=1)))

        A_block = dense[:q_dim, q_dim: q_dim + r_dim]
        B_block = dense[q_dim: q_dim + r_dim, :q_dim]
        P_block = dense[q_dim: q_dim + r_dim, q_dim: q_dim + r_dim]

        return {
            "A": _inf_norm(A_block),
            "B": _inf_norm(B_block),
            "P": _inf_norm(P_block),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_by_index(self, step_idx: int) -> SnapshotRecord:
        """Retrieve snapshot by index in the sorted list."""
        if step_idx < 0 or step_idx >= len(self._snapshots):
            raise IndexError("step_idx out of range")
        return self._snapshots[step_idx]

    @staticmethod
    def _recenter(values: np.ndarray) -> float | np.ndarray:
        """Recenter array to minimum or compute scalar difference."""
        if values.size == 0:
            return np.array([], dtype=float)
        if values.size == 2:
            return float(values[1] - values[0])
        offset = float(np.min(values))
        return values - offset

    @staticmethod
    def _as_array(value: float | np.ndarray) -> np.ndarray:
        """Convert scalar or array to numpy array."""
        if isinstance(value, np.ndarray):
            return value.reshape(-1)
        return np.array([float(value)], dtype=float)

    @staticmethod
    def _infer_domain(record: SnapshotRecord) -> Dict[str, int]:
        """Infer variable domains from snapshot data."""
        data = record.data
        domain: Dict[str, int] = {}

        # From assignments
        for var, val in data.assignments.items():
            if val is not None:
                domain[var] = max(domain.get(var, 0), val + 1)

        # From Q messages
        for (u, f), q_arr in data.Q.items():
            values = np.asarray(q_arr, dtype=float)
            if values.size:
                domain[u] = max(domain.get(u, 0), values.size)

        # From R messages
        for (f, v), r_arr in data.R.items():
            values = np.asarray(r_arr, dtype=float)
            if values.size:
                domain[v] = max(domain.get(v, 0), values.size)

        return domain

    def _coordinate_arrays(
        self, step_idx: int
    ) -> tuple[
        Dict[tuple[str, str], np.ndarray],
        Dict[tuple[str, str], np.ndarray],
        List[_Coordinate],
        List[_Coordinate],
    ]:
        """Build coordinate arrays and coordinate lists."""
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


class AnalysisReport:
    """Generate analysis reports from snapshots."""

    def __init__(self, analyzer: SnapshotAnalyzer) -> None:
        """Initialize with a snapshot analyzer.

        Args:
            analyzer: The SnapshotAnalyzer instance to generate reports from.
        """
        self._analyzer = analyzer

    def to_json(self, step_idx: int) -> Dict[str, Any]:
        """Generate a JSON-serializable analysis report for a snapshot.

        Args:
            step_idx: The index of the snapshot to analyze.

        Returns:
            A dictionary containing analysis results.
        """
        analyzer = self._analyzer
        beliefs = analyzer.beliefs_per_variable()
        nilpotent = analyzer.nilpotent_index(step_idx)
        block_norms = analyzer.block_norms(step_idx)
        cycles = analyzer.cycle_metrics(step_idx)

        # Spectral radius (if dense enough)
        spectral_radius = None
        try:
            matrix = analyzer.jacobian(step_idx)
            dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
            if dense.size:
                spectral_radius = float(np.max(np.abs(np.linalg.eigvals(dense))))
        except Exception:
            pass

        return {
            "step": step_idx,
            "beliefs": beliefs,
            "nilpotent_index": nilpotent,
            "block_norms": block_norms,
            "cycle_metrics": cycles,
            "spectral_radius": spectral_radius,
        }

    def to_csv(self, base_dir: str | Path, *, step_idx: int) -> None:
        """Export analysis results to CSV files.

        Args:
            base_dir: Directory to save CSV files.
            step_idx: The step index for the analysis.
        """
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

        summary = self.to_json(step_idx)
        with (base / "metrics.json").open("w", encoding="utf-8") as handle:
            import json
            json.dump(summary, handle, indent=2)


def _set_entry(matrix: np.ndarray | sparse.lil_matrix, row: int, col: int, value: float) -> None:
    """Set matrix entry, handling both dense and sparse matrices."""
    matrix[row, col] = value


__all__ = ["SnapshotAnalyzer", "AnalysisReport"]
