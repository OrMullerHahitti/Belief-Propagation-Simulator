"""Trace extraction for BP snapshots used by the chain diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def message_series(
    messages: dict[tuple[str, str], np.ndarray],
) -> dict[str, list[float]]:
    """Convert snapshot message mapping to stable ``sender->recipient`` keys."""

    result: dict[str, list[float]] = {}
    for (sender, recipient), values in sorted(messages.items()):
        result[f"{sender}->{recipient}"] = np.asarray(values, dtype=float).tolist()
    return result


def binary_deltas(series: dict[str, list[float]]) -> dict[str, float]:
    """Return value-1 minus value-0 for binary message/belief vectors."""

    deltas: dict[str, float] = {}
    for key, values in series.items():
        if len(values) == 2:
            deltas[key] = float(values[1] - values[0])
    return deltas


def selected_minimizers_from_snapshot(
    snapshot: Any, tolerance: float = 1e-9
) -> dict[str, dict[str, Any]]:
    """Reconstruct selected Min-Sum factor minimizers from snapshot state.

    The current engine snapshots already contain cost tables, factor labels, and
    Q messages. For Min-Sum this is enough to reconstruct the minimizing cost
    table entries selected by every factor-to-variable update.
    """

    selected: dict[str, dict[str, Any]] = {}
    for factor_name, table_like in sorted(snapshot.cost_tables.items()):
        labels = list(snapshot.cost_labels.get(factor_name, []))
        if not labels:
            continue
        table = np.asarray(table_like, dtype=float)
        aggregate = table.copy()
        broadcasts: list[np.ndarray] = []
        q_vectors: dict[str, np.ndarray] = {}
        for axis, variable_name in enumerate(labels):
            q_values = snapshot.Q.get((variable_name, factor_name))
            if q_values is None:
                vector = np.zeros(table.shape[axis], dtype=float)
            else:
                vector = np.asarray(q_values, dtype=float).reshape(-1)
            if vector.size != table.shape[axis]:
                vector = np.resize(vector, table.shape[axis])
            broadcast = vector.reshape(
                [table.shape[i] if i == axis else 1 for i in range(table.ndim)]
            )
            q_vectors[variable_name] = vector
            broadcasts.append(broadcast)
            aggregate = aggregate + broadcast

        factor_result: dict[str, Any] = {}
        for axis, variable_name in enumerate(labels):
            reduced = aggregate - broadcasts[axis]
            by_value: dict[str, list[list[int]]] = {}
            all_entries: list[list[int]] = []
            for value_index in range(table.shape[axis]):
                view = np.take(reduced, indices=value_index, axis=axis)
                min_value = float(np.min(view))
                indices = np.argwhere(np.isclose(view, min_value, atol=tolerance))
                entries: list[list[int]] = []
                for index_tuple in indices:
                    full: list[int] = []
                    cursor = 0
                    for dim in range(table.ndim):
                        if dim == axis:
                            full.append(int(value_index))
                        else:
                            full.append(int(index_tuple[cursor]))
                            cursor += 1
                    entries.append(full)
                by_value[str(value_index)] = entries
                all_entries.extend(entries)
            factor_result[variable_name] = {
                "selected_entries_by_value": by_value,
                "selected_entries": all_entries,
                "row_minimizer_map": (
                    _row_minimizer_map(table) if table.ndim == 2 else {}
                ),
                "column_minimizer_map": (
                    _column_minimizer_map(table) if table.ndim == 2 else {}
                ),
            }
        selected[factor_name] = factor_result
    return selected


def _row_minimizer_map(table: np.ndarray) -> dict[str, list[int]]:
    return {
        str(row): np.argwhere(np.isclose(table[row], np.min(table[row])))
        .reshape(-1)
        .astype(int)
        .tolist()
        for row in range(table.shape[0])
    }


def _column_minimizer_map(table: np.ndarray) -> dict[str, list[int]]:
    return {
        str(col): np.argwhere(np.isclose(table[:, col], np.min(table[:, col])))
        .reshape(-1)
        .astype(int)
        .tolist()
        for col in range(table.shape[1])
    }


def trace_from_engine(
    engine: Any,
    *,
    trace_every: int = 1,
    full_until: int | None = None,
    tolerance: float = 1e-9,
    split_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract a JSON-serializable trace from an engine's snapshots."""

    trace_every = max(1, int(trace_every))
    split_by_iteration = {
        int(event["iteration"]): event for event in split_events or []
    }
    rows: list[dict[str, Any]] = []
    for snapshot in engine.snapshots:
        if full_until is None:
            keep = snapshot.step % trace_every == 0
        else:
            keep = snapshot.step <= full_until or snapshot.step % trace_every == 0
        if not keep:
            continue
        beliefs = {
            name: np.asarray(values, dtype=float).tolist()
            for name, values in sorted(snapshot.beliefs.items())
        }
        q_messages = message_series(snapshot.Q)
        r_messages = message_series(snapshot.R)
        row = {
            "iteration": int(snapshot.step),
            "parity": "even" if snapshot.step % 2 == 0 else "odd",
            "assignments": {k: int(v) for k, v in sorted(snapshot.assignments.items())},
            "global_cost": (
                None if snapshot.global_cost is None else float(snapshot.global_cost)
            ),
            "beliefs": beliefs,
            "belief_deltas": binary_deltas(beliefs),
            "Q": q_messages,
            "R": r_messages,
            "Q_deltas": binary_deltas(q_messages),
            "R_deltas": binary_deltas(r_messages),
            "selected_minimizers": selected_minimizers_from_snapshot(
                snapshot, tolerance
            ),
            "split_event": split_by_iteration.get(int(snapshot.step)),
        }
        rows.append(row)
    return rows


def write_jsonl(trace: list[dict[str, Any]], path: str | Path) -> Path:
    """Write trace rows as deterministic JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in trace:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    return target
