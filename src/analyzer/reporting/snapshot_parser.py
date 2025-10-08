"""Snapshot parsing and adaptation utilities for analyzer/reporting.

This module bridges the JSON snapshots captured by :class:`EngineSnapshotRecorder`
with the richer typed representation consumed by the new reporting pipeline. It
also provides adapters for the in-engine :class:`propflow.snapshots.manager.SnapshotManager`.

The exported dataclasses form the canonical representation expected by the
``SnapshotAnalyzer`` and downstream tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

Tolerance = 1e-9
AbsTolerance = 1e-12


@dataclass(slots=True)
class MessageRecord:
    """Typed representation of a single message inside a snapshot."""

    flow: str
    sender: str
    recipient: str
    values: List[float]
    argmin_index: int | None
    neutral: bool


@dataclass(slots=True)
class SnapshotRecord:
    """Typed representation of a single engine step snapshot."""

    step: int
    messages: List[MessageRecord]
    assignments: Dict[str, int]
    cost: float | None
    neutral_messages: int
    step_neutral: bool


def load_snapshots(path: str | Path) -> List[Dict[str, Any]]:
    """Load snapshots from the JSON format emitted by ``EngineSnapshotRecorder``."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Snapshot JSON must contain a list of step dictionaries")
    return [dict(entry) for entry in data]


def parse_snapshots(obj: Sequence[Mapping[str, Any]]) -> List[SnapshotRecord]:
    """Parse a raw snapshot sequence into strongly typed ``SnapshotRecord`` instances."""
    if not isinstance(obj, Sequence):
        raise TypeError("Snapshots must be provided as a sequence of mappings")

    records: List[SnapshotRecord] = []
    expected_step: int | None = None
    for index, raw in enumerate(obj):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Snapshot entry {index} is not a mapping")

        step = _coerce_int(raw.get("step"), "step")
        if expected_step is None:
            expected_step = step
        elif step != expected_step:
            raise ValueError("Snapshot steps must be contiguous and ordered")

        messages = _parse_messages(raw.get("messages"), step)
        assignments = _parse_assignments(raw.get("assignments"))
        cost_value = raw.get("cost")
        cost = float(cost_value) if cost_value is not None else None

        declared_neutral = raw.get("neutral_messages")
        computed_neutral = sum(1 for msg in messages if msg.neutral)
        if declared_neutral is None:
            neutral_messages = computed_neutral
        else:
            neutral_messages = _coerce_int(declared_neutral, "neutral_messages")
            if neutral_messages != computed_neutral:
                raise ValueError(
                    f"Neutral message count mismatch at step {step}: "
                    f"declared {neutral_messages}, computed {computed_neutral}"
                )

        step_neutral_value = raw.get("step_neutral")
        step_neutral = bool(step_neutral_value) if step_neutral_value is not None else bool(messages) and neutral_messages == len(messages)

        records.append(
            SnapshotRecord(
                step=step,
                messages=messages,
                assignments=assignments,
                cost=cost,
                neutral_messages=neutral_messages,
                step_neutral=step_neutral,
            )
        )
        expected_step += 1
    return records


def from_engine_snapshot_manager(engine_record: "propflow.snapshots.types.SnapshotRecord") -> SnapshotRecord:
    """Coerce an in-engine snapshot manager record into a ``SnapshotRecord``."""
    data = engine_record.data
    min_idx = engine_record.min_idx or {}

    messages: List[MessageRecord] = []

    for (var, factor), values in sorted(data.Q.items()):
        float_values = _to_float_list(values)
        argmin_key = (var, factor)
        argmin_index = int(min_idx.get(argmin_key, _argmin(float_values))) if float_values else None
        neutral = _is_neutral(float_values, argmin_index)
        messages.append(
            MessageRecord(
                flow="variable_to_factor",
                sender=str(var),
                recipient=str(factor),
                values=float_values,
                argmin_index=argmin_index,
                neutral=neutral,
            )
        )

    for (factor, var), values in sorted(data.R.items()):
        float_values = _to_float_list(values)
        argmin_index = _argmin(float_values)
        neutral = _is_neutral(float_values, argmin_index)
        messages.append(
            MessageRecord(
                flow="factor_to_variable",
                sender=str(factor),
                recipient=str(var),
                values=float_values,
                argmin_index=argmin_index,
                neutral=neutral,
            )
        )

    assignments = _compute_assignments_from_r(data.R, data.dom)
    neutral_messages = sum(1 for msg in messages if msg.neutral)
    step_neutral = bool(messages) and neutral_messages == len(messages)

    step = int(getattr(data, "step", 0))
    cost = None

    return SnapshotRecord(
        step=step,
        messages=messages,
        assignments=assignments,
        cost=cost,
        neutral_messages=neutral_messages,
        step_neutral=step_neutral,
    )


def _parse_messages(payload: Any, step: int) -> List[MessageRecord]:
    if payload is None:
        return []
    if not isinstance(payload, Sequence):
        raise TypeError(f"Snapshot step {step} 'messages' entry must be a list")

    result: List[MessageRecord] = []
    for idx, message in enumerate(payload):
        if not isinstance(message, Mapping):
            raise TypeError(f"Message at index {idx} in step {step} is not a mapping")

        flow = str(message.get("flow", ""))
        if flow not in {"variable_to_factor", "factor_to_variable"}:
            raise ValueError(f"Unknown message flow '{flow}' at step {step}")

        sender = str(message.get("sender", ""))
        recipient = str(message.get("recipient", ""))

        values = _to_float_list(message.get("values", []))
        argmin_idx_raw = message.get("argmin_index")
        argmin_index = int(argmin_idx_raw) if argmin_idx_raw is not None else _argmin(values)
        neutral = bool(message.get("neutral")) if message.get("neutral") is not None else _is_neutral(values, argmin_index)

        result.append(
            MessageRecord(
                flow=flow,
                sender=sender,
                recipient=recipient,
                values=values,
                argmin_index=argmin_index,
                neutral=neutral,
            )
        )
    return result


def _parse_assignments(payload: Any) -> Dict[str, int]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("Snapshot assignments must be a mapping")
    assignments: Dict[str, int] = {}
    for key, value in payload.items():
        assignments[str(key)] = _coerce_int(value, f"assignments[{key}]")
    return assignments


def _coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise TypeError(f"Field '{field_name}' cannot be None")
    if isinstance(value, bool):
        raise TypeError(f"Field '{field_name}' must be an integer, not bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Field '{field_name}' must be coercible to int") from exc


def _to_float_list(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [float(v) for v in values.tolist()]
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return [float(v) for v in values]
    return [float(values)]


def _argmin(values: List[float]) -> int | None:
    if not values:
        return None
    min_value = min(values)
    for idx, value in enumerate(values):
        if isclose(value, min_value, rel_tol=Tolerance, abs_tol=AbsTolerance):
            return idx
    return None


def _is_neutral(values: List[float], argmin_index: int | None) -> bool:
    if not values or argmin_index is None:
        return False
    min_value = values[argmin_index]
    matches = sum(1 for value in values if isclose(value, min_value, rel_tol=Tolerance, abs_tol=AbsTolerance))
    return matches > 1


def _compute_assignments_from_r(
    r_messages: Mapping[tuple[str, str], np.ndarray],
    domains: Mapping[str, Sequence[str]],
) -> Dict[str, int]:
    assignments: Dict[str, int] = {}
    for var, labels in domains.items():
        label_count = len(labels)
        accum = np.zeros(label_count, dtype=float)
        has_message = False
        for (factor, target), values in r_messages.items():
            if target != var:
                continue
            float_values = np.asarray(values, dtype=float)
            if float_values.size != label_count:
                raise ValueError(
                    f"R message {factor}->{target} has length {float_values.size}, expected {label_count}"
                )
            accum += float_values
            has_message = True
        if has_message:
            assignments[var] = int(np.argmin(accum))
    return assignments


__all__ = [
    "MessageRecord",
    "SnapshotRecord",
    "load_snapshots",
    "parse_snapshots",
    "from_engine_snapshot_manager",
]
