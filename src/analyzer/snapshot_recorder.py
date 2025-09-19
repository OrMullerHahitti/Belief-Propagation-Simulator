"""External recorder for capturing per-step snapshots from belief propagation engines."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass(slots=True)
class MessageSnapshot:
    """Serializable record of a single message exchanged during an iteration."""
    

    flow: str # either "variable_to_factor" or "factor_to_variable"
    sender: str # typically the name of the sending agent
    recipient: str # typically the name of the receiving agent
    values: List[float] # the numeric contents of the message
    argmin_index: int | None # index of the minimum value in `values`, or None if empty
    neutral: bool # whether the message is neutral (multiple minima)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the snapshot to a JSON-serialisable dictionary."""
        return {
            "flow": self.flow,
            "sender": self.sender,
            "recipient": self.recipient,
            "values": self.values,
            "argmin_index": self.argmin_index,
            "neutral": self.neutral,
        }


class EngineSnapshotRecorder:
    """Capture step-by-step snapshots from a belief propagation engine.

    The recorder wraps an engine instance and drives the iteration loop externally,
    collecting detailed message traffic together with assignment and cost summaries.
    Snapshots are kept in memory and can be exported to JSON for post-run analysis.
    """

    def __init__(self, engine: Any):
        self._engine = engine
        self.snapshots: List[Dict[str, Any]] = []

    def record_run(
        self,
        max_steps: int,
        *,
        reset: bool = True,
        break_on_convergence: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute the engine step-by-step and capture per-iteration data.

        Args:
            max_steps: Maximum number of iterations to record. Must be positive.
            reset: Whether to clear any previously recorded data before running.
            break_on_convergence: Stop early if the engine reports convergence via
                a callable ``_is_converged`` attribute. Defaults to ``False``.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if reset:
            self.snapshots = []

        for step_index in range(max_steps):
            step_result = self._engine.step(step_index)
            snapshot = self._capture_step(step_index, step_result)
            self.snapshots.append(snapshot)

            if break_on_convergence and self._engine_reports_converged():
                break

        return self.snapshots

    # ---------------------------- persistence ----------------------------
    def to_json(self, filepath: str | Path, *, indent: int = 2) -> None:
        """Write the recorded snapshots to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(self.snapshots, json_file, indent=indent)

    # ---------------------------- internals -----------------------------
    def _capture_step(self, step_index: int, step_result: Any) -> Dict[str, Any]:
        messages = []
        messages.extend(self._collect_messages(step_result, "q"))
        messages.extend(self._collect_messages(step_result, "r"))

        assignments = self._resolve_assignments()
        cost = self._resolve_cost()

        neutral_messages = sum(1 for msg in messages if msg.neutral)
        step_neutral = bool(messages) and neutral_messages == len(messages)

        return {
            "step": step_index,
            "messages": [msg.to_dict() for msg in messages],
            "assignments": assignments,
            "cost": cost,
            "neutral_messages": neutral_messages,
            "step_neutral": step_neutral,
        }

    def _collect_messages(self, step_result: Any, flow: str) -> List[MessageSnapshot]:
        storage_name = "q_messages" if flow == "q" else "r_messages"
        snapshots: List[MessageSnapshot] = []
        message_map = getattr(step_result, storage_name, {}) or {}

        for _, message_list in message_map.items():
            for message in message_list or []:
                sender_name = self._safe_name(getattr(message, "sender", None))
                recipient_name = self._safe_name(getattr(message, "recipient", None))
                values = self._to_float_list(getattr(message, "data", []))
                argmin_index = self._argmin(values)
                neutral = self._is_neutral(values, argmin_index)
                snapshots.append(
                    MessageSnapshot(
                        flow="variable_to_factor" if flow == "q" else "factor_to_variable",
                        sender=sender_name,
                        recipient=recipient_name,
                        values=values,
                        argmin_index=argmin_index,
                        neutral=neutral,
                    )
                )
        return snapshots

    def _resolve_assignments(self) -> Dict[str, Any]:
        assignments = getattr(self._engine, "assignments", {})
        if callable(assignments):  # guard against property returning callable
            assignments = assignments()

        sanitized: Dict[str, Any] = {}
        for key, value in (assignments or {}).items():
            sanitized[str(key)] = self._coerce_numeric(value)
        return sanitized

    def _resolve_cost(self) -> float | None:
        calc = getattr(self._engine, "calculate_global_cost", None)
        if callable(calc):
            try:
                value = calc()
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None
        return None

    def _engine_reports_converged(self) -> bool:
        convergence_check = getattr(self._engine, "_is_converged", None)
        if callable(convergence_check):
            try:
                return bool(convergence_check())
            except Exception:
                return False
        return False

    @staticmethod
    def _safe_name(agent: Any) -> str:
        if agent is None:
            return "unknown"
        name = getattr(agent, "name", None)
        if name is not None:
            return str(name)
        return str(agent)

    @staticmethod
    def _to_float_list(values: Any) -> List[float]:
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            flat = values.astype(float).flatten()
            return flat.tolist()
        if isinstance(values, (list, tuple)):
            return [float(v) for v in values]
        return [float(values)]

    @staticmethod
    def _argmin(values: Iterable[float]) -> int | None:
        values_list = list(values)
        if not values_list:
            return None
        min_value = min(values_list)
        for idx, value in enumerate(values_list):
            if value == min_value:
                return idx
        return None

    @staticmethod
    def _is_neutral(values: Iterable[float], argmin_index: int | None) -> bool:
        values_list = list(values)
        if not values_list or argmin_index is None:
            return False
        min_value = values_list[argmin_index]
        matches = sum(
            1
            for value in values_list
            if math.isclose(value, min_value, rel_tol=1e-9, abs_tol=1e-12)
        )
        return matches > 1

    @staticmethod
    def _coerce_numeric(value: Any) -> Any:
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        return value


__all__ = ["EngineSnapshotRecorder", "MessageSnapshot"]
