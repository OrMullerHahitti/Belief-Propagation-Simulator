"""Convergence and oscillation diagnostics for BP traces."""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_assignment_convergence(
    trace: list[dict[str, Any]], window: int = 5, tolerance: float = 0.0
) -> bool:
    """Return True if assignments are stable over the final window."""

    del tolerance
    assignments = [row.get("assignments") for row in trace if row.get("assignments")]
    if len(assignments) < max(2, window):
        return False
    tail = assignments[-window:]
    return all(item == tail[0] for item in tail[1:])


def detect_belief_convergence(
    trace: list[dict[str, Any]], window: int = 5, tolerance: float = 1e-9
) -> bool:
    """Return True if belief vectors change by at most tolerance in the final window."""

    beliefs = [row.get("beliefs") for row in trace if row.get("beliefs")]
    if len(beliefs) < max(2, window):
        return False
    tail = beliefs[-window:]
    for prev, curr in zip(tail, tail[1:]):
        keys = set(prev) | set(curr)
        for key in keys:
            if key not in prev or key not in curr:
                return False
            if (
                np.linalg.norm(
                    np.asarray(curr[key], dtype=float)
                    - np.asarray(prev[key], dtype=float)
                )
                > tolerance
            ):
                return False
    return True


def detect_period(
    trace: list[dict[str, Any]],
    max_period: int = 20,
    min_repeats: int = 3,
    tolerance: float = 1e-9,
) -> dict[str, Any] | None:
    """Detect a stable periodic suffix in assignments or beliefs."""

    if not trace:
        return None
    states = [_state_payload(row) for row in trace]
    n = len(states)
    upper = min(max_period, max(1, n // max(1, min_repeats)))
    for period in range(1, upper + 1):
        needed = period * min_repeats
        if n < needed:
            continue
        pattern = states[n - period : n]
        stable = True
        for repeat in range(2, min_repeats + 1):
            left = n - repeat * period
            right = n - (repeat - 1) * period
            if not _states_equal(states[left:right], pattern, tolerance):
                stable = False
                break
        if not stable:
            continue
        for start in range(0, n - needed + 1):
            if _periodic_from(states, start, pattern, tolerance):
                return {
                    "period": period,
                    "start": start,
                    "pattern": pattern,
                }
    return None


def detect_even_odd_oscillation(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect stable period-2 behavior separated by parity."""

    info = detect_period(trace, max_period=2, min_repeats=3)
    is_period_2 = bool(info and info["period"] == 2)
    immediate = bool(is_period_2 and info and info["start"] <= 1)
    return {
        "is_even_odd": is_period_2,
        "starts_immediately": immediate,
        "period_start": None if info is None else info["start"],
    }


def estimate_tail_start_or_t0(trace: list[dict[str, Any]]) -> int | None:
    """Estimate the first iteration of the final periodic regime."""

    info = detect_period(trace)
    if info is None:
        return None
    return int(info["start"])


def classify_run(trace: list[dict[str, Any]]) -> str:
    """Classify a trace into convergence/oscillation categories."""

    if detect_belief_convergence(trace) or detect_assignment_convergence(trace):
        return "converged"

    period_info = detect_period(trace)
    if period_info is None:
        return "no_clear_classification"

    period = int(period_info["period"])
    start = int(period_info["start"])
    if start > 1:
        return "transient_then_oscillation"
    if period == 2:
        return "period_2_oscillation"
    return "period_k_oscillation"


def classification_details(
    trace: list[dict[str, Any]], tolerance: float = 1e-9
) -> dict[str, Any]:
    """Return classifier output plus supporting diagnostics."""

    period = detect_period(trace, tolerance=tolerance)
    even_odd = detect_even_odd_oscillation(trace)
    return {
        "classification": classify_run(trace),
        "period": None if period is None else period["period"],
        "tail_start": None if period is None else period["start"],
        "assignment_converged": detect_assignment_convergence(
            trace, tolerance=tolerance
        ),
        "belief_converged": detect_belief_convergence(trace, tolerance=tolerance),
        "even_odd_oscillation": even_odd["is_even_odd"],
        "immediate_even_odd_oscillation": even_odd["starts_immediately"],
    }


def _state_payload(row: dict[str, Any]) -> Any:
    if row.get("assignments"):
        return tuple(
            sorted((key, int(value)) for key, value in row["assignments"].items())
        )
    beliefs = row.get("beliefs", {})
    return tuple(
        (key, tuple(float(x) for x in np.asarray(value, dtype=float).reshape(-1)))
        for key, value in sorted(beliefs.items())
    )


def _states_equal(left: list[Any], right: list[Any], tolerance: float) -> bool:
    if len(left) != len(right):
        return False
    return all(_state_equal(a, b, tolerance) for a, b in zip(left, right))


def _state_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        return all(_state_equal(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def _periodic_from(
    states: list[Any], start: int, pattern: list[Any], tolerance: float
) -> bool:
    period = len(pattern)
    base = len(states) - period
    for idx in range(start, len(states)):
        expected = pattern[(idx - base) % period]
        if not _state_equal(states[idx], expected, tolerance):
            return False
    return True
