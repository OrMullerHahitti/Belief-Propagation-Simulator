"""Long-run damping stability checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from propflow import DampingEngine, FactorGraph, MinSumComputator

from .diagonal_analyzer import analyze_trace_diagonals, summarize_diagonal_diagnostics
from .oscillation_detector import classification_details
from .trace_recorder import trace_from_engine


def run_long_damping(
    graph_factory: Callable[[], FactorGraph],
    *,
    damping_factor: float,
    max_iter: int,
    trace_every: int = 1,
    full_trace_until: int = 200,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Run damped Min-Sum for many iterations with sampled trace retention."""

    engine = DampingEngine(
        factor_graph=graph_factory(),
        computator=MinSumComputator(),
        damping_factor=damping_factor,
        normalize_messages=False,
    )
    first_convergence_step = _run_full_iterations(engine, max_iter)
    trace = trace_from_engine(
        engine,
        trace_every=trace_every,
        full_until=full_trace_until,
        tolerance=tolerance,
    )
    details = classification_details(trace, tolerance=tolerance)
    diagonal_trace = analyze_trace_diagonals(trace)
    diagonal_summary = summarize_diagonal_diagnostics(
        diagonal_trace, details.get("tail_start")
    )
    safe_lambda = str(damping_factor).replace(".", "_")
    return {
        "run_name": f"long_damping_lambda_{safe_lambda}",
        "engine": engine,
        "trace": trace,
        "summary": {
            **details,
            "damping_factor": damping_factor,
            "max_iter": max_iter,
            "stop_on_convergence": False,
            "first_forced_convergence_step": first_convergence_step,
            "oscillation_returned_after_apparent_convergence": _oscillation_returned(
                details
            ),
            "measured_mechanism": _measured_mechanism(trace, diagonal_summary),
            "diagonal_summary": diagonal_summary,
        },
    }


def _run_full_iterations(engine: DampingEngine, max_iter: int) -> int | None:
    """Run exactly ``max_iter`` steps while recording convergence events."""

    first_convergence_step: int | None = None
    engine.convergence_monitor.reset()
    for i in range(max_iter):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            if first_convergence_step is None:
                first_convergence_step = i + 1
            continue
    return first_convergence_step


def _oscillation_returned(details: dict[str, Any]) -> bool:
    classification = details.get("classification")
    return classification in {
        "period_2_oscillation",
        "period_k_oscillation",
        "transient_then_oscillation",
    }


def _measured_mechanism(
    trace: list[dict[str, Any]], diagonal_summary: dict[str, Any]
) -> str:
    if not trace:
        return "no trace retained"
    late = trace[-min(20, len(trace)) :]
    max_delta = 0.0
    for row in late:
        for bucket in ("belief_deltas", "Q_deltas", "R_deltas"):
            values = row.get(bucket, {}).values()
            if values:
                max_delta = max(max_delta, max(abs(float(v)) for v in values))
    if max_delta < 1e-7:
        return "deltas_shrink_to_zero_or_near_zero"
    if not diagonal_summary.get("opposite_pairs_seen", False):
        return "selected_diagonal_orientation_stabilizes_or_no_opposition_seen"
    return "late_trace_retains_nonzero_deltas"
