"""Experiment helpers for mid-run factor splitting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from propflow import FactorGraph, MidRunSplitEngine, MinSumComputator

from .oscillation_detector import classification_details
from .trace_recorder import trace_from_engine


def run_midrun_split(
    graph_factory: Callable[[], FactorGraph],
    *,
    split_at_iter: int,
    max_iter: int,
    split_ratio: float = 0.5,
    split_targets: list[str] | None = None,
    split_fraction: float | None = None,
    split_seed: int | None = None,
    transfer_mode: str = "reset",
    trace_every: int = 1,
    tolerance: float = 1e-9,
    stop_on_convergence: bool = True,
) -> dict[str, Any]:
    """Run BP with a split injection before ``split_at_iter``."""

    engine = MidRunSplitEngine(
        factor_graph=graph_factory(),
        computator=MinSumComputator(),
        split_at_iter=split_at_iter,
        split_factor=split_ratio,
        split_targets=split_targets,
        split_fraction=split_fraction,
        split_seed=split_seed,
        transfer_mode=transfer_mode,  # type: ignore[arg-type]
        normalize_messages=False,
    )
    first_forced_convergence_step = None
    if stop_on_convergence:
        engine.run(max_iter=max_iter)
    else:
        first_forced_convergence_step = _run_full_iterations(engine, max_iter)
    trace = trace_from_engine(
        engine,
        trace_every=trace_every,
        tolerance=tolerance,
        split_events=engine.split_events,
    )
    details = classification_details(trace, tolerance=tolerance)
    return {
        "run_name": f"midrun_split_{split_at_iter}_{transfer_mode}",
        "engine": engine,
        "trace": trace,
        "summary": {
            **details,
            "split_at_iter": int(split_at_iter),
            "transfer_mode": transfer_mode,
            "split_mapping": [
                event.get("split_mapping", {}) for event in engine.split_events
            ],
            "stop_on_convergence": stop_on_convergence,
            "first_forced_convergence_step": first_forced_convergence_step,
        },
    }


def _run_full_iterations(engine: MidRunSplitEngine, max_iter: int) -> int | None:
    """Run exactly ``max_iter`` steps while still updating convergence history."""

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
