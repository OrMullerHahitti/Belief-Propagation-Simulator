"""CLI for the minimal symmetric-chain non-convergence study."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from propflow import BPEngine, FactorGraph, MinSumComputator

from experiments.non_convergence_chain.config import (
    NonConvergenceConfig,
    build_chain_graph,
    load_config,
)
from experiments.non_convergence_chain.oscillation_detector import (
    classification_details,
)
from experiments.non_convergence_chain.reporting import write_outputs
from experiments.non_convergence_chain.trace_recorder import trace_from_engine


def main(argv: list[str] | None = None) -> int:
    """Run the standard Min-Sum trace/classification slice."""

    args = _parse_args(argv)
    try:
        config = _apply_overrides(load_config(args.config, output_dir=args.out), args)
        _validate_standard_chain_config(config)
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_standard_chain(
        config,
        stop_on_convergence=args.stop_on_convergence,
        normalize_messages=args.normalize_messages,
    )
    route_analysis = {
        "status": "skipped",
        "warning": (
            "Route, damping, and split diagnostics are outside this minimal "
            "standard Min-Sum reproducibility slice."
        ),
    }
    write_outputs(
        [result],
        output_dir=output_dir,
        route_analysis=route_analysis,
        tolerance=config.tolerance,
    )
    print(f"Standard chain diagnostics written to {output_dir}")
    return 0


def run_standard_chain(
    config: NonConvergenceConfig,
    *,
    stop_on_convergence: bool = False,
    normalize_messages: bool = True,
) -> dict[str, Any]:
    """Build the configured symmetric chain and run standard Min-Sum."""

    return _run_standard(
        lambda: build_chain_graph(config),
        max_iter=config.max_iter,
        trace_every=config.trace_every,
        tolerance=config.tolerance,
        stop_on_convergence=stop_on_convergence,
        normalize_messages=normalize_messages,
    )


def _run_standard(
    graph_factory: Callable[[], FactorGraph],
    *,
    max_iter: int,
    trace_every: int,
    tolerance: float,
    stop_on_convergence: bool = False,
    normalize_messages: bool = True,
) -> dict[str, Any]:
    engine = BPEngine(
        factor_graph=graph_factory(),
        computator=MinSumComputator(),
        normalize_messages=normalize_messages,
    )
    first_forced_convergence_step = None
    if stop_on_convergence:
        engine.run(max_iter=max_iter)
    else:
        first_forced_convergence_step = _run_full_iterations(engine, max_iter)
    trace = trace_from_engine(engine, trace_every=trace_every, tolerance=tolerance)
    return {
        "run_name": "standard",
        "engine": engine,
        "trace": trace,
        "summary": {
            **classification_details(trace, tolerance=tolerance),
            "stop_on_convergence": stop_on_convergence,
            "first_forced_convergence_step": first_forced_convergence_step,
            "normalize_messages": normalize_messages,
        },
    }


def _run_full_iterations(engine: BPEngine, max_iter: int) -> int | None:
    """Run exactly ``max_iter`` steps while recording monitor stop points."""

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


def _validate_standard_chain_config(config: NonConvergenceConfig) -> None:
    if config.run_chain is False:
        raise ValueError(
            "This minimal runner only supports the symmetric chain study. "
            "Set run_chain: true and provide F12/F23 cost tables."
        )
    if config.random_graph.enabled:
        raise ValueError(
            "Random-graph studies are outside this minimal slice. "
            "Use a chain config with random_graph.enabled: false."
        )
    config.require_cost_tables()


def _apply_overrides(
    config: NonConvergenceConfig, args: argparse.Namespace
) -> NonConvergenceConfig:
    updates: dict[str, Any] = {}
    if args.max_iter is not None:
        updates["max_iter"] = args.max_iter
    if args.trace_every is not None:
        updates["trace_every"] = max(1, args.trace_every)
    if args.tolerance is not None:
        updates["tolerance"] = args.tolerance
    return replace(config, **updates) if updates else config


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--out", help="Output directory.")
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--trace-every", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument(
        "--stop-on-convergence",
        action="store_true",
        help=(
            "Allow the engine convergence monitor to stop early. By default the "
            "runner records the full max-iter trace for oscillation diagnosis."
        ),
    )
    parser.add_argument(
        "--no-normalize-messages",
        action="store_false",
        dest="normalize_messages",
        help="Disable PropFlow message normalization during the standard run.",
    )
    parser.set_defaults(normalize_messages=True)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
