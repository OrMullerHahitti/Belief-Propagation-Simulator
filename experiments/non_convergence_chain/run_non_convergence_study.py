"""CLI for the chain-focused non-convergence diagnostic package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from propflow import BPEngine, DampingEngine, MinSumComputator

from experiments.non_convergence_chain.config import (
    NonConvergenceConfig,
    build_chain_graph,
    build_random_graph,
    load_config,
    normalize_split_percentage,
)
from experiments.non_convergence_chain.long_damping import run_long_damping
from experiments.non_convergence_chain.midrun_split import run_midrun_split
from experiments.non_convergence_chain.oscillation_detector import (
    classification_details,
)
from experiments.non_convergence_chain.plotting import write_plots
from experiments.non_convergence_chain.reporting import write_outputs
from experiments.non_convergence_chain.route_analyzer import analyze_routes_from_graph
from experiments.non_convergence_chain.trace_recorder import trace_from_engine


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _apply_overrides(load_config(args.config, output_dir=args.out), args)
        run_chain = config.should_run_chain()
        run_random_graph = config.random_graph.enabled
        if not run_chain and not run_random_graph:
            raise ValueError(
                "No experiments are enabled. Set run_chain: true for the "
                "symmetric split X1-X2-X3 study, set random_graph.enabled: "
                "true for random-graph studies, or use --mode both."
            )
        if run_chain:
            config.require_cost_tables()
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    standard: dict[str, Any] | None = None

    if run_chain:
        chain_results = _run_chain_experiments(config)
        run_results.extend(chain_results)
        standard = chain_results[0]

    if config.random_graph.enabled:
        run_results.extend(_run_random_graph_experiments(config))

    route_analysis = _run_route_analysis(config, standard)
    write_outputs(
        run_results,
        output_dir=output_dir,
        route_analysis=route_analysis,
        tolerance=config.tolerance,
    )
    write_plots(run_results, output_dir)
    print(f"Diagnostics written to {output_dir}")
    return 0


def _run_chain_experiments(config: NonConvergenceConfig) -> list[dict[str, Any]]:
    graph_factory = lambda: build_chain_graph(config)
    results: list[dict[str, Any]] = []

    standard = _run_standard(
        graph_factory,
        max_iter=config.max_iter,
        trace_every=config.trace_every,
        tolerance=config.tolerance,
    )
    results.append(standard)

    for damping_factor in config.damping_factors:
        results.append(
            _run_damping(
                graph_factory,
                damping_factor=damping_factor,
                max_iter=config.max_iter,
                trace_every=config.trace_every,
                tolerance=config.tolerance,
            )
        )
        results.append(
            run_long_damping(
                graph_factory,
                damping_factor=damping_factor,
                max_iter=config.long_damping_max_iter,
                trace_every=config.trace_every,
                tolerance=config.tolerance,
            )
        )

    return results


def _run_route_analysis(
    config: NonConvergenceConfig, standard: dict[str, Any] | None
) -> dict[str, Any]:
    if standard is None:
        return {
            "status": "skipped",
            "warning": (
                "Chain experiments were disabled or skipped; exact route analysis "
                "is only run against the configured symmetric chain/cycle graph."
            ),
        }
    return analyze_routes_from_graph(
        build_chain_graph(config),
        simulation_classification=standard["summary"]["classification"],
        tolerance=config.tolerance,
    )


def _run_standard(
    graph_factory,
    *,
    max_iter: int,
    trace_every: int,
    tolerance: float,
    stop_on_convergence: bool = True,
) -> dict[str, Any]:
    engine = BPEngine(
        factor_graph=graph_factory(),
        computator=MinSumComputator(),
        normalize_messages=False,
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
        },
    }


def _run_damping(
    graph_factory,
    *,
    damping_factor: float,
    max_iter: int,
    trace_every: int,
    tolerance: float,
) -> dict[str, Any]:
    engine = DampingEngine(
        factor_graph=graph_factory(),
        computator=MinSumComputator(),
        damping_factor=damping_factor,
        normalize_messages=False,
    )
    engine.run(max_iter=max_iter)
    trace = trace_from_engine(engine, trace_every=trace_every, tolerance=tolerance)
    safe_lambda = str(damping_factor).replace(".", "_")
    return {
        "run_name": f"damping_lambda_{safe_lambda}",
        "engine": engine,
        "trace": trace,
        "summary": {
            **classification_details(trace, tolerance=tolerance),
            "damping_factor": damping_factor,
            "stop_on_convergence": True,
        },
    }


def _run_random_graph_experiments(config: NonConvergenceConfig) -> list[dict[str, Any]]:
    random_cfg = config.random_graph
    max_iter = int(random_cfg.max_iter or config.max_iter)
    split_iters = random_cfg.split_at_iters or config.split_at_iters
    percentage_split_at_iter = random_cfg.percentage_split_at_iter or split_iters[0]
    base_graph = build_random_graph(config)
    graph_metadata = _random_graph_metadata(config, base_graph)
    name_prefix = _random_graph_name_prefix(config)
    graph_factory = lambda: deepcopy(base_graph)
    results = [
        {
            **_run_standard(
                graph_factory,
                max_iter=max_iter,
                trace_every=config.trace_every,
                tolerance=config.tolerance,
                stop_on_convergence=False,
            ),
            "run_name": f"{name_prefix}_standard",
        }
    ]
    _attach_random_metadata(results[-1], graph_metadata)
    if random_cfg.run_split_at_sweep:
        for split_at in split_iters:
            for mode in random_cfg.split_transfer_modes:
                result = _run_random_split(
                    graph_factory,
                    split_at=split_at,
                    split_fraction=1.0,
                    mode=mode,
                    max_iter=max_iter,
                    config=config,
                )
                result["run_name"] = f"{name_prefix}_split_at_{split_at}_all_{mode}"
                result["summary"]["random_experiment_family"] = "split_at_sweep"
                _attach_random_metadata(result, graph_metadata)
                results.append(result)

    if random_cfg.run_percentage_sweep:
        for percentage in random_cfg.split_percentages:
            for mode in random_cfg.split_transfer_modes:
                result = _run_random_split(
                    graph_factory,
                    split_at=percentage_split_at_iter,
                    split_fraction=percentage,
                    mode=mode,
                    max_iter=max_iter,
                    config=config,
                )
                label = _fraction_label(percentage)
                result["run_name"] = (
                    f"{name_prefix}_split_pct_{label}_at_"
                    f"{percentage_split_at_iter}_{mode}"
                )
                result["summary"]["random_experiment_family"] = "percentage_sweep"
                _attach_random_metadata(result, graph_metadata)
                results.append(result)

    if random_cfg.run_combined_sweep:
        for split_at in split_iters:
            for percentage in random_cfg.split_percentages:
                for mode in random_cfg.split_transfer_modes:
                    result = _run_random_split(
                        graph_factory,
                        split_at=split_at,
                        split_fraction=percentage,
                        mode=mode,
                        max_iter=max_iter,
                        config=config,
                    )
                    label = _fraction_label(percentage)
                    result["run_name"] = (
                        f"{name_prefix}_split_grid_at_{split_at}_pct_{label}_{mode}"
                    )
                    result["summary"]["random_experiment_family"] = "combined_grid"
                    _attach_random_metadata(result, graph_metadata)
                    results.append(result)
    return results


def _random_graph_name_prefix(config: NonConvergenceConfig) -> str:
    random_cfg = config.random_graph
    density = _safe_float_label(random_cfg.density)
    return f"random_graph_vars_{random_cfg.num_vars}_density_{density}"


def _safe_float_label(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def _attach_random_metadata(result: dict[str, Any], metadata: dict[str, Any]) -> None:
    result["artifact_stem"] = f"seed_{metadata['random_seed']}"
    result.setdefault("summary", {})
    result["summary"].update(metadata)
    result["summary"]["artifact_stem"] = result["artifact_stem"]


def _random_graph_metadata(config: NonConvergenceConfig, graph) -> dict[str, Any]:
    random_cfg = config.random_graph
    return {
        "random_seed": config.seed,
        "random_num_variable_nodes": random_cfg.num_vars,
        "random_domain_size": random_cfg.domain_size,
        "random_density": random_cfg.density,
        "random_num_factor_nodes": len(graph.factors),
        "random_graph_fingerprint": _graph_fingerprint(graph),
    }


def _graph_fingerprint(graph) -> str:
    rows: list[dict[str, Any]] = []
    for factor in sorted(graph.factors, key=lambda item: item.name):
        variables = [variable.name for variable in graph.edges[factor]]
        table = np.asarray(factor.cost_table, dtype=float).tolist()
        rows.append(
            {
                "factor": factor.name,
                "variables": variables,
                "cost_table": table,
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_random_split(
    graph_factory,
    *,
    split_at: int,
    split_fraction: float,
    mode: str,
    max_iter: int,
    config: NonConvergenceConfig,
) -> dict[str, Any]:
    result = run_midrun_split(
        graph_factory,
        split_at_iter=split_at,
        max_iter=max_iter,
        split_ratio=config.split_ratio,
        split_targets=config.split_targets,
        split_fraction=split_fraction,
        split_seed=config.seed,
        transfer_mode=mode,
        trace_every=config.trace_every,
        tolerance=config.tolerance,
        stop_on_convergence=False,
    )
    result["summary"]["split_fraction"] = split_fraction
    result["summary"]["split_percentage"] = split_fraction * 100.0
    return result


def _fraction_label(value: float) -> str:
    percentage = value * 100.0
    if abs(percentage - round(percentage)) < 1e-9:
        return str(int(round(percentage)))
    return str(percentage).replace(".", "_")


def _run_full_iterations(engine: BPEngine, max_iter: int) -> int | None:
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


def _apply_overrides(
    config: NonConvergenceConfig, args: argparse.Namespace
) -> NonConvergenceConfig:
    updates: dict[str, Any] = {}
    random_graph = config.random_graph
    if args.mode == "chain":
        updates["run_chain"] = True
        random_graph = replace(random_graph, enabled=False)
    elif args.mode == "random-graph":
        updates["run_chain"] = False
        random_graph = replace(random_graph, enabled=True)
    elif args.mode == "both":
        updates["run_chain"] = True
        random_graph = replace(random_graph, enabled=True)
    if args.max_iter is not None:
        updates["max_iter"] = args.max_iter
    if args.split_at is not None:
        updates["split_at_iters"] = args.split_at
        random_graph = replace(random_graph, split_at_iters=args.split_at)
        if args.percentage_split_at is None and args.split_at:
            random_graph = replace(
                random_graph, percentage_split_at_iter=args.split_at[0]
            )
    if args.split_percentages is not None:
        random_graph = replace(
            random_graph,
            split_percentages=[
                normalize_split_percentage(value) for value in args.split_percentages
            ],
        )
    if args.percentage_split_at is not None:
        random_graph = replace(
            random_graph, percentage_split_at_iter=args.percentage_split_at
        )
    if args.damping is not None:
        updates["damping_factors"] = args.damping
    if args.long_damping_max_iter is not None:
        updates["long_damping_max_iter"] = args.long_damping_max_iter
    if args.trace_every is not None:
        updates["trace_every"] = max(1, args.trace_every)
    if random_graph is not config.random_graph:
        updates["random_graph"] = random_graph
    return replace(config, **updates) if updates else config


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument(
        "--mode",
        choices=("chain", "random-graph", "both"),
        help=(
            "Override enabled experiment families. chain requires F12/F23 tables; "
            "random-graph does not."
        ),
    )
    parser.add_argument("--out", help="Output directory.")
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--split-at", nargs="*", type=int)
    parser.add_argument("--split-percentages", nargs="*", type=float)
    parser.add_argument("--percentage-split-at", type=int)
    parser.add_argument("--damping", nargs="*", type=float)
    parser.add_argument("--long-damping-max-iter", type=int)
    parser.add_argument("--trace-every", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
