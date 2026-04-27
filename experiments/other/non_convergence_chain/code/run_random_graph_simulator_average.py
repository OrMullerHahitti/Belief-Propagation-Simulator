"""Simulator-based averaged random-graph splitting plots.

This runner uses :class:`propflow.Simulator` to average global-cost trajectories
over multiple random graph instances. It is intentionally cost-trace focused:
snapshots are compact and keep only the fields needed by ``Simulator``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from propflow import DampingEngine, FGBuilder, MidRunSplitEngine, MinSumComputator
from propflow.policies.damping import damp
from propflow.simulator import Simulator


def default_output_dir(args: argparse.Namespace) -> Path:
    density = str(args.density).replace(".", "_")
    damping = str(args.damping_factor).replace(".", "_")
    seed_end = args.seed_start + args.graph_count - 1
    return Path(
        "results/"
        f"non_convergence_random_graph_vars_{args.num_vars}_"
        f"domain_{args.domain_size}_density_{density}_"
        f"seeds_{args.seed_start}_{seed_end}_iter_{args.max_iter}_"
        f"damping_{damping}_simulator_averages"
    )


class CostOnlySnapshot:
    """Minimal snapshot object compatible with ``Simulator`` cost extraction."""

    def __init__(self, step: int) -> None:
        self.step = int(step)
        self.global_cost: float | None = None
        self.metadata: dict[str, Any] = {}


class CostOnlySnapshotManager:
    """Capture only per-step global cost to keep simulator workers small."""

    def capture_step(self, step_index: int, step: Any, engine: Any) -> CostOnlySnapshot:
        del step, engine
        return CostOnlySnapshot(step_index)


class FullRunMixin:
    """Run all requested iterations while preserving update-rule semantics."""

    def run(
        self,
        max_iter: int | None = None,
        save_json: bool = False,
        save_csv: bool = False,
        filename: str | None = None,
        config_name: str | None = None,
    ) -> None:
        del save_json, save_csv, filename, config_name
        if max_iter is None:
            raise ValueError("This experiment requires an explicit max_iter.")

        self.convergence_monitor.reset()
        for i in range(int(max_iter)):
            self.step(i)
            try:
                self._handle_cycle_events(i)
            except StopIteration:
                # Match the previous long-run sweeps: record convergence but
                # keep iterating so every curve has the full requested horizon.
                continue
        return None


class FullRunDampingEngine(FullRunMixin, DampingEngine):
    """DampingEngine variant that continues through ``max_iter``."""


class FullRunDampedMidRunSplitEngine(FullRunMixin, MidRunSplitEngine):
    """Mid-run splitting with the same Q-message damping hook as DampingEngine."""

    def __init__(
        self, *args: Any, damping_factor: float = 0.9, **kwargs: Any
    ) -> None:
        self.damping_factor = float(damping_factor)
        super().__init__(*args, **kwargs)
        self._name = "FullRunDampedMidRunSplitEngine"
        self._set_name({"damping": str(self.damping_factor)})

    def post_var_compute(self, var: Any) -> None:
        damp(var, self.damping_factor)
        var.append_last_iteration()


def graph_fingerprint(graph: Any) -> str:
    rows = []
    for factor in sorted(graph.factors, key=lambda item: item.name):
        rows.append(
            (
                factor.name,
                [variable.name for variable in graph.edges[factor]],
                np.asarray(factor.cost_table).astype(float).tolist(),
            )
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_graphs(
    *,
    graph_count: int,
    seed_start: int,
    num_vars: int,
    domain_size: int,
    density: float,
    ct_low: int,
    ct_high: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    graphs: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + graph_count):
        np.random.seed(seed)
        graph = FGBuilder.build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory="random_int",
            ct_params={"low": ct_low, "high": ct_high},
            density=density,
            seed=seed,
        )
        graphs.append(graph)
        metadata.append(
            {
                "seed": seed,
                "graph_fingerprint": graph_fingerprint(graph),
                "variable_count": len(graph.variables),
                "factor_count": len(graph.factors),
                "edge_count": sum(len(vs) for vs in graph.edges.values()),
            }
        )
    return graphs, metadata


def base_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "computator": MinSumComputator(),
        "normalize_messages": True,
        "snapshot_manager": CostOnlySnapshotManager(),
    }


def percentage_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        "baseline_no_split": {
            "class": FullRunDampingEngine,
            "damping_factor": args.damping_factor,
            **base_engine_kwargs(args),
        }
    }
    for pct in range(10, 101, 10):
        configs[f"split_pct_{pct}_at_0_transfer"] = {
            "class": FullRunDampedMidRunSplitEngine,
            "damping_factor": args.damping_factor,
            "split_at_iter": 0,
            "split_factor": args.split_ratio,
            "split_fraction": pct / 100,
            "split_seed": args.seed_start,
            "transfer_mode": "transfer",
            **base_engine_kwargs(args),
        }
    return configs


def split_at_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        "baseline_no_split": {
            "class": FullRunDampingEngine,
            "damping_factor": args.damping_factor,
            **base_engine_kwargs(args),
        },
        "split_pct_100_at_0_transfer": {
            "class": FullRunDampedMidRunSplitEngine,
            "damping_factor": args.damping_factor,
            "split_at_iter": 0,
            "split_factor": args.split_ratio,
            "transfer_mode": "transfer",
            **base_engine_kwargs(args),
        },
    }
    for split_at in args.split_at_iters:
        configs[f"split_all_at_{split_at}_transfer"] = {
            "class": FullRunDampedMidRunSplitEngine,
            "damping_factor": args.damping_factor,
            "split_at_iter": split_at,
            "split_factor": args.split_ratio,
            "transfer_mode": "transfer",
            **base_engine_kwargs(args),
        }
    return configs


def run_simulator_batch(
    *,
    batch_name: str,
    configs: dict[str, dict[str, Any]],
    graphs: list[Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, list[list[float]]]:
    print(
        f"START {batch_name}: {len(configs)} engines x {len(graphs)} graphs",
        flush=True,
    )
    simulator = Simulator(configs, seed=args.seed_start)
    simulator.timeout = args.timeout_seconds
    results = simulator.run_simulations(graphs, max_iter=args.max_iter)
    write_raw_costs(output_dir / batch_name / "raw_costs.csv", results)
    write_average_costs(output_dir / batch_name / "average_costs.csv", results)
    print(f"DONE {batch_name}", flush=True)
    return results


def pad_costs(costs_list: list[list[float]], max_iter: int) -> np.ndarray:
    rows = []
    for costs in costs_list:
        if not costs:
            rows.append([np.nan] * max_iter)
            continue
        trimmed = [float(value) for value in costs[:max_iter]]
        if len(trimmed) < max_iter:
            trimmed.extend([trimmed[-1]] * (max_iter - len(trimmed)))
        rows.append(trimmed)
    return np.asarray(rows, dtype=float)


def write_raw_costs(path: Path, results: dict[str, list[list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine", "graph_index", "iteration", "global_cost"])
        for engine_name, costs_list in results.items():
            for graph_index, costs in enumerate(costs_list):
                for iteration, cost in enumerate(costs):
                    writer.writerow([engine_name, graph_index, iteration, cost])


def write_average_costs(path: Path, results: dict[str, list[list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_iter = max((len(costs) for costs_list in results.values() for costs in costs_list), default=0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine", "iteration", "mean_global_cost", "std_global_cost", "run_count"])
        for engine_name, costs_list in results.items():
            padded = pad_costs(costs_list, max_iter)
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            for iteration, (mean_value, std_value) in enumerate(zip(mean, std)):
                writer.writerow(
                    [
                        engine_name,
                        iteration,
                        float(mean_value),
                        float(std_value),
                        len(costs_list),
                    ]
                )


def percentage_label(engine_name: str) -> str:
    if engine_name == "baseline_no_split":
        return "Baseline (no split)"
    pct = int(engine_name.split("_")[2])
    return f"{pct}% factor split"


def split_at_label(engine_name: str) -> str:
    if engine_name == "baseline_no_split":
        return "Baseline (no split)"
    if engine_name == "split_pct_100_at_0_transfer":
        return "Splitting at iteration 0"
    split_at = int(engine_name.split("_")[3])
    return f"Splitting at iteration {split_at}"


def plot_average(
    *,
    path: Path,
    results: dict[str, list[list[float]]],
    order: list[str],
    title: str,
    labeler: Any,
    max_iter: int,
) -> None:
    plt.figure(figsize=(14, 8))
    for engine_name in order:
        padded = pad_costs(results[engine_name], max_iter)
        mean = np.nanmean(padded, axis=0)
        plt.plot(
            np.arange(max_iter),
            mean,
            linewidth=1.7,
            label=labeler(engine_name),
        )
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Average global cost")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_graph_metadata(path: Path, graph_metadata: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "graph_fingerprint",
                "variable_count",
                "factor_count",
                "edge_count",
            ],
        )
        writer.writeheader()
        writer.writerows(graph_metadata)


def write_summary_md(output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Simulator Averaged Random-Graph Splitting Sweep",
        "",
        "This run uses `propflow.Simulator` to average global-cost trajectories over multiple random graph examples.",
        "",
        "## Settings",
        f"- Graph seeds: `{args.seed_start}` through `{args.seed_start + args.graph_count - 1}`",
        f"- Graph examples per plot family: `{args.graph_count}`",
        f"- Variables: `{args.num_vars}`",
        f"- Domain size: `{args.domain_size}`",
        f"- Density: `{args.density}`",
        f"- Max iterations: `{args.max_iter}`",
        f"- Normalized messages: `True`",
        f"- Damping factor: `{args.damping_factor}`",
        f"- Split ratio: `{args.split_ratio}`",
        f"- Transfer mode: `transfer`",
        "",
        "## Outputs",
        "- `percentage_sweep/average_costs.csv` and `percentage_sweep/raw_costs.csv`",
        "- `split_at_sweep/average_costs.csv` and `split_at_sweep/raw_costs.csv`",
        f"- `global_cost_percentage_sweep_average_{args.graph_count}_graphs.png`",
        f"- `global_cost_split_at_sweep_average_{args.graph_count}_graphs.png`",
        "- `graph_metadata.csv` and `settings.json`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--graph-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-vars", type=int, default=100)
    parser.add_argument("--domain-size", type=int, default=10)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--damping-factor", type=float, default=0.9)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--ct-low", type=int, default=0)
    parser.add_argument("--ct-high", type=int, default=10)
    parser.add_argument(
        "--split-at-iters",
        type=int,
        nargs="+",
        default=[20, 40, 80, 120, 160],
    )
    parser.add_argument("--timeout-seconds", type=int, default=28_800)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out if args.out is not None else default_output_dir(args)
    if output_dir.exists() and not args.keep_existing:
        backup_idx = 1
        while True:
            backup = output_dir.with_name(f"{output_dir.name}.previous_{backup_idx}")
            if not backup.exists():
                shutil.move(str(output_dir), str(backup))
                break
            backup_idx += 1
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "graph_count": args.graph_count,
        "seed_start": args.seed_start,
        "num_vars": args.num_vars,
        "domain_size": args.domain_size,
        "density": args.density,
        "max_iter": args.max_iter,
        "damping_factor": args.damping_factor,
        "split_ratio": args.split_ratio,
        "ct_params": {"low": args.ct_low, "high": args.ct_high},
        "split_at_iters": args.split_at_iters,
        "simulator_module": "propflow.simulator.Simulator",
        "snapshot_mode": "cost_only",
        "full_run_after_convergence": True,
    }
    write_json(output_dir / "settings.json", settings)

    graphs, graph_metadata = build_graphs(
        graph_count=args.graph_count,
        seed_start=args.seed_start,
        num_vars=args.num_vars,
        domain_size=args.domain_size,
        density=args.density,
        ct_low=args.ct_low,
        ct_high=args.ct_high,
    )
    write_json(output_dir / "graph_metadata.json", graph_metadata)
    write_graph_metadata(output_dir / "graph_metadata.csv", graph_metadata)

    percentage_order = ["baseline_no_split"] + [
        f"split_pct_{pct}_at_0_transfer" for pct in range(10, 101, 10)
    ]
    split_at_order = [
        "baseline_no_split",
        "split_pct_100_at_0_transfer",
        *[f"split_all_at_{split_at}_transfer" for split_at in args.split_at_iters],
    ]

    percentage_results = run_simulator_batch(
        batch_name="percentage_sweep",
        configs=percentage_configs(args),
        graphs=graphs,
        args=args,
        output_dir=output_dir,
    )
    plot_average(
        path=output_dir
        / f"global_cost_percentage_sweep_average_{args.graph_count}_graphs.png",
        results=percentage_results,
        order=percentage_order,
        title="Average Global Cost: Factor Split Percentage Sweep",
        labeler=percentage_label,
        max_iter=args.max_iter,
    )

    split_at_results = run_simulator_batch(
        batch_name="split_at_sweep",
        configs=split_at_configs(args),
        graphs=graphs,
        args=args,
        output_dir=output_dir,
    )
    plot_average(
        path=output_dir
        / f"global_cost_split_at_sweep_average_{args.graph_count}_graphs.png",
        results=split_at_results,
        order=split_at_order,
        title="Average Global Cost: Splitting Iteration Sweep",
        labeler=split_at_label,
        max_iter=args.max_iter,
    )

    write_summary_md(output_dir, args)
    print(f"OUTPUT_DIR={output_dir}", flush=True)


if __name__ == "__main__":
    main()
