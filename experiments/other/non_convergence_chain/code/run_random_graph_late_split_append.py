"""Append late split-at-iteration runs to the simulator averaged sweep."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .run_random_graph_simulator_average import (
    FullRunDampedMidRunSplitEngine,
    base_engine_kwargs,
    build_graphs,
    graph_fingerprint,
    run_simulator_batch,
    split_at_label,
)


DEFAULT_OUTPUT_DIR = Path(
    "results/"
    "non_convergence_random_graph_vars_100_domain_10_density_0_5_"
    "seeds_0_9_iter_5000_damping_0_9_simulator_averages"
)


def late_split_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        f"split_all_at_{split_at}_transfer": {
            "class": FullRunDampedMidRunSplitEngine,
            "damping_factor": args.damping_factor,
            "split_at_iter": split_at,
            "split_factor": args.split_ratio,
            "transfer_mode": "transfer",
            **base_engine_kwargs(args),
        }
        for split_at in args.late_split_at_iters
    }


def verify_graphs_match(output_dir: Path, graphs: list[Any]) -> None:
    metadata_path = output_dir / "graph_metadata.json"
    expected = json.loads(metadata_path.read_text())
    if len(expected) != len(graphs):
        raise SystemExit(
            f"Graph count mismatch: metadata has {len(expected)}, rebuilt {len(graphs)}"
        )
    for index, (row, graph) in enumerate(zip(expected, graphs)):
        actual = graph_fingerprint(graph)
        if actual != row["graph_fingerprint"]:
            raise SystemExit(
                "Graph fingerprint mismatch at index "
                f"{index}: expected {row['graph_fingerprint']}, got {actual}"
            )


def read_average_csv(path: Path) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            engine = row["engine"]
            series.setdefault(engine, []).append(float(row["mean_global_cost"]))
    return series


def write_merged_average_csv(
    path: Path, existing_csv: Path, late_csv: Path, late_engines: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as out_handle:
        fieldnames = [
            "engine",
            "iteration",
            "mean_global_cost",
            "std_global_cost",
            "run_count",
        ]
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in [existing_csv, late_csv]:
            with source.open() as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    if source == existing_csv and row["engine"] in late_engines:
                        continue
                    if source == late_csv and row["engine"] not in late_engines:
                        continue
                    writer.writerow({field: row[field] for field in fieldnames})


def _split_iteration(engine_name: str) -> int:
    if engine_name == "split_pct_100_at_0_transfer":
        return 0
    return int(engine_name.split("_")[3])


def split_at_order_from_series(series: dict[str, list[float]]) -> list[str]:
    order = ["baseline_no_split"]
    if "split_pct_100_at_0_transfer" in series:
        order.append("split_pct_100_at_0_transfer")
    split_engines = sorted(
        (
            engine
            for engine in series
            if engine.startswith("split_all_at_") and engine.endswith("_transfer")
        ),
        key=_split_iteration,
    )
    order.extend(split_engines)
    return [engine for engine in order if engine in series]


def plot_series(
    *,
    path: Path,
    series: dict[str, list[float]],
    order: list[str],
    title: str,
    max_iter: int,
) -> None:
    plt.figure(figsize=(14, 8))
    for engine in order:
        values = series[engine][:max_iter]
        plt.plot(
            np.arange(len(values)),
            values,
            linewidth=1.7,
            label=split_at_label(engine),
        )
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Average global cost")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-name", default="late_split_at_sweep")
    parser.add_argument("--merged-source", type=Path, default=None)
    parser.add_argument("--merged-output", type=Path, default=None)
    parser.add_argument(
        "--combined-plot-name",
        default="global_cost_split_at_sweep_average_10_graphs_with_late_splits.png",
    )
    parser.add_argument(
        "--focused-plot-name",
        default="global_cost_late_split_at_sweep_average_10_graphs.png",
    )
    parser.add_argument("--graph-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-vars", type=int, default=100)
    parser.add_argument("--domain-size", type=int, default=10)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--damping-factor", type=float, default=0.9)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--ct-low", type=int, default=0)
    parser.add_argument("--ct-high", type=int, default=10)
    parser.add_argument(
        "--late-split-at-iters",
        type=int,
        nargs="+",
        default=[1000, 2000],
    )
    parser.add_argument("--timeout-seconds", type=int, default=28_800)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out
    if not (output_dir / "split_at_sweep" / "average_costs.csv").exists():
        raise SystemExit(
            "Existing split-at sweep is required before appending late split runs."
        )

    late_dir = output_dir / args.batch_name
    if late_dir.exists():
        if not args.force:
            raise SystemExit(
                f"{late_dir} already exists. Pass --force to replace it."
            )
        backup_idx = 1
        while True:
            backup = output_dir / f"{args.batch_name}.previous_{backup_idx}"
            if not backup.exists():
                shutil.move(str(late_dir), str(backup))
                break
            backup_idx += 1

    graphs, _metadata = build_graphs(
        graph_count=args.graph_count,
        seed_start=args.seed_start,
        num_vars=args.num_vars,
        domain_size=args.domain_size,
        density=args.density,
        ct_low=args.ct_low,
        ct_high=args.ct_high,
    )
    verify_graphs_match(output_dir, graphs)

    late_results = run_simulator_batch(
        batch_name=args.batch_name,
        configs=late_split_configs(args),
        graphs=graphs,
        args=args,
        output_dir=output_dir,
    )
    del late_results

    default_merged_source = (
        output_dir / "split_at_sweep" / "average_costs_with_late.csv"
    )
    existing_average = (
        args.merged_source
        if args.merged_source is not None
        else (
            default_merged_source
            if default_merged_source.exists()
            else output_dir / "split_at_sweep" / "average_costs.csv"
        )
    )
    late_average = output_dir / args.batch_name / "average_costs.csv"
    merged_average = (
        args.merged_output
        if args.merged_output is not None
        else output_dir / "split_at_sweep" / f"average_costs_with_{args.batch_name}.csv"
    )
    late_engines = [
        f"split_all_at_{split_at}_transfer" for split_at in args.late_split_at_iters
    ]
    write_merged_average_csv(
        merged_average,
        existing_average,
        late_average,
        late_engines,
    )

    combined_series = read_average_csv(merged_average)
    combined_order = split_at_order_from_series(combined_series)
    late_only_order = ["baseline_no_split", *late_engines]
    plot_series(
        path=output_dir / args.combined_plot_name,
        series=combined_series,
        order=combined_order,
        title="Average Global Cost: Splitting Iteration Sweep with Late Splits",
        max_iter=args.max_iter,
    )
    plot_series(
        path=output_dir / args.focused_plot_name,
        series=combined_series,
        order=late_only_order,
        title="Average Global Cost: Late Splitting Iteration Sweep",
        max_iter=args.max_iter,
    )

    late_settings = {
        "graph_count": args.graph_count,
        "seed_start": args.seed_start,
        "num_vars": args.num_vars,
        "domain_size": args.domain_size,
        "density": args.density,
        "max_iter": args.max_iter,
        "damping_factor": args.damping_factor,
        "split_ratio": args.split_ratio,
        "late_split_at_iters": args.late_split_at_iters,
        "merged_source": str(existing_average.relative_to(output_dir)),
        "merged_average_costs": str(merged_average.relative_to(output_dir)),
    }
    (late_dir / "settings.json").write_text(
        json.dumps(late_settings, indent=2, sort_keys=True)
    )
    print(f"OUTPUT_DIR={output_dir}", flush=True)


if __name__ == "__main__":
    main()
