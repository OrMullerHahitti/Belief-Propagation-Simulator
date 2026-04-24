"""Plot generation for non-convergence diagnostics."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_plots(
    run_results: list[dict[str, Any]], output_dir: str | Path
) -> list[Path]:
    """Write standard plots for every retained run trace."""

    out = Path(output_dir)
    aggregate_plot_dir = out / "plots"
    aggregate_plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for result in run_results:
        name = result["run_name"]
        trace = result["trace"]
        if not trace:
            continue
        artifact_stem = result.get("artifact_stem")
        plot_dir = out / name / "plots"
        if artifact_stem:
            plot_dir = plot_dir / artifact_stem
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(
            [
                _plot_assignments(trace, plot_dir / "assignments.png"),
                _plot_cost(trace, plot_dir / "global_cost.png"),
                _plot_beliefs(trace, plot_dir / "beliefs.png"),
                _plot_deltas(trace, "belief_deltas", plot_dir / "belief_deltas.png"),
                _plot_deltas(trace, "Q_deltas", plot_dir / "q_deltas.png"),
                _plot_deltas(trace, "R_deltas", plot_dir / "r_deltas.png"),
                _plot_parity(trace, plot_dir / "parity.png"),
                _plot_diagonal_orientation(
                    trace, plot_dir / "diagonal_orientation.png"
                ),
            ]
        )
    if run_results:
        paths.append(
            _plot_run_comparison(
                run_results, aggregate_plot_dir / "run_cost_comparison.png"
            )
        )
    return [path for path in paths if path.exists()]


def _plot_assignments(trace: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [row["iteration"] for row in trace]
    variables = sorted({key for row in trace for key in row.get("assignments", {})})
    for variable in variables:
        ax.step(
            xs,
            [row.get("assignments", {}).get(variable) for row in trace],
            where="post",
            label=variable,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Assignment")
    ax.legend(loc="best")
    return _save(fig, path)


def _plot_cost(trace: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        [row["iteration"] for row in trace], [row.get("global_cost") for row in trace]
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global cost")
    return _save(fig, path)


def _plot_beliefs(trace: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [row["iteration"] for row in trace]
    keys = sorted(
        (var, idx)
        for row in trace
        for var, values in row.get("beliefs", {}).items()
        for idx in range(len(values))
    )
    for variable, idx in keys:
        ax.plot(
            xs,
            [
                row.get("beliefs", {}).get(variable, [None] * (idx + 1))[idx]
                for row in trace
            ],
            label=f"{variable}[{idx}]",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Belief")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, path)


def _plot_deltas(trace: list[dict[str, Any]], bucket: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [row["iteration"] for row in trace]
    keys = sorted({key for row in trace for key in row.get(bucket, {})})
    for key in keys:
        ax.plot(xs, [row.get(bucket, {}).get(key) for row in trace], label=key)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(bucket)
    if keys:
        ax.legend(loc="best", fontsize=7)
    return _save(fig, path)


def _plot_parity(trace: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [row["iteration"] for row in trace]
    first_delta = sorted({key for row in trace for key in row.get("belief_deltas", {})})
    if first_delta:
        key = first_delta[0]
        even_x = [row["iteration"] for row in trace if row["iteration"] % 2 == 0]
        even_y = [
            row["belief_deltas"].get(key) for row in trace if row["iteration"] % 2 == 0
        ]
        odd_x = [row["iteration"] for row in trace if row["iteration"] % 2 == 1]
        odd_y = [
            row["belief_deltas"].get(key) for row in trace if row["iteration"] % 2 == 1
        ]
        ax.scatter(even_x, even_y, label=f"even {key}", s=16)
        ax.scatter(odd_x, odd_y, label=f"odd {key}", s=16)
    else:
        ax.plot(xs, [0 for _ in xs])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parity diagnostic")
    ax.legend(loc="best")
    return _save(fig, path)


def _plot_diagonal_orientation(trace: list[dict[str, Any]], path: Path) -> Path:
    orientation_code = {"unknown": 0, "main": 1, "anti": 2, "mixed": 3}
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = [row["iteration"] for row in trace]
    factors = sorted(
        {factor for row in trace for factor in row.get("selected_minimizers", {})}
    )
    for factor in factors:
        ys = []
        for row in trace:
            entries = []
            for metadata in row.get("selected_minimizers", {}).get(factor, {}).values():
                entries.extend(metadata.get("selected_entries", []))
            from .diagonal_analyzer import binary_diagonal_orientation

            ys.append(orientation_code[binary_diagonal_orientation(entries)])
        ax.step(xs, ys, where="post", label=factor)
    ax.set_yticks(list(orientation_code.values()), list(orientation_code.keys()))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Orientation")
    if factors:
        ax.legend(loc="best")
    return _save(fig, path)


def _plot_run_comparison(run_results: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in run_results:
        trace = result["trace"]
        if trace:
            ax.plot(
                [row["iteration"] for row in trace],
                [row.get("global_cost") for row in trace],
                label=result["run_name"],
            )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global cost")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tight layout not applied.*",
            category=UserWarning,
        )
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
