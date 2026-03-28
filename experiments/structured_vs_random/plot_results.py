"""Step 3: Plot results from the structured-vs-random experiment.

Prerequisites::

    uv run python expiriments/structured_vs_random/run_experiment.py

Run::

    uv run python expiriments/structured_vs_random/plot_results.py

Outputs (in plots/)
-------------------
cost_curves_bw.png       — B&W cost curves, filtered to interesting pct_random values
final_cost_bw.png        — B&W final cost vs pct_random
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_results(results_dir: Path) -> pd.DataFrame:
    """Concatenate all CSV files in results/ into one DataFrame."""
    csv_files = sorted(results_dir.glob("results_graph_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"no CSV files found in {results_dir} — run run_experiment.py first"
        )
    return pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)


# ── B&W line style palette ────────────────────────────────────────────────────

_BW_STYLES = [
    ("-", 2.0),
    ("--", 2.0),
    ("-.", 2.0),
    (":", 2.0),
    ((0, (5, 1)), 2.0),
    ((0, (1, 1)), 2.0),
    ((0, (3, 1, 1, 1)), 2.0),
    ((0, (5, 1, 1, 1, 1, 1)), 2.0),
    ("-", 1.0),
    ("--", 1.0),
    ("-.", 1.0),
    (":", 1.0),
]

# pct_random values to show in cost curves:
# low end in jumps (0, 10, 30, ~50) + all higher ones that don't converge immediately
_COST_CURVE_PCT_SHOW = {0, 10, 30, 49, 59, 69, 79, 89, 99}


# ── plot A: B&W cost curves ──────────────────────────────────────────────────

def plot_cost_curves_bw(df: pd.DataFrame, plots_dir: Path) -> None:
    """One subplot per engine; filtered B&W lines with dash patterns."""
    engines = sorted(df["engine"].unique())
    all_pct = sorted(df["pct_random"].unique())
    # pick the closest available values to our desired set
    pct_show = sorted(p for p in all_pct if p in _COST_CURVE_PCT_SHOW)

    fig, axes = plt.subplots(1, len(engines), figsize=(7 * len(engines), 4.5), sharey=False)
    if len(engines) == 1:
        axes = [axes]

    max_iter = int(df["iteration"].max())

    for ax, engine_name in zip(axes, engines):
        for i, pct in enumerate(pct_show):
            subset = df[(df["engine"] == engine_name) & (df["pct_random"] == pct)]
            if subset.empty:
                continue
            # extend converged runs to full length so all lines span the same x-range
            iters = subset["iteration"].values
            costs = subset["cost"].values
            if len(iters) < max_iter + 1:
                full_iters = np.arange(max_iter + 1)
                full_costs = np.empty(max_iter + 1)
                full_costs[:len(costs)] = costs
                full_costs[len(costs):] = costs[-1]
                iters, costs = full_iters, full_costs
            lstyle, lw = _BW_STYLES[i % len(_BW_STYLES)]
            ax.plot(
                iters,
                costs,
                color="black",
                linestyle=lstyle,
                linewidth=lw,
                label=f"{pct}%",
            )
        ax.set_title(engine_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.legend(fontsize=7, ncol=2, loc="best")

    plt.tight_layout()

    out_path = plots_dir / "cost_curves_bw.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot_results] saved {out_path}")
    plt.close(fig)


# ── plot B: B&W final cost ───────────────────────────────────────────────────

_ENGINE_BW_STYLES = [
    {"linestyle": "-", "linewidth": 2.0, "marker": "x", "markersize": 7},
    {"linestyle": "--", "linewidth": 2.0, "marker": "o", "markersize": 5, "fillstyle": "none"},
    {"linestyle": "-.", "linewidth": 2.0, "marker": "s", "markersize": 5, "fillstyle": "none"},
    {"linestyle": ":", "linewidth": 2.0, "marker": "^", "markersize": 5, "fillstyle": "none"},
]


def plot_final_cost_bw(df: pd.DataFrame, plots_dir: Path) -> None:
    """Final cost vs pct_random for all engines, B&W."""
    engines = sorted(df["engine"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for engine_name, style in zip(engines, _ENGINE_BW_STYLES):
        subset = df[df["engine"] == engine_name]
        final = (
            subset.sort_values("iteration")
            .groupby("pct_random")["cost"]
            .last()
            .reset_index()
        )
        ax.plot(
            final["pct_random"],
            final["cost"],
            color="black",
            label=engine_name,
            **style,
        )

    ax.set_xlabel("% random factors")
    ax.set_ylabel(f"Final cost (iteration {df['iteration'].max()})")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = plots_dir / "final_cost_bw.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot_results] saved {out_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir  = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    plots_dir   = script_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = _load_results(results_dir)
    print(f"[plot_results] loaded {len(df)} rows from {results_dir}")

    plot_cost_curves_bw(df, plots_dir)
    plot_final_cost_bw(df, plots_dir)

    print("[plot_results] done")


if __name__ == "__main__":
    main()
