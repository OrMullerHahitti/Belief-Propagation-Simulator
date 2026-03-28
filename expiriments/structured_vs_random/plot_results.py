"""Step 3: Plot results from the structured-vs-random experiment.

Prerequisites::

    uv run python expiriments/structured_vs_random/run_experiment.py

Run::

    uv run python expiriments/structured_vs_random/plot_results.py

Outputs (in plots/)
-------------------
cost_curves.png   — one subplot per engine; 11 lines coloured by n_random_factors
final_cost.png    — final cost vs n_random_factors for both engines on one axes
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


def _n_random_colormap(n_random_values: list[int]):
    """Return a dict mapping n_random → colour using a sequential colormap."""
    cmap = plt.cm.viridis
    norm_vals = np.linspace(0, 1, len(n_random_values))
    return {n: cmap(v) for n, v in zip(n_random_values, norm_vals)}


# ── plot A: cost curves ───────────────────────────────────────────────────────

def plot_cost_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    """One subplot per engine; 11 coloured lines (one per pct_random value)."""
    engines = sorted(df["engine"].unique())
    pct_values = sorted(df["pct_random"].unique())
    color_map = _n_random_colormap(pct_values)

    fig, axes = plt.subplots(1, len(engines), figsize=(7 * len(engines), 4.5), sharey=False)
    if len(engines) == 1:
        axes = [axes]

    for ax, engine_name in zip(axes, engines):
        for pct in pct_values:
            subset = df[(df["engine"] == engine_name) & (df["pct_random"] == pct)]
            if subset.empty:
                continue
            ax.plot(
                subset["iteration"],
                subset["cost"],
                color=color_map[pct],
                linewidth=1.4,
                label=f"{pct}%",
            )
        ax.set_title(engine_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")

        # colour legend
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=min(pct_values), vmax=max(pct_values)),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="% random factors")

    fig.suptitle("Cost curves: structured → random factors", fontsize=13)
    plt.tight_layout()

    out_path = plots_dir / "cost_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot_results] saved {out_path}")
    plt.close(fig)


# ── plot B: final cost vs n_random ────────────────────────────────────────────

def plot_final_cost(df: pd.DataFrame, plots_dir: Path) -> None:
    """Final cost vs n_random_factors for both engines on one axes."""
    engines = sorted(df["engine"].unique())
    engine_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for engine_name, color in zip(engines, engine_colors):
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
            marker="o",
            color=color,
            linewidth=2,
            label=engine_name,
        )

    ax.set_xlabel("% random factors")
    ax.set_ylabel(f"Final cost (iteration {df['iteration'].max()})")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Degradation: final cost vs. fraction of random factors")
    ax.legend()
    plt.tight_layout()

    out_path = plots_dir / "final_cost.png"
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

    plot_cost_curves(df, plots_dir)
    plot_final_cost(df, plots_dir)

    print("[plot_results] done")


if __name__ == "__main__":
    main()
