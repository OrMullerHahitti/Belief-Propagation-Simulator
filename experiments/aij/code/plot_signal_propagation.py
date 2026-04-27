"""Visualize signal propagation results.

Reads ``results/`` produced by ``run_experiment.py`` and generates:
- Per-engine factor graph comparison (each with its own color scale)
- Per-engine bar charts side by side
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.aij.code.utils.plot_helpers import remove_frame as _remove_frame  # noqa: E402

_AIJ_DIR = Path(__file__).resolve().parent.parent  # experiments/aij/
RESULTS_DIR = _AIJ_DIR / "data"
PLOTS_DIR = _AIJ_DIR / "plots"

plt.rcParams.update({
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})
VAR_COLOR = "#4c72b0"
OBSERVER_COLOR = "#e63946"


def _build_nx_graph(meta):
    """Reconstruct bipartite nx graph from saved topology."""
    G = nx.Graph()
    var_set = set()
    for fname, va, vb in meta["topology_edges"]:
        var_set.update([va, vb])
        G.add_node(fname, bipartite=1, kind="factor")
        G.add_edge(fname, va)
        G.add_edge(fname, vb)
    for vn in var_set:
        G.nodes[vn]["bipartite"] = 0
        G.nodes[vn]["kind"] = "variable"
    return G, sorted(var_set)


def _load_combined(results_dir=RESULTS_DIR):
    """Load combined multi-engine results."""
    with open(results_dir / "signal_propagation_metadata.json") as f:
        meta = json.load(f)
    df = pd.read_csv(results_dir / "signal_results_combined.csv")
    df["belief_at_observer"] = pd.to_numeric(df["belief_at_observer"])
    return meta, df


def plot_graph_per_engine(meta, df_combined, G, var_names, plots_dir):
    """Side-by-side factor graphs, each engine with its own normalized color scale."""
    engines = df_combined["engine"].unique()
    n_engines = len(engines)
    observer = meta["observer"]
    factor_names = meta["factor_names"]

    pos = nx.spring_layout(G, seed=42, k=1.8 / np.sqrt(len(G)))
    cmap = plt.cm.YlOrRd
    regular_vars = [v for v in var_names if v != observer]

    fig, axes = plt.subplots(1, n_engines, figsize=(12 * n_engines, 9))
    if n_engines == 1:
        axes = [axes]

    for ax, engine_name in zip(axes, engines):
        engine_df = df_combined[df_combined["engine"] == engine_name]
        belief_map = dict(zip(engine_df["activated_factor"], engine_df["belief_at_observer"].astype(float)))

        factor_vals = np.array([float(belief_map.get(fn, 0)) for fn in factor_names])
        # per-engine normalization: each engine gets its own [0, 1] scale
        safe_vals = np.where(factor_vals > 0, factor_vals, 1.0)
        log_vals = np.log10(safe_vals)
        vmin, vmax = log_vals.min(), log_vals.max()
        normed = (log_vals - vmin) / max(vmax - vmin, 1e-12)
        factor_colors = [cmap(v) for v in normed]

        ax.set_axis_off()
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=regular_vars, node_color=VAR_COLOR,
                               node_size=400, ax=ax, edgecolors="white", linewidths=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=[observer], node_color=OBSERVER_COLOR,
                               node_size=700, ax=ax, edgecolors="white", linewidths=2.0)
        nx.draw_networkx_nodes(G, pos, nodelist=factor_names, node_color=factor_colors,
                               node_shape="s", node_size=300, ax=ax,
                               edgecolors="#666", linewidths=0.5)
        nx.draw_networkx_labels(G, pos, font_size=6, font_color="#222", ax=ax)

        # per-engine colorbar
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.7)
        cbar.set_label(f"log\u2081\u2080(belief) — {engine_name}", fontsize=9)

        ax.set_title(engine_name, fontsize=13, fontweight="bold", pad=15)

    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VAR_COLOR,
               markersize=10, label="variable"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OBSERVER_COLOR,
               markersize=12, label=f"observer ({observer})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#dd8452",
               markersize=9, label="factor (colored by signal)"),
    ]
    fig.legend(handles=legend_els, loc="upper left", fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.01, 0.98))

    fig.suptitle(f"signal propagation — per-engine (own color scale), observer={observer}, {meta['max_iter']} iterations",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = plots_dir / "factor_graph_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def plot_comparison_bars(meta, df_combined, plots_dir):
    """Side-by-side bar charts, one per engine, each with own scale."""
    engines = df_combined["engine"].unique()
    n_engines = len(engines)

    # sort by first engine's values
    first_df = df_combined[df_combined["engine"] == engines[0]].sort_values("belief_at_observer", ascending=True)
    factor_order = first_df["activated_factor"].tolist()
    n = len(factor_order)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    fig, axes = plt.subplots(1, n_engines, figsize=(8 * n_engines, max(7, n * 0.28)))
    if n_engines == 1:
        axes = [axes]

    for ax, engine_name in zip(axes, engines):
        engine_df = df_combined[df_combined["engine"] == engine_name]
        belief_map = dict(zip(engine_df["activated_factor"], engine_df["belief_at_observer"].astype(float)))
        vals = [belief_map[fn] for fn in factor_order]

        ax.barh(range(n), vals,
                color=colors, edgecolor="white", linewidth=0.3, height=0.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(factor_order, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("belief at observer (log scale)", fontsize=10)
        ax.set_title(f"{engine_name} — signal at {meta['observer']}",
                     fontsize=11, fontweight="bold")
        _remove_frame(ax)

    plt.tight_layout()
    out = plots_dir / "signal_comparison_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta, df_combined = _load_combined()
    G, var_names = _build_nx_graph(meta)

    plot_graph_per_engine(meta, df_combined, G, var_names, plots_dir)
    plot_comparison_bars(meta, df_combined, plots_dir)


if __name__ == "__main__":
    main()
