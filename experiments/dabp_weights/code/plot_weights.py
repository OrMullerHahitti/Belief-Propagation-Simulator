"""Exploratory figures for the DABP-SymSplit edge-weight analysis.

Reads the CSVs written by analyze_weights.py (plus the raw npz files for
trajectory panels) and writes four multi-panel PDF figures into ``plots/``:

- pair_asymmetry.pdf        does DABP break the 0.5/0.5 split symmetry?
- weights_distribution.pdf  effective damping / attention weight distributions
- trajectories.pdf          weight dynamics over iterations (representative seed)
- structure_correlation.pdf effective damping vs graph structure

Example:
    uv run python experiments/dabp_weights/code/plot_weights.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.plot_helpers import remove_frame  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"

# okabe-ito, one color per attention head
HEAD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
ROLL_WINDOW = 20


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def first_pair_edges(run: dict) -> tuple[int, int]:
    """edge rows of the two halves of the first (variable, factor) pair."""
    slot: dict[tuple[int, int], list] = {}
    for k in range(run["damped"].shape[1]):
        fn = int(run["trg_fn"][k])
        key = (int(run["trg_var_idx"][k]), int(run["fn_orig_idx"][fn]))
        slot.setdefault(key, [None, None])[int(run["fn_half"][fn])] = k
    k0, k1 = slot[min(slot)]
    return k0, k1


def phase_guides(ax, update_interval: int, xmax: float) -> None:
    for x in range(update_interval, int(xmax) + 1, update_interval):
        ax.axvline(x, color="gray", lw=0.5, alpha=0.15, zorder=0)


def save(fig, out: Path) -> None:
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def plot_pair_asymmetry(data_dir: Path, plots_dir: Path, update_interval: int) -> None:
    pairs = pd.read_csv(data_dir / "pair_asymmetry.csv")
    dyn = pd.read_csv(data_dir / "pair_asymmetry_dynamics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.hist(pairs["log_ratio"] / np.log(10), bins=60, color="#0072B2", alpha=0.85)
    ax.axvline(0.0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("final log10(w half0 / w half1)")
    ax.set_ylabel("pair-head count")

    ax = axes[0, 1]
    ax.plot(dyn["iteration"], dyn["abs_log_ratio_median"], color="#0072B2", lw=1.5)
    ax.fill_between(
        dyn["iteration"],
        dyn["abs_log_ratio_q25"],
        dyn["abs_log_ratio_q75"],
        color="#0072B2",
        alpha=0.25,
        lw=0,
    )
    ax.plot(
        dyn["iteration"], dyn["abs_log_ratio_max"], color="#D55E00", lw=0.8, alpha=0.7
    )
    phase_guides(ax, update_interval, dyn["iteration"].max())
    ax.set_xlabel("iteration")
    ax.set_ylabel("|log(w0/w1)| (median, IQR, max)")

    ax = axes[1, 0]
    for h, color in enumerate(HEAD_COLORS):
        sub = pairs[pairs["head"] == h]
        ax.scatter(sub["w_half0"], sub["w_half1"], s=6, alpha=0.4, color=color)
    lims = [
        min(pairs["w_half0"].min(), pairs["w_half1"].min()),
        max(pairs["w_half0"].max(), pairs["w_half1"].max()),
    ]
    ax.plot(lims, lims, color="black", lw=0.8, ls="--")
    ax.set_xlabel("w half0 (final)")
    ax.set_ylabel("w half1 (final)")

    ax = axes[1, 1]
    values = np.sort(pairs["asym"].to_numpy())
    ax.plot(values, np.arange(1, values.size + 1) / values.size, color="#0072B2")
    ax.set_xlabel("|w0 - w1| / (w0 + w1) (final)")
    ax.set_ylabel("ECDF")

    for ax in axes.ravel():
        remove_frame(ax)
    fig.tight_layout()
    save(fig, plots_dir / "pair_asymmetry.pdf")


def plot_weights_distribution(data_dir: Path, plots_dir: Path) -> None:
    edges = pd.read_csv(data_dir / "edge_weights.csv")
    attention = pd.read_csv(data_dir / "attention_final.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    lam = edges.groupby(["seed", "var", "factor", "half"])["w_final"].mean()
    ax.hist(lam, bins=40, color="#0072B2", alpha=0.85)
    ax.set_xlabel("effective damping (head mean, final)")
    ax.set_ylabel("edge count")

    ax = axes[0, 1]
    data = [edges.loc[edges["head"] == h, "w_final"] for h in range(len(HEAD_COLORS))]
    boxes = ax.boxplot(
        data, tick_labels=[f"head {h}" for h in range(len(data))], patch_artist=True
    )
    for patch, color in zip(boxes["boxes"], HEAD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("previous-message weight (final)")

    ax = axes[1, 0]
    ax.hist(attention["weight"], bins=40, color="#009E73", alpha=0.85)
    ax.set_xlabel("attention weight of source in target group (final)")
    ax.set_ylabel("row count")

    ax = axes[1, 1]
    ax.hist(
        attention.loc[attention["is_twin"] == 1, "weight"],
        bins=40,
        color="#D55E00",
        alpha=0.85,
    )
    ax.set_xlabel("attention weight of the twin half (final)")
    ax.set_ylabel("row count")

    for ax in axes.ravel():
        remove_frame(ax)
    fig.tight_layout()
    save(fig, plots_dir / "weights_distribution.pdf")


def plot_trajectories(data_dir: Path, plots_dir: Path) -> None:
    raw_paths = sorted((data_dir / "raw").glob("seed*.npz"))
    rep = load_run(raw_paths[0])
    lam_rep = rep["damped"][:, :, 1, :].mean(axis=2)
    update_interval = int(rep["update_interval"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(lam_rep, color="#0072B2", lw=0.7, alpha=0.3)
    phase_guides(ax, update_interval, lam_rep.shape[0])
    ax.set_xlabel(f"iteration (seed {int(rep['seed'])})")
    ax.set_ylabel("effective damping per edge")

    ax = axes[0, 1]
    for path in raw_paths:
        run = load_run(path)
        lam = run["damped"][:, :, 1, :].mean(axis=2)
        if lam.shape[0] < ROLL_WINDOW + 1:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(lam, ROLL_WINDOW, axis=0)
        rolling = windows.std(axis=-1).mean(axis=1)
        ax.plot(
            np.arange(ROLL_WINDOW - 1, lam.shape[0]),
            rolling,
            color="#0072B2",
            lw=0.8,
            alpha=0.35,
        )
    ax.set_xlabel("iteration (all seeds)")
    ax.set_ylabel(f"rolling-{ROLL_WINDOW} std of damping (edge mean)")

    ax = axes[1, 0]
    k0, k1 = first_pair_edges(rep)
    for h, color in enumerate(HEAD_COLORS):
        ax.plot(rep["damped"][:, k0, 1, h], color=color, lw=1.2, label=f"head {h}")
        ax.plot(rep["damped"][:, k1, 1, h], color=color, lw=1.2, ls="--")
    phase_guides(ax, update_interval, rep["damped"].shape[0])
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration (one pair: solid=half0, dashed=half1)")
    ax.set_ylabel("previous-message weight")

    ax = axes[1, 1]
    ax.plot(rep["costs"], color="#0072B2", lw=1.2)
    if bool(rep["converged"]):
        ax.axvline(int(rep["t_first_stable"]), color="#D55E00", lw=1.0, ls="--")
    ax.set_xlabel(f"iteration (seed {int(rep['seed'])})")
    ax.set_ylabel("solution cost")

    for ax in axes.ravel():
        remove_frame(ax)
    fig.tight_layout()
    save(fig, plots_dir / "trajectories.pdf")


def plot_structure_correlation(data_dir: Path, plots_dir: Path) -> None:
    structure = pd.read_csv(data_dir / "structure_correlation.csv")
    stats = pd.read_csv(data_dir / "correlation_stats.csv").set_index("feature")

    def annotate(ax, feature: str) -> None:
        rho = stats.loc[feature, "spearman_rho"]
        p = stats.loc[feature, "p_value"]
        ax.text(
            0.02,
            0.98,
            f"Spearman rho={rho:.3f}, p={p:.2g}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    degrees = sorted(structure["var_degree"].unique())
    data = [structure.loc[structure["var_degree"] == d, "lam_final"] for d in degrees]
    ax.boxplot(data, tick_labels=[str(d) for d in degrees])
    annotate(ax, "var_degree")
    ax.set_xlabel("variable primal degree")
    ax.set_ylabel("effective damping (pair mean, final)")

    ax = axes[0, 1]
    data = [structure.loc[structure["edge_on_cycle"] == v, "lam_final"] for v in (0, 1)]
    ax.boxplot(data, tick_labels=["bridge", "on cycle"])
    annotate(ax, "edge_on_cycle")
    ax.set_ylabel("effective damping (pair mean, final)")

    ax = axes[1, 0]
    ax.scatter(
        structure["ct_std"], structure["lam_final"], s=8, alpha=0.4, color="#0072B2"
    )
    annotate(ax, "ct_std")
    ax.set_xlabel("cost-table std")
    ax.set_ylabel("effective damping (pair mean, final)")

    ax = axes[1, 1]
    ax.scatter(
        structure["ct_range"], structure["lam_final"], s=8, alpha=0.4, color="#0072B2"
    )
    annotate(ax, "ct_range")
    ax.set_xlabel("cost-table range")
    ax.set_ylabel("effective damping (pair mean, final)")

    for ax in axes.ravel():
        remove_frame(ax)
    fig.tight_layout()
    save(fig, plots_dir / "structure_correlation.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    first = load_run(sorted((args.data_dir / "raw").glob("seed*.npz"))[0])
    update_interval = int(first["update_interval"])

    plot_pair_asymmetry(args.data_dir, args.plots_dir, update_interval)
    plot_weights_distribution(args.data_dir, args.plots_dir)
    plot_trajectories(args.data_dir, args.plots_dir)
    plot_structure_correlation(args.data_dir, args.plots_dir)


if __name__ == "__main__":
    main()
