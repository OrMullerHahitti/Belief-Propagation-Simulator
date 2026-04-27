"""Generate B&W analysis plots for the signal propagation experiment.

Reads signal_results_combined.csv and produces plots in subdirectories.

Usage::

    uv run python experiments/signal_propagation/plot_range_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.aij.code.utils.plot_helpers import remove_frame as _remove_frame  # noqa: E402

_AIJ_DIR = Path(__file__).resolve().parent.parent  # experiments/aij/
RESULTS_DIR = _AIJ_DIR / "data"
BASE_OUT = _AIJ_DIR


def _load_combined() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "signal_results_combined.csv")
    df["q_message_value"] = pd.to_numeric(df["q_message_value"], errors="coerce")
    return df


def _split_engines(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bp_df, damp_df) sorted by factor name."""
    bp = df[df["engine"] == "BPEngine"].sort_values("activated_factor").reset_index(drop=True)
    damp = df[df["engine"] == "DampingEngine"].sort_values("activated_factor").reset_index(drop=True)
    return bp, damp


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-30:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


# ── plot 1: dynamic range compression ────────────────────────────────────────

def plot_dynamic_range(df: pd.DataFrame) -> None:
    bp, damp = _split_engines(df)

    # sort both by undamped value so factor ordering is shared
    bp_sorted = bp.sort_values("q_message_value").reset_index(drop=True)
    factor_order = bp_sorted["activated_factor"].values
    damp_ordered = damp.set_index("activated_factor").loc[factor_order].reset_index()

    # normalize each to its own max so bars fill [0, 1]
    bp_max = bp_sorted["q_message_value"].max()
    damp_max = damp_ordered["q_message_value"].max()
    bp_norm = bp_sorted["q_message_value"].values / bp_max
    damp_norm = damp_ordered["q_message_value"].values / damp_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    y = np.arange(len(factor_order))

    # left: undamped — solid black bars
    ax1.barh(y, bp_norm, color="black", height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(factor_order, fontsize=5)
    ax1.set_xlabel("Relative signal (fraction of max)")
    ax1.set_xlim(0, 1.05)
    bp_range = bp_sorted["q_message_value"].max() - bp_sorted["q_message_value"].min()
    ax1.set_title(f"Undamped\nrange = {bp_range:.2e}", fontsize=11)
    _remove_frame(ax1)

    # right: damped — hatched bars, lighter fill
    ax2.barh(y, damp_norm, color="white", edgecolor="black", linewidth=0.5,
             hatch="///", height=0.7)
    ax2.set_xlabel("Relative signal (fraction of max)")
    ax2.set_xlim(0, 1.05)
    damp_range = damp_ordered["q_message_value"].max() - damp_ordered["q_message_value"].min()
    ax2.set_title(f"Damped\nrange = {damp_range:.2e}", fontsize=11)
    _remove_frame(ax2)

    plt.tight_layout()

    out_dir = BASE_OUT / "dynamic_range"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dynamic_range.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ── plot 4: min-max range comparison ─────────────────────────────────────────

def plot_minmax_range(df: pd.DataFrame) -> None:
    bp, damp = _split_engines(df)

    bp_vals = bp["q_message_value"].values
    damp_vals = damp["q_message_value"].values

    bp_range = float(bp_vals.max() - bp_vals.min())
    damp_range = float(damp_vals.max() - damp_vals.min())

    out_dir = BASE_OUT / "minmax_range"
    out_dir.mkdir(parents=True, exist_ok=True)

    # plot A: both on same scale (undamped dominates, damped invisible)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Undamped", "Damped"], [bp_range, damp_range], color="black", width=0.5)
    ax.set_ylabel("Signal range (max \u2212 min)")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    # annotate values on bars
    for bar, val in zip(bars, [bp_range, damp_range]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    _remove_frame(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "minmax_range_same_scale.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_dir / 'minmax_range_same_scale.png'}")

    # plot B: two subplots, each with its own scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

    b1 = ax1.bar(["Undamped"], [bp_range], color="black", width=0.4)
    ax1.set_ylabel("Signal range (max \u2212 min)")
    ax1.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax1.text(b1[0].get_x() + b1[0].get_width() / 2, bp_range,
             f"{bp_range:.2e}", ha="center", va="bottom", fontsize=9)
    _remove_frame(ax1)

    b2 = ax2.bar(["Damped"], [damp_range], color="white", edgecolor="black",
                 linewidth=1.0, hatch="///", width=0.4)
    ax2.set_ylabel("Signal range (max \u2212 min)")
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax2.text(b2[0].get_x() + b2[0].get_width() / 2, damp_range,
             f"{damp_range:.2e}", ha="center", va="bottom", fontsize=9)
    _remove_frame(ax2)

    plt.tight_layout()
    fig.savefig(out_dir / "minmax_range_split.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_dir / 'minmax_range_split.png'}")


# ── plot 2: per-factor ratio ─────────────────────────────────────────────────

def plot_per_factor_ratio(df: pd.DataFrame) -> None:
    bp, damp = _split_engines(df)

    bp_norm = _minmax_normalize(bp["q_message_value"].values)
    damp_norm = _minmax_normalize(damp["q_message_value"].values)

    # ratio of normalized values per factor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bp_norm > 1e-30, damp_norm / bp_norm, np.nan)

    factors = bp["activated_factor"].values
    mask = ~np.isnan(ratio)
    factors = factors[mask]
    ratio = ratio[mask]

    order = np.argsort(ratio)
    factors = factors[order]
    ratio = ratio[order]

    fig, ax = plt.subplots(figsize=(7, 10))

    y_pos = np.arange(len(factors))
    ax.barh(y_pos, ratio, color="black", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, fontsize=6)
    ax.set_xlabel("Damped / Undamped (normalized ratio)")

    median_ratio = float(np.median(ratio))
    ax.axvline(median_ratio, color="black", linestyle="--", linewidth=1.0, label=f"median = {median_ratio:.2f}")
    ax.legend(fontsize=9)
    _remove_frame(ax)

    plt.tight_layout()

    out_dir = BASE_OUT / "per_factor_ratio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_factor_ratio.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ── plot 3: signal uniformity ────────────────────────────────────────────────

def plot_signal_uniformity(df: pd.DataFrame) -> None:
    bp, damp = _split_engines(df)

    bp_sorted = np.sort(_minmax_normalize(bp["q_message_value"].values))
    damp_sorted = np.sort(_minmax_normalize(damp["q_message_value"].values))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ranks = np.arange(1, len(bp_sorted) + 1)
    ax.plot(ranks, bp_sorted, linestyle="-", linewidth=2.0, color="black", label="Undamped")
    ax.plot(ranks, damp_sorted, linestyle="--", linewidth=2.0, color="black", label="Damped")

    ax.set_xlabel("Factor rank")
    ax.set_ylabel("Normalized signal")
    ax.set_xlim(1, len(ranks))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    _remove_frame(ax)

    plt.tight_layout()

    out_dir = BASE_OUT / "signal_uniformity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signal_uniformity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def _superscript_int(n: int) -> str:
    _sup = {
        "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
        "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
        "8": "\u2078", "9": "\u2079", "-": "\u207b",
    }
    return "".join(_sup.get(c, c) for c in str(n))


def _scatter_stats_text(vals: np.ndarray, exponent: int | None = None) -> str:
    """Format std and range with a shared exponent for easy comparison."""
    std_val = float(np.std(vals))
    range_val = float(np.ptp(vals))
    if exponent is None:
        max_abs = max(abs(std_val), abs(range_val), 1e-300)
        exponent = int(np.floor(np.log10(max_abs)))
    scale = 10.0 ** exponent
    std_scaled = std_val / scale
    range_scaled = range_val / scale
    return f"std = {std_scaled:.2f}\nrange = {range_scaled:.2f}\n(\u00d710{_superscript_int(exponent)})"


# ── plot A: scatter undamped only ─────────────────────────────────────────────

def plot_scatter_undamped(df: pd.DataFrame) -> None:
    bp, _ = _split_engines(df)
    vals = bp["q_message_value"].values

    out_dir = BASE_OUT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.scatter(np.zeros_like(vals), vals, marker="x", color="black", s=30, linewidths=1.0)
    ax.set_xticks([])
    ax.set_ylabel("Aggregated coefficients")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    # draw once to let matplotlib pick the axis exponent, then match it
    fig.canvas.draw()
    axis_exp = int(ax.yaxis.get_offset_text().get_text().replace("1e", "").replace("\u2212", "-") or "0")
    ax.text(0.95, 0.95, _scatter_stats_text(vals, exponent=axis_exp), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace")
    ax.set_xlabel("Undamped")
    _remove_frame(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_undamped.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_dir / 'scatter_undamped.png'}")


# ── plot B: scatter damped only ───────────────────────────────────────────────

def plot_scatter_damped(df: pd.DataFrame) -> None:
    _, damp = _split_engines(df)
    vals = damp["q_message_value"].values

    out_dir = BASE_OUT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.scatter(np.zeros_like(vals), vals, marker="o", facecolors="none", edgecolors="black", s=30, linewidths=1.0)
    ax.set_xticks([])
    ax.set_ylabel("Aggregated coefficients")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.text(0.95, 0.95, _scatter_stats_text(vals), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace")
    ax.set_xlabel("Damped")
    _remove_frame(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_damped.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_dir / 'scatter_damped.png'}")


# ── plot C: side-by-side scatter (separate columns) ──────────────────────────

def _plot_scatter_side_by_side(bp_vals, damp_vals, out_path, log_scale=False):
    fig, ax = plt.subplots(figsize=(5, 6))
    # undamped at x=0, damped at x=1
    ax.scatter(np.zeros_like(bp_vals), bp_vals, marker="x", color="black", s=30, linewidths=1.0, label="Undamped")
    ax.scatter(np.ones_like(damp_vals), damp_vals, marker="o", facecolors="none", edgecolors="black",
               s=30, linewidths=1.0, label="Damped")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Undamped", "Damped"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Aggregated coefficients")
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.legend(fontsize=9)
    _remove_frame(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_scatter_side_by_side(df: pd.DataFrame) -> None:
    bp, damp = _split_engines(df)
    bp_vals = bp["q_message_value"].values
    damp_vals = damp["q_message_value"].values

    out_dir = BASE_OUT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_scatter_side_by_side(bp_vals, damp_vals, out_dir / "scatter_side_by_side.png", log_scale=False)
    _plot_scatter_side_by_side(bp_vals, damp_vals, out_dir / "scatter_side_by_side_log.png", log_scale=True)


# ── plot D: overlay scatter (same column, x vs o markers) ────────────────────

def _plot_scatter_overlay(bp_vals, damp_vals, out_path, log_scale=False):
    fig, ax = plt.subplots(figsize=(4, 6))
    # both on x=0, distinguished by marker
    jitter_bp = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(bp_vals))
    jitter_damp = np.random.default_rng(1).uniform(-0.08, 0.08, size=len(damp_vals))
    ax.scatter(jitter_bp, bp_vals, marker="x", color="black", s=30, linewidths=1.0, label="Undamped")
    ax.scatter(jitter_damp, damp_vals, marker="o", facecolors="none", edgecolors="black",
               s=30, linewidths=1.0, label="Damped")
    ax.set_xticks([])
    ax.set_ylabel("Aggregated coefficients")
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.legend(fontsize=9)
    _remove_frame(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_scatter_overlay(df: pd.DataFrame) -> None:
    bp, damp = _split_engines(df)
    bp_vals = bp["q_message_value"].values
    damp_vals = damp["q_message_value"].values

    out_dir = BASE_OUT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_scatter_overlay(bp_vals, damp_vals, out_dir / "scatter_overlay.png", log_scale=False)
    _plot_scatter_overlay(bp_vals, damp_vals, out_dir / "scatter_overlay_log.png", log_scale=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = _load_combined()
    print(f"[plot_range_analysis] loaded {len(df)} rows from signal_results_combined.csv")

    plot_dynamic_range(df)
    plot_minmax_range(df)
    plot_per_factor_ratio(df)
    plot_signal_uniformity(df)
    plot_scatter_undamped(df)
    plot_scatter_damped(df)
    plot_scatter_side_by_side(df)
    plot_scatter_overlay(df)

    print("[plot_range_analysis] done — all plots generated")


if __name__ == "__main__":
    main()
