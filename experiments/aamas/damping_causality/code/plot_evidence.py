"""Render minimal, source-backed side-by-side damping evidence figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.aaai.code.utils.plot_helpers import remove_frame  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/damping_causality_20260915/native"
OUT = SOURCE.parent / "figures"
COLORS = ("#454B54", "#2456A6", "#888E96")


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def _frame() -> tuple:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.85, wspace=0.24)
    for ax in axes:
        remove_frame(ax)
        ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Iteration")
    return fig, axes


def _save(fig, axes, name: str) -> None:
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        frameon=False,
        ncol=3,
        handlelength=2.8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Plot saved native trajectories; preserve all fluctuations and raw costs."""
    OUT.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    plain = _load("robust_path_split05_d0")
    damped = _load("robust_path_split05_d05")
    fig, axes = _frame()
    for data, label, color, style in (
        (plain, "MS · split 0.5", COLORS[0], "--"),
        (damped, "DMS · split 0.5 · λ = 0.5", COLORS[1], "-"),
    ):
        updates = np.arange(1, 51)
        axes[0].plot(
            updates,
            data["beliefs"][:50, 1, 1],
            label=label,
            color=color,
            linestyle=style,
            linewidth=1.4,
        )
        axes[1].plot(
            updates,
            data["costs"][:50],
            color=color,
            linestyle=style,
            linewidth=1.4,
        )
    axes[0].axhline(0, color="#ADB1B7", linewidth=0.8)
    axes[0].set_ylabel("Belief difference, x₂")
    axes[1].set_ylabel("Original cost")
    axes[1].set_yticks([13, 32, 61])
    for ax in axes:
        ax.set_xlim(1, 50)
    _save(fig, axes, "path_damping")

    fig, axes = _frame()
    switched = _load("robust_path_split05_d0_to_d05_at32_to_d0_at256")
    for data, label, color, style in (
        (plain, "MS throughout", COLORS[0], "--"),
        (switched, "Damping during updates 33–256", COLORS[1], "-"),
    ):
        updates = np.arange(1, 301)
        axes[0].plot(
            updates,
            data["costs"][:300],
            label=label,
            color=color,
            linestyle=style,
            linewidth=1,
        )
        axes[1].step(
            updates,
            data["damping"][:300],
            where="post",
            color=color,
            linestyle=style,
            linewidth=1.6,
        )
    axes[0].set_ylabel("Original cost")
    axes[0].set_yticks([13, 32, 61])
    axes[1].set_ylabel("Old-Q damping, λ")
    axes[1].set_ylim(-0.04, 0.55)
    for ax in axes:
        ax.set_xlim(1, 300)
    _save(fig, axes, "same_state_intervention")

    triangle = _load("frustrated_triangle_split05_d09")
    fig, axes = _frame()
    start, end = len(triangle["costs"]) - 100, len(triangle["costs"])
    updates = np.arange(start + 1, end + 1)
    axes[0].plot(
        updates,
        triangle["beliefs"][start:end, 1, 1],
        color=COLORS[1],
        label="DMS · triangle · split 0.5 · λ = 0.9",
        linewidth=1.5,
    )
    axes[0].axhline(0, color="#ADB1B7", linewidth=0.8)
    axes[1].plot(updates, triangle["costs"][start:end], color=COLORS[1], linewidth=1.5)
    axes[0].set_ylabel("Belief difference, x₂")
    axes[1].set_ylabel("Original cost")
    axes[1].ticklabel_format(axis="y", useOffset=False)
    for ax in axes:
        ax.set_xlim(start + 1, end)
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_xticks(np.linspace(start + 1, end, 4, dtype=int))
        ax.get_xticklabels()[0].set_horizontalalignment("left")
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
    _save(fig, axes, "damping_counterexample")

    paths = [
        SOURCE / f"{name}.npz"
        for name in (
            "robust_path_split05_d0",
            "robust_path_split05_d05",
            "robust_path_split05_d0_to_d05_at32_to_d0_at256",
            "frustrated_triangle_split05_d09",
        )
    ]
    contract = {
        "renderer": "Matplotlib PNG and vector PDF",
        "layout": "side-by-side; no title, subtitle, or footnote per user request",
        "data": "individual native runs; actual per-update values; no smoothing or averaging",
        "metric": "reference-label belief difference and original-objective decoded cost",
        "scope": "selected examples, not benchmark averages",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
