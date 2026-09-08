"""One summary figure for the DABP damping audit (Codex worktree, branch orx/dabp-damping-audit).

The audit asks whether DABP's learning changes anything. Phase 1 replaced the
learned damping weight with a fixed 0.5 or 0.9 on the saved seed-0 20-node
graph (3 representations x 3 damping settings x 3 network seeds, 500
iterations, no early stop). Phase 2 additionally replaced the learned
attention with uniform shares on 5 new graphs x 2 objectives (540 runs).
Its result CSVs live in the worktree's gitignored ``results/`` directory, so
the location is a flag.

Example:
    uv run python experiments/dabp_plots/code/plot_audit.py
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BLUE,
    GREEN,
    PLOTS_ROOT,
    START_WEIGHT,
    VERMILION,
    plain_axes,
    save,
)

DEFAULT_RESULTS = Path(
    "/Users/or/.codex/worktrees/ab25/Belief-Propagation-Simulator/results/dabp_damping_audit"
)
OUT_DIR = PLOTS_ROOT / "damping_audit"
REPR_ORDER = ("unsplit", "split_50", "split_95")
REPR_LABELS = {
    "unsplit": "no split",
    "split_50": "0.5/0.5 split",
    "split_95": "0.95/0.05 split",
}
DAMP_ORDER = ("learned", "fixed_50", "fixed_90")
DAMP_LABELS = {
    "learned": "learned damping",
    "fixed_50": "damping fixed at 0.5",
    "fixed_90": "damping fixed at 0.9",
}
DAMP_COLORS = {"learned": BLUE, "fixed_50": GREEN, "fixed_90": VERMILION}
DAMP_OFFSETS = {"learned": -0.22, "fixed_50": 0.0, "fixed_90": 0.22}
# the softmax over two bounded scores keeps the damping weight inside these limits (AUDIT.md)
REACHABLE = (1.0 / (1.0 + np.e), np.e / (1.0 + np.e))
PAIR_KEY = ["problem", "representation", "damping", "seed"]


def _title(ax, text: str) -> None:
    ax.set_title(textwrap.fill(text, 62), fontsize=9, loc="left")


def dot_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    for damping in DAMP_ORDER:
        sub = df[df["damping"] == damping]
        xs = [
            REPR_ORDER.index(r) + DAMP_OFFSETS[damping] for r in sub["representation"]
        ]
        ax.scatter(
            xs,
            sub[column],
            color=DAMP_COLORS[damping],
            s=30,
            alpha=0.8,
            label=DAMP_LABELS[damping],
        )
    ax.set_xticks(range(len(REPR_ORDER)))
    ax.set_xticklabels([REPR_LABELS[r] for r in REPR_ORDER])
    ax.set_ylabel("cost on the original tables")
    _title(ax, title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    pilot = pd.read_csv(args.results / "pilot_500" / "summary.csv")
    matched = pd.read_csv(args.results / "matched_500" / "summary.csv")
    saved = pilot[pilot["problem"] == "saved_20"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    dot_panel(
        axes[0, 0],
        saved,
        "best_cost",
        "Saved 20-node graph, seed 0: best cost found within 500 iterations "
        "(3 network seeds each; identical values overlap)",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    dot_panel(
        axes[0, 1],
        saved,
        "final_cost",
        "The same runs: cost at iteration 500 (above the best = the run drifted away from it)",
    )

    ax = axes[1, 0]
    learned = matched[matched["attention"] == "learned"].set_index(PAIR_KEY)[
        "best_cost"
    ]
    uniform = matched[matched["attention"] == "uniform"].set_index(PAIR_KEY)[
        "best_cost"
    ]
    pairs = pd.concat(
        [learned.rename("learned"), uniform.rename("uniform")], axis=1, join="inner"
    ).reset_index()
    for family, color in (("coloring", BLUE), ("shuffled", VERMILION)):
        sub = pairs[pairs["problem"].str.startswith(family)]
        ax.scatter(
            sub["uniform"],
            sub["learned"],
            s=18,
            alpha=0.6,
            color=color,
            label=f"{family} tables ({len(sub)} pairs)",
        )
    lo, hi = (
        pairs[["uniform", "learned"]].min().min(),
        pairs[["uniform", "learned"]].max().max(),
    )
    pad = (hi - lo) * 0.05
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="black",
        lw=0.8,
        ls="--",
        label="same cost",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("best cost with uniform attention (every neighbor weighted equally)")
    ax.set_ylabel("best cost with learned attention")
    same = int((np.abs(pairs["learned"] - pairs["uniform"]) < 1e-9).sum())
    # a violated constraint costs 10; the fractional part is the unary tie-break
    violations = np.floor(pairs[["learned", "uniform"]] / 10)
    same_violations = int((violations["learned"] == violations["uniform"]).sum())
    _title(
        ax,
        f"Phase 2: learned against uniform attention on the same run. {same} of "
        f"{len(pairs)} pairs reach exactly the same best cost; {same_violations} of "
        f"{len(pairs)} violate the same number of constraints (the rest differ only "
        f"by tie-break terms, at most {np.abs(pairs['learned'] - pairs['uniform']).max():.3f})",
    )

    ax = axes[1, 1]
    learned_runs = pd.concat(
        [pilot[pilot["damping"] == "learned"], matched[matched["damping"] == "learned"]]
    )
    # runs of a single unsplit factor have no variable-to-factor edge and so no weight
    learned_runs = learned_runs.dropna(subset=["damping_min", "damping_max"])
    lo, hi = learned_runs["damping_min"].min(), learned_runs["damping_max"].max()
    bins = np.linspace(lo, hi, 40)
    ax.hist(
        learned_runs["damping_min"],
        bins=bins,
        color=BLUE,
        alpha=0.6,
        label="lowest per-head weight seen in the run",
    )
    ax.hist(
        learned_runs["damping_max"],
        bins=bins,
        color=VERMILION,
        alpha=0.6,
        label="highest per-head weight seen in the run",
    )
    ax.axvline(START_WEIGHT, color="black", lw=0.8, ls="--", label="start value 0.5")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("damping weight (weight on the previous message)")
    ax.set_ylabel("number of runs")
    _title(
        ax,
        f"Where the learned damping weight went in the {len(learned_runs)} runs that "
        f"learn it: every run stayed between {lo:.5f} and {hi:.5f}, while the network "
        f"could reach anything from {REACHABLE[0]:.3f} to {REACHABLE[1]:.3f} (inset)",
    )
    inset = ax.inset_axes([0.6, 0.4, 0.38, 0.25])
    inset.axvspan(*REACHABLE, color="lightgray", label="reachable by the network")
    inset.axvline(
        START_WEIGHT, color=BLUE, lw=3, label=f"observed: {lo:.4f} to {hi:.4f}"
    )
    inset.set_xlim(REACHABLE[0] - 0.03, REACHABLE[1] + 0.03)
    inset.set_xticks([REACHABLE[0], START_WEIGHT, REACHABLE[1]])
    inset.set_xticklabels(
        [f"{REACHABLE[0]:.3f}", "0.5", f"{REACHABLE[1]:.3f}"], fontsize=7
    )
    inset.set_yticks([])
    inset.legend(
        frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.5)
    )
    for side in ("top", "right", "left"):
        inset.spines[side].set_visible(False)

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, args.out_dir / "audit_summary.pdf")


if __name__ == "__main__":
    main()
