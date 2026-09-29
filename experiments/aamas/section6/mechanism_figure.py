"""section 6, item 1: fraction of function-to-variable messages at a bound, per iteration.

one row of five panels (the paper's benchmarks), linear iteration axis in paper units
(two paper iterations per library iteration), lines MS / DMS / MS-SCFG / DMS-SCFG and the
delayed split DMS-kDS at one k. "at a bound" = committed arc in exp1's sense: one sender value
minimizes for every receiver value, which in the binary case is exactly a message difference
sitting at a bound of Section 4.

inputs: experiments/aamas/splitting_explanation/results/exp1_<bench>.npz and exp1_delayed_<bench>.npz
output: experiments/aamas/section6/out/mechanism_fraction_at_bound.pdf

usage: uv run python experiments/aamas/section6/mechanism_figure.py [--k 500]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "splitting_explanation" / "results"
OUT = HERE / "out"
BENCHES = [
    ("random_sparse", "random sparse"),
    ("random_dense", "random dense"),
    ("scale_free", "scale free"),
    ("graph_coloring", "graph coloring"),
    ("meeting_scheduling", "meeting scheduling"),
]
LINES = {
    "MS": dict(label="MS", color="0.55", ls=":"),
    "DMS": dict(label="DMS", color="0.2", ls="--"),
    "MS_split": dict(label="MS-SCFG", color="tab:orange", ls=":"),
    "DMS_split": dict(label="DMS-SCFG", color="tab:red", ls="-"),
}
DELAYED = dict(color="tab:blue", ls="-")


def remove_frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="split iteration of the DMS-kDS line, library units",
    )
    args = parser.parse_args()
    k = args.k

    fig, axes = plt.subplots(
        1, len(BENCHES), figsize=(1.45 * len(BENCHES), 1.9), sharey=True
    )
    for ax, (bench, title) in zip(axes, BENCHES):
        z = np.load(RESULTS / f"exp1_{bench}.npz")
        zd = np.load(RESULTS / f"exp1_delayed_{bench}.npz")
        n_it = z["MS/sats"].shape[1]
        it = 2 * np.arange(1, n_it + 1)  # paper iterations
        for alg, style in LINES.items():
            ax.plot(it, z[f"{alg}/sats"].mean(axis=0), lw=1.0, **style)
        ax.plot(
            it,
            zd[f"DMS_split_at_{k}/sats"].mean(axis=0),
            lw=1.0,
            label=f"DMS-kDS, k={2 * k}",
            **DELAYED,
        )
        ax.axvline(2 * k, color="tab:blue", lw=0.5, ls=":", alpha=0.7)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, 2 * n_it)
        ax.set_xticks([0, 2000, 4000])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("iteration", fontsize=7)
        remove_frame(ax)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("messages at a bound", fontsize=7)
    # legend outside the right-most panel, so it covers no line
    axes[-1].legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(w_pad=0.6)
    OUT.mkdir(exist_ok=True)
    path = OUT / "mechanism_fraction_at_bound.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")
    # the numbers behind the figure: final fraction and the level just before the split
    print("bench            MS    DMS  MS-SCFG DMS-SCFG | DMS at k  kDS final")
    for bench, _ in BENCHES:
        z = np.load(RESULTS / f"exp1_{bench}.npz")
        zd = np.load(RESULTS / f"exp1_delayed_{bench}.npz")
        s = zd[f"DMS_split_at_{k}/sats"]
        print(
            f"{bench:18}"
            + " ".join(f"{z[f'{a}/sats'][:, -1].mean():6.2f}" for a in LINES)
            + f" | {s[:, k - 1].mean():6.2f}  {s[:, -1].mean():6.2f}"
        )


if __name__ == "__main__":
    main()
