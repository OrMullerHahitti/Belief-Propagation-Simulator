"""Edge-weight ratio of the two halves of a factor, one line per pair, seed 0 of the 0.95/0.05 run.

For every message a variable sends to a third factor, both halves of another
factor f are among its sources and each gets an attention share. The ratio
share(half A) / share(half B) (head mean) is 1 when the halves are weighted
equally. Writes into ``small_10agents_50seeds/split_0.95_0.05/``:

- ``edge_pair_ratios_all.pdf``: every pair's ratio over the run in one plot
- ``edge_pair_ratios_grid.pdf``: one small panel per pair; when there are more
  pairs than ``--max-panels`` the grid holds an evenly spaced sample of the
  pairs ordered by how far their final ratio is from 1 (so the least and the
  most asymmetric pairs are always included)

Example:
    uv run python experiments/dabp_plots/code/plot_pair_ratios.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BLUE,
    PLOTS_ROOT,
    SMALL_DATA,
    load_small_run,
    plain_axes,
    save,
    split_dirname,
    update_guides,
)
from figures import _source_pair_name, _source_pairs  # noqa: E402

DATA_FILE = SMALL_DATA / "data_asym" / "raw" / "seed000.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--max-panels", type=int, default=160)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run = load_small_run(args.data_file)
    out_dir = args.out_dir or PLOTS_ROOT / "small_10agents_50seeds" / split_dirname(
        run["split_ratio"]
    )
    s_a, s_b = _source_pairs(run)
    share = run["attention"].mean(
        axis=2
    )  # [n_iter, S], head mean = the share applied to the message
    ratio = share[:, s_a] / share[:, s_b]  # [n_iter, n_pairs]
    n_pairs = ratio.shape[1]
    final_dev = np.abs(ratio[-1] - 1.0)
    print(
        f"{n_pairs} edge pairs in {run['label']}; final ratio from {ratio[-1].min():.6f} to {ratio[-1].max():.6f}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ratio, color=BLUE, lw=0.7, alpha=0.4)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="equal shares")
    update_guides(ax, run["update_interval"], run["n_iter"])
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("share of half A / share of half B")
    ax.set_title(
        f"Edge-weight ratio of the two halves, all {n_pairs} pairs ({run['label']}, split "
        f"{run['split_ratio']:g}/{1 - run['split_ratio']:g}; gray = network update)",
        fontsize=9,
        loc="left",
    )
    plain_axes(ax)
    fig.tight_layout()
    save(fig, out_dir / "edge_pair_ratios_all.pdf")

    order = np.argsort(final_dev)
    if n_pairs > args.max_panels:
        # evenly spaced ranks keep both ends and the middle of the asymmetry range
        picks = order[np.linspace(0, n_pairs - 1, args.max_panels).round().astype(int)]
        note = f"{args.max_panels} of {n_pairs} pairs, evenly spaced by final asymmetry"
    else:
        picks = order
        note = f"all {n_pairs} pairs"
    cols = math.ceil(math.sqrt(len(picks) * 1.3))
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows), sharex=True)
    ylo, yhi = ratio[:, picks].min(), ratio[:, picks].max()
    pad = (yhi - ylo) * 0.05 or 1e-6
    for ax, j in zip(axes.ravel(), picks):
        ax.plot(ratio[:, j], color=BLUE, lw=1.0)
        ax.axhline(1.0, color="black", lw=0.6, ls="--")
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(_source_pair_name(run, s_a[j]), fontsize=6, loc="left")
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.ticklabel_format(useOffset=False, style="plain", axis="y")
    unused = axes.ravel()[len(picks) :]  # noqa: E203
    for ax in unused:
        ax.set_visible(False)
    fig.suptitle(
        f"Edge-weight ratio share(half A) / share(half B) over the run, {note} ({run['label']}, "
        "same y range everywhere, panels ordered from least to most asymmetric at the end)",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.supxlabel("iteration", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, out_dir / "edge_pair_ratios_grid.pdf")


if __name__ == "__main__":
    main()
