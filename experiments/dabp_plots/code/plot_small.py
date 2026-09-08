"""Figures for the small experiment: experiments/dabp_weights (10 agents x 50 seeds).

Reads the raw per-seed npz files of both splits (``data/`` = 0.5/0.5,
``data_asym/`` = 0.95/0.05) plus the structure CSVs written by
analyze_weights.py, and writes into ``small_10agents_50seeds/``:

- ``split_0.5_0.5/`` and ``split_0.95_0.05/``: damping_weights,
  split_halves_damping, split_halves_edge_weights, attention_weights,
  cost_and_convergence, structure
- ``compare_splits.pdf``: the two splits side by side on the same 50 problems

Example:
    uv run python experiments/dabp_plots/code/plot_small.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PLOTS_ROOT, SMALL_DATA, load_small_run, split_dirname  # noqa: E402
from figures import (  # noqa: E402
    fig_attention,
    fig_compare_splits_small,
    fig_cost_small,
    fig_damping_weights,
    fig_split_halves_damping,
    fig_split_halves_edge,
    fig_structure_small,
)

OUT_DIR = PLOTS_ROOT / "small_10agents_50seeds"
SOURCES = ("data", "data_asym")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    runs_by_split: dict[float, list[dict]] = {}
    for sub in SOURCES:
        data_dir = SMALL_DATA / sub
        paths = sorted((data_dir / "raw").glob("seed*.npz"))
        if not paths:
            # run_full.sh calls this after a single split has been recorded
            print(
                f"skipping {data_dir.relative_to(SMALL_DATA.parent.parent)}: no raw npz files"
            )
            continue
        runs = [load_small_run(p) for p in paths]
        split = runs[0]["split_ratio"]
        runs_by_split[split] = runs
        out = args.out_dir / split_dirname(split)
        fig_damping_weights(runs, out / "damping_weights.pdf")
        fig_split_halves_damping(runs, out / "split_halves_damping.pdf")
        fig_split_halves_edge(runs, out / "split_halves_edge_weights.pdf")
        fig_attention(runs, out / "attention_weights.pdf")
        fig_cost_small(runs, out / "cost_and_convergence.pdf")
        fig_structure_small(
            data_dir / "structure_correlation.csv",
            data_dir / "correlation_stats.csv",
            out / "structure.pdf",
        )
    if not runs_by_split:
        raise SystemExit(
            f"no raw npz files under {SMALL_DATA}/{{{','.join(SOURCES)}}}/raw"
        )
    if len(runs_by_split) == 2:
        fig_compare_splits_small(runs_by_split, args.out_dir / "compare_splits.pdf")


if __name__ == "__main__":
    main()
