"""Figures for the bigger experiment: experiments/dabp_node_dynamics (seed-0 graph).

One random graph (edge probability 0.5, domain 10, graph seed 0) at 20 and at
50 nodes, each run once per split with identical initial network parameters
and every iteration retained. Reads ``outputs/{20,50}nodes_seed0/
{symmetric,asymmetric}.npz`` and writes into ``bigger_seed0/{20,50}nodes/``:

- ``split_0.5_0.5/`` and ``split_0.95_0.05/``: damping_weights,
  split_halves_damping, split_halves_edge_weights, attention_weights,
  cost_and_convergence
- ``compare_splits.pdf``: both splits on the same graph

Example:
    uv run python experiments/dabp_plots/code/plot_bigger.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BIGGER_DATA, PLOTS_ROOT, load_bigger_run, split_dirname  # noqa: E402
from figures import (  # noqa: E402
    fig_attention,
    fig_compare_splits_single,
    fig_cost_single,
    fig_damping_weights,
    fig_split_halves_damping,
    fig_split_halves_edge,
)

OUT_DIR = PLOTS_ROOT / "bigger_seed0"
NODE_RUNS = (20, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--nodes", type=int, nargs="+", default=list(NODE_RUNS))
    args = parser.parse_args()

    for nodes in args.nodes:
        src = BIGGER_DATA / f"{nodes}nodes_seed0"
        if not src.is_dir():
            raise SystemExit(f"missing run directory {src}")
        runs_by_split: dict[float, dict] = {}
        for variant in ("symmetric", "asymmetric"):
            run = load_bigger_run(src / f"{variant}.npz")
            out = args.out_dir / f"{nodes}nodes" / split_dirname(run["split_ratio"])
            fig_damping_weights([run], out / "damping_weights.pdf")
            fig_split_halves_damping([run], out / "split_halves_damping.pdf")
            fig_split_halves_edge([run], out / "split_halves_edge_weights.pdf")
            fig_attention([run], out / "attention_weights.pdf")
            fig_cost_single(run, out / "cost_and_convergence.pdf")
            # the 50-node attention array is ~1 GB and the comparison never reads it
            del run["attention"]
            runs_by_split[run["split_ratio"]] = run
        fig_compare_splits_single(
            runs_by_split, args.out_dir / f"{nodes}nodes" / "compare_splits.pdf"
        )


if __name__ == "__main__":
    main()
