"""the DMS-BDS line for the cost figures, drawn as in the earlier late-split study.

protocol (split_at_best_window/, run_split_at_best.py --window W --after A): DMS runs for W library
iterations, the best state seen in that window is restored at iteration W and split there, and the
run continues for A more iterations. the split run itself started the split at t* + 1, so its
post-split costs are shifted to start at W; before W the line is the DMS run of the paper folder.

output: experiments/aamas/section6/best_point_curve/<bench>_raw_costs.csv with algorithm
DMS_split_at_best and iterations 0 .. W + A - 1 (library units), for plot_final.py --extra-raw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "aaai" / "data_paper_20260928"
RUNS = HERE / "split_at_best_window"
OUT = HERE / "best_point_curve"
BENCHES = [
    "random_sparse",
    "random_dense",
    "scale_free",
    "graph_coloring",
    "meeting_scheduling",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--after", type=int, default=1000)
    args = parser.parse_args()
    W, A = args.window, args.after
    OUT.mkdir(exist_ok=True)
    for bench in BENCHES:
        dms = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
        dms = dms[dms.algorithm == "DMS"].pivot(
            index="seed", columns="iteration", values="cost"
        )
        fin = pd.read_csv(RUNS / f"{bench}_final_costs.csv").set_index("seed")
        raw = pd.read_csv(RUNS / f"{bench}_raw_costs.csv").pivot(
            index="seed", columns="iteration", values="cost"
        )
        rows = []
        for seed in fin.index:
            split_iter = int(fin.loc[seed, "split_iter"])
            assert split_iter <= W, (bench, seed, split_iter)
            post = raw.loc[seed].values[split_iter : split_iter + A]
            assert len(post) == A and not np.isnan(post).any(), (bench, seed, len(post))
            curve = np.concatenate([dms.loc[seed].values[:W], post])
            rows.append(
                pd.DataFrame(
                    {
                        "algorithm": "DMS_split_at_best",
                        "seed": seed,
                        "iteration": np.arange(W + A),
                        "cost": curve,
                    }
                )
            )
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(OUT / f"{bench}_raw_costs.csv", index=False)
        at_w = out[out.iteration == W - 1].cost.mean()
        restored = out[out.iteration == W].cost.mean()
        print(
            f"{bench:19} DMS at {2 * W} paper it.: {at_w:,.2f}  restored best (split point): {fin.cost_at_split.mean():,.2f}  first post-split cost {restored:,.2f}  end {out[out.iteration == W + A - 1].cost.mean():,.2f}"
        )


if __name__ == "__main__":
    main()
