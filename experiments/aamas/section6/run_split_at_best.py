"""split at the best point: DMS until its best-cost iteration, then the 0.5 split, damping kept.

a demonstration line, not a heuristic: it shows what the split does to the best solution DMS finds.
for every benchmark and seed, t* = the first iteration at which the DMS run of the paper folder
reaches its minimum cost. the delayed-split engine repeats the DMS run exactly until the split
(checked: DMS_split_at_K equals DMS before K), so splitting at K = t* + 1 starts from the state
that produced the best cost. every run then gets the same AFTER iterations after the split.

outputs (experiments/aamas/section6/split_at_best/):
  <bench>_final_costs.csv  algorithm, seed, t_star, split_iter, cost_at_split, final_cost, best_after_split, changed
  <bench>_raw_costs.csv    algorithm, seed, iteration, cost   (iterations 0 .. split_iter + AFTER - 1)

usage: uv run python experiments/aamas/section6/run_split_at_best.py [--benchmarks ...] [--seeds 50] [--jobs N] [--after 1000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "aaai" / "data_paper_20260928"
OUT = HERE / "split_at_best"
OUT_FIXED = HERE / "split_at_best_fixed_horizon"
OUT_WINDOW = HERE / "split_at_best_window"
BENCHES = [
    "random_sparse",
    "random_dense",
    "scale_free",
    "graph_coloring",
    "meeting_scheduling",
]
LABEL = "DMS_split_at_best"

# the paper folder's DMS lines on random sparse/dense are float-table runs; the prefix must match them
os.environ["AAAI_FLOAT_TABLES"] = "1"
sys.path.insert(0, str(ROOT / "aaai" / "code"))
from run_experiments import run_engine_task  # noqa: E402


def t_star(bench: str, window: int = 0) -> dict[int, int]:
    """first iteration of the lowest DMS cost, searched in the first `window` library iterations (0 = all)."""
    raw = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
    dms = raw[raw.algorithm == "DMS"].pivot(
        index="seed", columns="iteration", values="cost"
    )
    vals = dms.values[:, :window] if window else dms.values
    return {int(seed): int(np.argmin(row)) for seed, row in zip(dms.index, vals)}


def task(args):
    bench, seed, ts, after, horizon = args
    split_iter = ts + 1
    # fixed horizon: every run ends at the same iteration as the paper's other lines
    max_iter = horizon if horizon else split_iter + after
    (rec,) = run_engine_task(bench, seed, f"DMS_split_at_{split_iter}", max_iter)
    costs = np.asarray(rec["costs"], dtype=float)
    return bench, seed, ts, split_iter, costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHES)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument(
        "--after", type=int, default=1000, help="library iterations run after the split"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="run every seed to this library iteration instead of split + AFTER (output in split_at_best_fixed_horizon/)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="search the best DMS iteration in the first W library iterations only (the earlier study's protocol: W = 1000, then AFTER iterations after the split; output in split_at_best_window/)",
    )
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    args = parser.parse_args()
    out = OUT_WINDOW if args.window else (OUT_FIXED if args.horizon else OUT)
    out.mkdir(exist_ok=True)

    for bench in args.benchmarks:
        t0 = time.time()
        ts = t_star(bench, args.window)
        dms_best = {}
        raw = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
        dms = raw[raw.algorithm == "DMS"]
        for seed in range(args.seeds):
            d = dms[dms.seed == seed].sort_values("iteration").cost.values
            dms_best[seed] = float((d[: args.window] if args.window else d).min())
        jobs = [(bench, seed, ts[seed], args.after, args.horizon) for seed in range(args.seeds)]
        print(f"START {bench}: {len(jobs)} runs on {args.jobs} workers", flush=True)
        finals, raws = [], []
        with Pool(args.jobs) as pool:
            for i, (b, seed, t, split_iter, costs) in enumerate(
                pool.imap_unordered(task, jobs), 1
            ):
                cost_at_split = float(costs[split_iter - 1])
                # the last pre-split cost must be the DMS best of the paper folder (same prefix;
                # the csv holds costs rounded to 4 decimals)
                prefix_ok = abs(cost_at_split - dms_best[seed]) < 1e-3
                finals.append(
                    dict(
                        algorithm=LABEL,
                        seed=seed,
                        t_star=t,
                        split_iter=split_iter,
                        cost_at_split=cost_at_split,
                        final_cost=float(costs[-1]),
                        best_after_split=float(costs[split_iter:].min()),
                        changed=bool(abs(costs[-1] - cost_at_split) > 1e-6),
                        prefix_matches_dms=prefix_ok,
                    )
                )
                raws.append(
                    pd.DataFrame(
                        {
                            "algorithm": LABEL,
                            "seed": seed,
                            "iteration": np.arange(len(costs)),
                            "cost": costs,
                        }
                    )
                )
                if i % 10 == 0:
                    print(
                        f"  {bench}: {i}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                        flush=True,
                    )
        fin = pd.DataFrame(finals).sort_values("seed")
        fin.to_csv(out / f"{bench}_final_costs.csv", index=False)
        pd.concat(raws).sort_values(["seed", "iteration"]).to_csv(
            out / f"{bench}_raw_costs.csv", index=False
        )
        print(
            f"DONE {bench} in {(time.time() - t0) / 60:.1f} min: prefix matches DMS on {fin.prefix_matches_dms.sum()}/{len(fin)}, "
            f"final changed on {fin.changed.sum()}/{len(fin)}, mean at split {fin.cost_at_split.mean():,.2f}, "
            f"mean final {fin.final_cost.mean():,.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
