"""check: are the two MS-SCFG assignments handed to MGM the ones whose costs the paper plots?

run_split_ms_task (experiments/aaai/code/run_experiments.py) reads the two branches of MS-SCFG-MGM and
MS-SCFG-opt at library steps 198 and 199 (paper iterations 398 and 400) after the cycle events. the
assignment is recomputed from the inbox on every read, and the message normalization in the cycle
events can flip an argmin when the undamped messages are large. this reruns the same MS-SCFG
(SplitEngine 0.5, no damping, the paper's default tables) for 200 library steps per instance, reads
both steps before and after the cycle events, and runs the paper's MGM (MGM-1 from each branch, the
better kept) on both pairs.

checks: the rerun's costs equal the paper's MS-SCFG line, and MGM on the after-pair reproduces the
paper's MS-SCFG-MGM value.

output (experiments/aamas/section6/ms_scfg_read_check/): <bench>.csv, one row per instance

usage: uv run python experiments/aamas/section6/check_ms_scfg_read.py [--benchmarks ...] [--jobs N]
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "aaai" / "code"))
from merge import mgm1_binary_merge, score_assignment  # noqa: E402
from problems import capture_original  # noqa: E402
from run_experiments import BENCHMARK_BUILDERS, MGM_LABEL, SPLIT_MS_LABEL, make_engine  # noqa: E402

# its MS-SCFG and MS-SCFG-MGM rows are identical to data_paper_20261002 (which only replaced DABP lines),
# and it exists on rtx as well
DATA = ROOT / "aaai" / "data_paper_20260928"
OUT = HERE / "ms_scfg_read_check"
BENCHES = [
    "random_sparse",
    "random_dense",
    "scale_free",
    "graph_coloring",
    "meeting_scheduling",
]
# the paper's merge point: branches at library steps 198 and 199
MERGE_AT = 200
BRANCHES = (MERGE_AT - 2, MERGE_AT - 1)


def mgm_cost(branch1, branch2, var_names, factor_vars, tables) -> float:
    """the paper's MGM: MGM-1 restricted to the two-value menu, from each branch, the better kept."""
    merged = [
        mgm1_binary_merge(branch1, branch2, start, var_names, factor_vars, tables)[0]
        for start in ("branch1", "branch2")
    ]
    return min(score_assignment(m, tables, factor_vars) for m in merged)


def task(args):
    bench, seed = args
    fg = BENCHMARK_BUILDERS[bench](seed)
    var_names, factor_vars, tables = capture_original(fg)
    engine = make_engine(SPLIT_MS_LABEL, fg, seed)
    engine.convergence_monitor.reset()
    before, after = {}, {}
    for i in range(MERGE_AT):
        engine.step(i)
        if i in BRANCHES:
            before[i] = {k: int(v) for k, v in engine.assignments.items()}
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        if i in BRANCHES:
            after[i] = {k: int(v) for k, v in engine.assignments.items()}
    costs = np.array([float(engine._snapshots[i].global_cost) for i in range(MERGE_AT)])
    b0, b1 = BRANCHES
    row = dict(seed=seed, diameter=int(engine.graph_diameter))
    for i in BRANCHES:
        row[f"cost_{i}"] = costs[i]
        row[f"before_{i}_gives_cost"] = (
            abs(score_assignment(before[i], tables, factor_vars) - costs[i]) < 1e-6
        )
        row[f"after_{i}_gives_cost"] = (
            abs(score_assignment(after[i], tables, factor_vars) - costs[i]) < 1e-6
        )
        row[f"vars_changed_by_read_{i}"] = sum(
            before[i][v] != after[i][v] for v in var_names
        )
    row["branches_differ_before"] = sum(
        before[b0][v] != before[b1][v] for v in var_names
    )
    row["branches_differ_after"] = sum(after[b0][v] != after[b1][v] for v in var_names)
    row["mgm_before"] = mgm_cost(before[b0], before[b1], var_names, factor_vars, tables)
    row["mgm_after"] = mgm_cost(after[b0], after[b1], var_names, factor_vars, tables)
    return bench, row, costs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmarks", nargs="+", default=BENCHES)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for bench in args.benchmarks:
        with Pool(args.jobs) as pool:
            results = pool.map(task, [(bench, seed) for seed in range(args.seeds)])
        df = (
            pd.DataFrame([row for _, row, _ in results])
            .sort_values("seed")
            .set_index("seed")
        )
        costs = {row["seed"]: c for _, row, c in results}

        # the rerun must be the paper's run: same MS-SCFG costs, same MGM value from the after-pair
        raw = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
        ms = raw[(raw.algorithm == SPLIT_MS_LABEL) & (raw.iteration < MERGE_AT)].pivot(
            index="seed", columns="iteration", values="cost"
        )
        paper_mgm = pd.read_csv(DATA / f"{bench}_final_costs.csv").pivot(
            index="seed", columns="algorithm", values="final_cost"
        )[MGM_LABEL]
        df["prefix_equals_paper"] = [
            np.allclose(costs[s], ms.loc[s].to_numpy(), rtol=0, atol=1e-3)
            for s in df.index
        ]
        df["mgm_paper"] = paper_mgm.reindex(df.index)
        df.to_csv(OUT / f"{bench}.csv")

        read_changed = (df.vars_changed_by_read_198 > 0) | (
            df.vars_changed_by_read_199 > 0
        )
        mgm_changed = (df.mgm_before - df.mgm_after).abs() > 1e-6
        print(
            f"{bench}: rerun equals the paper's MS-SCFG on {int(df.prefix_equals_paper.sum())}/{len(df)}, "
            f"MGM on the after-pair equals the paper's MGM on {int(((df.mgm_after - df.mgm_paper).abs() < 1e-3).sum())}/{len(df)}; "
            f"read after normalization changed a branch on {int(read_changed.sum())}/{len(df)} "
            f"(diameters {sorted(set(df.diameter[read_changed]))}), MGM changes on {int(mgm_changed.sum())}; "
            f"mean MGM paper {df.mgm_paper.mean():,.4f} vs correct read {df.mgm_before.mean():,.4f}; "
            f"variables differing between the two branches, median: after {df.branches_differ_after.median():.1f}, "
            f"before {df.branches_differ_before.median():.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
