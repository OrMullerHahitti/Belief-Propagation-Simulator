"""Measure the wall-clock cost of the two "choose between 2 domains" merges.

The split-only min-sum run (``MS_split_0.5``) oscillates in a period-2 cycle; the
assignments at iterations 198/199 are its two branches. ``MS_split_MGM_200`` and
``MS_split_opt_200`` merge those two branches over the per-variable binary menu
``{branch1[v], branch2[v]}`` -- MGM-1 local search and branch-and-bound
respectively (see ``run_experiments.run_split_ms_task`` and ``merge.py``). Both
run *once* at the merge point, so their wall-clock is not reflected on the
iteration axis.

This script times those two merges for seed 0 of each binary-menu benchmark and
writes ``data/merge_timing.csv`` with, per benchmark, the merge seconds and the
per-iteration *stretch ratio* ``T_merge / DMS_per_iter`` -- the same DMS baseline
``time_dabp.py`` uses, so DABP and merge stretches share one wall-clock x-axis.
``plot_results.py`` reads it and draws the merged-cost point at ``x = merge_at +
ratio`` (holding the pre-merge cost flat until then). B&B on ``random_dense``
cannot prove optimality (high induced width) and hits its time cap, so its ratio
is huge and lands off the plotted horizon -- recorded honestly via ``bnb_complete``.

Example:
  uv run python experiments/aaai/code/time_merges.py --dms-from experiments/aaai/data_cuda/dabp_timing.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from csv_backups import backup_existing_csvs
from merge import branch_and_bound, mgm1_binary_merge, score_assignment
from problems import BENCHMARKS, capture_original
from run_experiments import (
    RANDOM_TERNARY_BENCHMARK,
    SPLIT_MS_LABEL,
    make_engine,
)

# DMS-baseline fallback (only if --dms-from has no row for a benchmark)
N_WARMUP = 5
N_TIMED = 40
# MGM is fast + deterministic, so average a few repeats for a stable number.
MGM_REPEATS = 5
# B&B is deterministic; repeat only the cheap (already-complete, <2s) solves.
BNB_REPEAT_THRESHOLD_S = 2.0
BNB_EXTRA_REPEATS = 2
# matches the hardcoded merge cap in run_experiments.run_split_ms_task
BNB_TIME_LIMIT_S = 300.0
MERGE_AT = 200


def _time_steps(engine, n_warmup: int, n_timed: int) -> float:
    """per-iteration wall-clock of engine.step (+ cycle events), after warm-up.

    Mirrors time_dabp._time_steps so the fallback DMS baseline is measured
    identically.
    """
    engine.convergence_monitor.reset()
    i = 0
    for _ in range(n_warmup):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        i += 1
    start = time.perf_counter()
    for _ in range(n_timed):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        i += 1
    return (time.perf_counter() - start) / n_timed


def _dms_per_iter(benchmark: str, seed: int, dms_lookup: dict) -> float:
    """DMS per-iteration baseline: reuse the DABP-timing value if available
    (keeps DABP and merge stretches on the same denominator), else measure."""
    if benchmark in dms_lookup:
        return dms_lookup[benchmark]
    return _time_steps(make_engine("DMS", BENCHMARKS[benchmark](seed), seed), N_WARMUP, N_TIMED)


def _branches(benchmark: str, seed: int):
    """run MS_split_0.5 to the merge point and return the two oscillation
    branches plus the original (pre-split) tables -- exactly as
    run_experiments.run_split_ms_task does."""
    fg = BENCHMARKS[benchmark](seed)
    var_names, factor_vars, tables = capture_original(fg)
    engine = make_engine(SPLIT_MS_LABEL, fg, seed)
    engine.convergence_monitor.reset()
    branch_iters = (MERGE_AT - 2, MERGE_AT - 1)
    branches: dict[int, dict[str, int]] = {}
    for i in range(MERGE_AT):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        if i in branch_iters:
            branches[i] = {k: int(v) for k, v in engine.assignments.items()}
    return (
        branches[branch_iters[0]],
        branches[branch_iters[1]],
        var_names,
        factor_vars,
        tables,
    )


def _time_mgm(branch1, branch2, var_names, factor_vars, tables):
    """seconds for the MGM-1 binary-menu merge (both start branches, best kept),
    averaged over MGM_REPEATS; also returns the best merged assignment+cost."""
    best_assign, best_cost = None, float("inf")
    start = time.perf_counter()
    for _ in range(MGM_REPEATS):
        merged_a, _, _ = mgm1_binary_merge(
            branch1, branch2, "branch1", var_names, factor_vars, tables
        )
        merged_b, _, _ = mgm1_binary_merge(
            branch1, branch2, "branch2", var_names, factor_vars, tables
        )
        ca = score_assignment(merged_a, tables, factor_vars)
        cb = score_assignment(merged_b, tables, factor_vars)
        best_assign, best_cost = (merged_a, ca) if ca <= cb else (merged_b, cb)
    secs = (time.perf_counter() - start) / MGM_REPEATS
    return secs, best_assign, best_cost


def _time_bnb(
    branch1, branch2, var_names, factor_vars, tables, mgm_assign, mgm_cost
):
    """seconds for the branch-and-bound binary-menu merge, warm-started exactly
    like run_split_ms_task. Returns (seconds, opt_cost, complete)."""
    score1 = score_assignment(branch1, tables, factor_vars)
    score2 = score_assignment(branch2, tables, factor_vars)
    menus = {v: sorted({int(branch1[v]), int(branch2[v])}) for v in var_names}
    red_tables = {
        fname: tables[fname][np.ix_(*[menus[v] for v in vs])]
        for fname, vs in factor_vars.items()
    }
    red_domains = {v: list(range(len(menus[v]))) for v in var_names}
    warm_assign, warm_cost = (
        (branch1, score1) if score1 <= score2 else (branch2, score2)
    )
    if mgm_assign is not None and mgm_cost < warm_cost:
        warm_assign, warm_cost = mgm_assign, mgm_cost
    warm_pos = {v: menus[v].index(int(warm_assign[v])) for v in var_names}

    def run_once():
        return branch_and_bound(
            var_names,
            factor_vars,
            red_tables,
            red_domains,
            initial_upper_bound=warm_cost,
            initial_assignment=warm_pos,
            time_limit_s=BNB_TIME_LIMIT_S,
        )

    start = time.perf_counter()
    opt_cost, _, stats = run_once()
    secs = time.perf_counter() - start
    # repeat only cheap, already-complete solves for a stabler number
    if stats["complete"] and secs < BNB_REPEAT_THRESHOLD_S:
        start = time.perf_counter()
        for _ in range(BNB_EXTRA_REPEATS):
            opt_cost, _, stats = run_once()
        secs = (secs + (time.perf_counter() - start)) / (1 + BNB_EXTRA_REPEATS)
    return secs, opt_cost, bool(stats["complete"])


def time_benchmark(benchmark: str, seed: int, dms_lookup: dict) -> dict:
    dms = _dms_per_iter(benchmark, seed, dms_lookup)
    branch1, branch2, var_names, factor_vars, tables = _branches(benchmark, seed)
    mgm_s, mgm_assign, mgm_cost = _time_mgm(
        branch1, branch2, var_names, factor_vars, tables
    )
    bnb_s, bnb_cost, bnb_complete = _time_bnb(
        branch1, branch2, var_names, factor_vars, tables, mgm_assign, mgm_cost
    )
    return {
        "benchmark": benchmark,
        "dms_s_per_iter": dms,
        "mgm_s": mgm_s,
        "bnb_s": bnb_s,
        "mgm_ratio": mgm_s / dms if dms > 0 else float("nan"),
        "bnb_ratio": bnb_s / dms if dms > 0 else float("nan"),
        "bnb_complete": bnb_complete,
    }


def _load_dms(path: Path) -> dict:
    """benchmark -> dms_s_per_iter from a dabp_timing.csv (for a shared baseline)."""
    if not path or not path.exists():
        return {}
    lookup = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                lookup[row["benchmark"]] = float(row["dms_s_per_iter"])
            except (KeyError, TypeError, ValueError):
                continue
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    parser.add_argument(
        "--dms-from",
        default="",
        help="dabp_timing.csv to read dms_s_per_iter from (shared DMS baseline); "
        "missing benchmarks are measured fresh",
    )
    parser.add_argument("--skip-backup", action="store_true")
    args = parser.parse_args()

    benchmarks = list(BENCHMARKS) if args.benchmarks == ["all"] else args.benchmarks
    unknown = set(benchmarks) - set(BENCHMARKS)
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)}")
    # only benchmarks with a binary-menu merge (random_ternary runs only DMS_split)
    skipped = [b for b in benchmarks if b == RANDOM_TERNARY_BENCHMARK]
    for b in skipped:
        print(f"skipping {b}: no binary-menu merge variants", flush=True)
    benchmarks = [b for b in benchmarks if b not in skipped]
    if not benchmarks:
        print("no merge-supported benchmarks requested; nothing to time", flush=True)
        return

    dms_lookup = _load_dms(Path(args.dms_from)) if args.dms_from else {}
    if dms_lookup:
        print(f"using DMS baseline from {args.dms_from}", flush=True)

    rows = []
    for benchmark in benchmarks:
        print(f"timing merges for {benchmark} (seed {args.seed}) ...", flush=True)
        row = time_benchmark(benchmark, args.seed, dms_lookup)
        rows.append(row)
        cap = "" if row["bnb_complete"] else " [TIME-CAPPED, not optimal]"
        print(
            f"  DMS {row['dms_s_per_iter'] * 1e3:.3f} ms/iter | "
            f"MGM {row['mgm_s'] * 1e3:.2f} ms (ratio {row['mgm_ratio']:.2f}) | "
            f"B&B {row['bnb_s'] * 1e3:.2f} ms (ratio {row['bnb_ratio']:.2f}){cap}",
            flush=True,
        )

    out_path = Path(args.out_dir) / "merge_timing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_backup:
        backup_dir = backup_existing_csvs(out_path.parent, label="data_before_merge_timing")
        if backup_dir is not None:
            print(f"BACKUP existing CSVs -> {backup_dir}", flush=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["benchmark", "dms_s_per_iter", "mgm_s", "bnb_s", "mgm_ratio", "bnb_ratio", "bnb_complete"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["benchmark"],
                    f"{row['dms_s_per_iter']:.6f}",
                    f"{row['mgm_s']:.6f}",
                    f"{row['bnb_s']:.6f}",
                    f"{row['mgm_ratio']:.4f}",
                    f"{row['bnb_ratio']:.4f}",
                    int(row["bnb_complete"]),
                ]
            )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
