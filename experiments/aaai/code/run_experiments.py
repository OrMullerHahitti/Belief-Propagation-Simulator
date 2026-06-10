"""Run the AAAI experiments: 5 benchmarks x algorithms x N problem instances.

Algorithms (professor's list):
  a. DMS                       damped min-sum, lambda = 0.9
  b. DMS_split_0.5             DMS on an SCFG, constant symmetric split (0.5/0.5)
  c. DMS_split_0.4_0.6         DMS on a random SCFG, per-entry split in [0.4, 0.6)
  d. DMS_split_at_{K}          DMS that splits all factors at iteration K,
                               K in {50, 100, 300, 500, 1000} (transfer mode)
  e. Attentive                 min-sum with degree-inverse inbox discounting
  f. Optimal                   branch and bound (reported only when it completes
                               within the time limit)
  g. MS_split_0.5              undamped min-sum on an SCFG (0.5/0.5)
  h. MS_split_MGM_200          the two assignments at iterations 198/199 of (g)
                               merged with MGM-1 restricted to the binary menu
  i. MS_split_opt_200          same two assignments merged optimally (branch and
                               bound over the binary menu)

g, h and i share a single engine run per instance: the "two options after 200
iterations" are the assignments at the last two iterations before the merge
point of the split-only run (its period-2 oscillation branches).

Outputs per benchmark (in --out-dir):
  {benchmark}_final_costs.csv   algorithm, seed, final_cost, anytime_cost
  {benchmark}_raw_costs.csv     algorithm, seed, iteration, cost
  {benchmark}_metadata.json     run parameters

Example:
  uv run python experiments/aaai/code/run_experiments.py --benchmarks all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from propflow.bp.computators import MinSumComputator
from propflow.bp.engines import DampingEngine, DampingSCFGEngine, SplitEngine

from engines import (
    AttentiveEngine,
    CostOnlySnapshotManager,
    DampedMidRunSplitEngine,
    DampingRandomSplitEngine,
    run_full_horizon,
)
from merge import branch_and_bound, mgm1_binary_merge, score_assignment
from problems import BENCHMARKS, capture_original

DAMPING = 0.9
SPLIT_AT_ITERS = (50, 100, 300, 500, 1000)

SPLIT_MS_LABEL = "MS_split_0.5"
MGM_LABEL = "MS_split_MGM_200"
OPT_MERGE_LABEL = "MS_split_opt_200"
OPTIMAL_LABEL = "Optimal"


def _common_kwargs() -> dict:
    return {
        "computator": MinSumComputator(),
        "normalize_messages": True,
        "anytime": False,
        "snapshot_manager": CostOnlySnapshotManager(),
    }


def make_engine(label: str, fg, seed: int):
    if label == "DMS":
        return DampingEngine(factor_graph=fg, damping_factor=DAMPING, **_common_kwargs())
    if label == "DMS_split_0.5":
        return DampingSCFGEngine(
            factor_graph=fg, damping_factor=DAMPING, split_factor=0.5, **_common_kwargs()
        )
    if label == "DMS_split_0.4_0.6":
        return DampingRandomSplitEngine(
            factor_graph=fg,
            damping_factor=DAMPING,
            split_low=0.4,
            split_high=0.6,
            split_seed=seed,
            **_common_kwargs(),
        )
    if label.startswith("DMS_split_at_"):
        split_at = int(label.rsplit("_", 1)[1])
        return DampedMidRunSplitEngine(
            factor_graph=fg,
            damping_factor=DAMPING,
            split_at_iter=split_at,
            split_factor=0.5,
            transfer_mode="transfer",
            **_common_kwargs(),
        )
    if label == "Attentive":
        return AttentiveEngine(factor_graph=fg, **_common_kwargs())
    if label == SPLIT_MS_LABEL:
        return SplitEngine(factor_graph=fg, split_factor=0.5, **_common_kwargs())
    raise ValueError(f"unknown engine label: {label}")


# note: "Attentive" (item 2e) is left out for now — the intended algorithm is
# not yet defined (the repo's discount_attentive policy was rejected). the
# AttentiveEngine scaffold remains in engines.py; once defined, add the label
# back here and wire it in make_engine.
ENGINE_LABELS = (
    ["DMS", "DMS_split_0.5", "DMS_split_0.4_0.6"]
    + [f"DMS_split_at_{k}" for k in SPLIT_AT_ITERS]
)
ALL_LABELS = ENGINE_LABELS + [SPLIT_MS_LABEL, MGM_LABEL, OPT_MERGE_LABEL, OPTIMAL_LABEL]


def run_engine_task(benchmark: str, seed: int, label: str, max_iter: int) -> list[dict]:
    fg = BENCHMARKS[benchmark](seed)
    engine = make_engine(label, fg, seed)
    costs = run_full_horizon(engine, max_iter)
    return [
        {
            "algorithm": label,
            "seed": seed,
            "final_cost": costs[-1],
            "anytime_cost": min(costs),
            "costs": costs,
        }
    ]


def run_split_ms_task(
    benchmark: str, seed: int, max_iter: int, merge_at: int, wanted: set[str]
) -> list[dict]:
    """one split-only min-sum run serving MS_split_0.5 and both merge variants."""
    fg = BENCHMARKS[benchmark](seed)
    var_names, factor_vars, tables = capture_original(fg)

    engine = make_engine(SPLIT_MS_LABEL, fg, seed)
    engine.convergence_monitor.reset()
    branch_iters = (merge_at - 2, merge_at - 1)
    branches: dict[int, dict[str, int]] = {}
    for i in range(max_iter):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        if i in branch_iters:
            branches[i] = {k: int(v) for k, v in engine.assignments.items()}
    costs = [float(engine._snapshots[i].global_cost) for i in range(max_iter)]

    rows = []
    if SPLIT_MS_LABEL in wanted:
        rows.append(
            {
                "algorithm": SPLIT_MS_LABEL,
                "seed": seed,
                "final_cost": costs[-1],
                "anytime_cost": min(costs),
                "costs": costs,
            }
        )

    branch1, branch2 = branches[branch_iters[0]], branches[branch_iters[1]]
    pre_merge = costs[:merge_at]
    score1 = score_assignment(branch1, tables, factor_vars)
    score2 = score_assignment(branch2, tables, factor_vars)

    best_merge: tuple[dict, float] | None = None
    if MGM_LABEL in wanted or OPT_MERGE_LABEL in wanted:
        merged_a, _, _ = mgm1_binary_merge(
            branch1, branch2, "branch1", var_names, factor_vars, tables
        )
        merged_b, _, _ = mgm1_binary_merge(
            branch1, branch2, "branch2", var_names, factor_vars, tables
        )
        candidates = [
            (merged_a, score_assignment(merged_a, tables, factor_vars)),
            (merged_b, score_assignment(merged_b, tables, factor_vars)),
        ]
        best_merge = min(candidates, key=lambda c: c[1])

    if MGM_LABEL in wanted:
        merged_cost = best_merge[1]
        rows.append(
            {
                "algorithm": MGM_LABEL,
                "seed": seed,
                "final_cost": merged_cost,
                "anytime_cost": min(min(pre_merge), merged_cost),
                "costs": pre_merge + [merged_cost],
            }
        )

    if OPT_MERGE_LABEL in wanted:
        menus = {
            v: sorted({int(branch1[v]), int(branch2[v])}) for v in var_names
        }
        # condition every table on the menus (np.ix_ keeps singleton axes) so
        # the bounds are tight and only disagreement variables actually branch;
        # the unconditioned tables make branch and bound blow up on the
        # random-cost benchmarks
        red_tables = {
            fname: tables[fname][np.ix_(*[menus[v] for v in vs])]
            for fname, vs in factor_vars.items()
        }
        red_domains = {v: list(range(len(menus[v]))) for v in var_names}
        warm_assign, warm_cost = (
            (branch1, score1) if score1 <= score2 else (branch2, score2)
        )
        if best_merge is not None and best_merge[1] < warm_cost:
            warm_assign, warm_cost = best_merge
        warm_pos = {v: menus[v].index(int(warm_assign[v])) for v in var_names}
        opt_cost, _, stats = branch_and_bound(
            var_names,
            factor_vars,
            red_tables,
            red_domains,
            initial_upper_bound=warm_cost,
            initial_assignment=warm_pos,
            time_limit_s=300.0,
        )
        if not stats["complete"]:
            print(
                f"WARNING: optimal merge hit its time limit "
                f"({benchmark} seed={seed}); reporting best bound found",
                flush=True,
            )
        rows.append(
            {
                "algorithm": OPT_MERGE_LABEL,
                "seed": seed,
                "final_cost": opt_cost,
                "anytime_cost": min(min(pre_merge), opt_cost),
                "costs": pre_merge + [opt_cost],
            }
        )

    return rows


def run_optimal_task(benchmark: str, seed: int, time_limit_s: float) -> list[dict]:
    fg = BENCHMARKS[benchmark](seed)
    var_names, factor_vars, tables = capture_original(fg)
    domains = {v.name: list(range(v.domain)) for v in fg.variables}
    cost, _, stats = branch_and_bound(
        var_names, factor_vars, tables, domains, time_limit_s=time_limit_s
    )
    final = cost if stats["complete"] else float("nan")
    return [
        {
            "algorithm": OPTIMAL_LABEL,
            "seed": seed,
            "final_cost": final,
            "anytime_cost": final,
            "costs": [],
        }
    ]


def run_task(task: tuple) -> list[dict]:
    kind, benchmark, seed, payload = task
    if kind == "engine":
        return run_engine_task(benchmark, seed, payload["label"], payload["max_iter"])
    if kind == "split_ms":
        return run_split_ms_task(
            benchmark, seed, payload["max_iter"], payload["merge_at"], payload["wanted"]
        )
    if kind == "optimal":
        return run_optimal_task(benchmark, seed, payload["time_limit_s"])
    raise ValueError(f"unknown task kind: {kind}")


def build_tasks(benchmark: str, args, labels: set[str]) -> list[tuple]:
    tasks = []
    for seed in range(args.seed_start, args.seed_start + args.n_problems):
        for label in ENGINE_LABELS:
            if label in labels:
                tasks.append(
                    ("engine", benchmark, seed, {"label": label, "max_iter": args.max_iter})
                )
        wanted = labels & {SPLIT_MS_LABEL, MGM_LABEL, OPT_MERGE_LABEL}
        if wanted:
            tasks.append(
                (
                    "split_ms",
                    benchmark,
                    seed,
                    {
                        "max_iter": args.max_iter,
                        "merge_at": args.merge_at,
                        "wanted": wanted,
                    },
                )
            )
        if OPTIMAL_LABEL in labels:
            tasks.append(
                ("optimal", benchmark, seed, {"time_limit_s": args.opt_time_limit})
            )
    return tasks


def run_benchmark(benchmark: str, args, labels: set[str]) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{benchmark}_final_costs.csv"
    raw_path = out_dir / f"{benchmark}_raw_costs.csv"

    tasks = build_tasks(benchmark, args, labels)
    print(f"START {benchmark}: {len(tasks)} tasks on {args.jobs} workers", flush=True)
    started = time.time()

    with final_path.open("w", newline="") as final_handle, raw_path.open(
        "w", newline=""
    ) as raw_handle:
        final_writer = csv.writer(final_handle)
        final_writer.writerow(["algorithm", "seed", "final_cost", "anytime_cost"])
        raw_writer = csv.writer(raw_handle)
        raw_writer.writerow(["algorithm", "seed", "iteration", "cost"])

        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                kind, bench, seed, _ = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED {bench} seed={seed} kind={kind}: {exc!r}", flush=True)
                    rows = []
                for row in rows:
                    final_writer.writerow(
                        [
                            row["algorithm"],
                            row["seed"],
                            f"{row['final_cost']:.6f}",
                            f"{row['anytime_cost']:.6f}",
                        ]
                    )
                    for it, cost in enumerate(row["costs"]):
                        raw_writer.writerow(
                            [row["algorithm"], row["seed"], it, f"{cost:.4f}"]
                        )
                final_handle.flush()
                raw_handle.flush()
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    elapsed = time.time() - started
                    print(
                        f"  {benchmark}: {done}/{len(tasks)} tasks "
                        f"({elapsed / 60:.1f} min)",
                        flush=True,
                    )

    metadata = {
        "benchmark": benchmark,
        "n_problems": args.n_problems,
        "seed_start": args.seed_start,
        "max_iter": args.max_iter,
        "merge_at": args.merge_at,
        "damping": DAMPING,
        "split_at_iters": list(SPLIT_AT_ITERS),
        "opt_time_limit_s": args.opt_time_limit,
        "algorithms": sorted(labels),
        "elapsed_s": round(time.time() - started, 1),
    }
    (out_dir / f"{benchmark}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"DONE {benchmark} in {metadata['elapsed_s'] / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["all"], help="benchmark names or 'all'")
    parser.add_argument("--algorithms", nargs="+", default=["all"], help="algorithm labels or 'all'")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--merge-at", type=int, default=200)
    parser.add_argument("--opt-time-limit", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    args = parser.parse_args()

    benchmarks = (
        list(BENCHMARKS) if args.benchmarks == ["all"] else args.benchmarks
    )
    unknown = set(benchmarks) - set(BENCHMARKS)
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)}")

    labels = set(ALL_LABELS) if args.algorithms == ["all"] else set(args.algorithms)
    unknown = labels - set(ALL_LABELS)
    if unknown:
        raise SystemExit(f"unknown algorithms: {sorted(unknown)}; known: {ALL_LABELS}")

    if args.merge_at < 2 or args.merge_at > args.max_iter:
        raise SystemExit("--merge-at must be in [2, --max-iter]")

    for benchmark in benchmarks:
        run_benchmark(benchmark, args, labels)


if __name__ == "__main__":
    main()
