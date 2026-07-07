"""Run the AAAI experiments: benchmarks x algorithms x N problem instances.

Algorithms (professor's list + plain min-sum baseline):
  baseline. MS                       normal undamped min-sum on original graph
  a. DMS                       damped min-sum, lambda = 0.9
  b. DMS_split_0.5             DMS on an SCFG, constant symmetric split (0.5/0.5)
  c. DMS_split_0.4_0.6         DMS on a random SCFG, per-entry split in [0.4, 0.6)
  d. DMS_split_at_{K}          DMS that splits all factors at iteration K,
                               K in {50, 100, 300, 500, 1000} (transfer mode)
  e. Attentive                 DABP (Deep Attentive Belief Propagation): a
                               graph-attention network trained per instance,
                               driven one BP iteration per step
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
from propflow.bp.engine_base import BPEngine
from propflow.bp.engines import DampingEngine, DampingSCFGEngine, SplitEngine

from csv_backups import backup_existing_csvs
from engines import (
    AttentiveEngine,
    AttentiveNoSplitEngine,
    AttentiveSymSplitEngine,
    CostOnlySnapshotManager,
    DampedMidRunSplitEngine,
    DampingRandomSplitEngine,
    run_full_horizon,
)
from merge import branch_and_bound, mgm1_binary_merge, score_assignment
from problems import BENCHMARKS, capture_original
from problems_ternary import TERNARY_BENCHMARKS

DAMPING = 0.9
# how many times to re-run tasks whose worker died (self-healing pool); the
# first attempt plus this many retries on progressively smaller pools.
MAX_PASSES = 4
SPLIT_AT_ITERS = (50, 100, 300, 500, 1000)
# opt-in split points, NOT part of the "all" expansion. run them only where
# requested explicitly (e.g. split@1500 on the dense benchmark via run_full.sh)
EXTRA_SPLIT_AT_ITERS = (1500,)

SPLIT_MS_LABEL = "MS_split_0.5"
MGM_LABEL = "MS_split_MGM_200"
OPT_MERGE_LABEL = "MS_split_opt_200"
OPTIMAL_LABEL = "Optimal"
PLAIN_MS_LABEL = "MS"
ATTENTIVE_LABEL = "Attentive"
ATTENTIVE_NOSPLIT_LABEL = "Attentive_NoSplit"
ATTENTIVE_SYMSPLIT_LABEL = "Attentive_SymSplit"
RANDOM_TERNARY_BENCHMARK = "random_ternary"
RANDOM_TERNARY_LABELS = {"DMS_split_0.5"}

# The parallel arity-3 suite (problems_ternary). Builders are looked up alongside
# the binary benchmarks, but the ternary names are kept OUT of the default "all"
# expansion so the binary suite and run_full.sh behave exactly as before; opt in
# with explicit names or `--benchmarks all_ternary`.
BENCHMARK_BUILDERS = {**BENCHMARKS, **TERNARY_BENCHMARKS}
TERNARY_SUITE = set(TERNARY_BENCHMARKS)


def _common_kwargs() -> dict:
    return {
        "computator": MinSumComputator(),
        "normalize_messages": True,
        "anytime": False,
        "snapshot_manager": CostOnlySnapshotManager(),
    }


def make_engine(label: str, fg, seed: int):
    if label == PLAIN_MS_LABEL:
        return BPEngine(factor_graph=fg, **_common_kwargs())
    if label == "DMS":
        return DampingEngine(
            factor_graph=fg, damping_factor=DAMPING, **_common_kwargs()
        )
    if label == "DMS_split_0.5":
        return DampingSCFGEngine(
            factor_graph=fg,
            damping_factor=DAMPING,
            split_factor=0.5,
            **_common_kwargs(),
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
    if label == ATTENTIVE_LABEL:
        return AttentiveEngine(factor_graph=fg, **_common_kwargs())
    if label == ATTENTIVE_NOSPLIT_LABEL:
        return AttentiveNoSplitEngine(factor_graph=fg, **_common_kwargs())
    if label == ATTENTIVE_SYMSPLIT_LABEL:
        return AttentiveSymSplitEngine(factor_graph=fg, **_common_kwargs())
    if label == SPLIT_MS_LABEL:
        return SplitEngine(factor_graph=fg, split_factor=0.5, **_common_kwargs())
    raise ValueError(f"unknown engine label: {label}")


# "Attentive" (item 2e) is DABP, wired via AttentiveEngine in engines.py. It needs
# the optional 'dabp' extra (torch + torch-geometric); the other families do not.
ENGINE_LABELS = (
    [PLAIN_MS_LABEL, "DMS", "DMS_split_0.5", "DMS_split_0.4_0.6"]
    + [f"DMS_split_at_{k}" for k in SPLIT_AT_ITERS]
    + [ATTENTIVE_LABEL, ATTENTIVE_NOSPLIT_LABEL, ATTENTIVE_SYMSPLIT_LABEL]
)
# extra engine columns that build a normal task but are excluded from "all"
EXTRA_ENGINE_LABELS = [f"DMS_split_at_{k}" for k in EXTRA_SPLIT_AT_ITERS]
# what "--algorithms all" expands to (unchanged: no opt-in extras)
ALL_LABELS = ENGINE_LABELS + [SPLIT_MS_LABEL, MGM_LABEL, OPT_MERGE_LABEL, OPTIMAL_LABEL]
# everything a user may name explicitly via --algorithms (for validation)
KNOWN_LABELS = ALL_LABELS + EXTRA_ENGINE_LABELS

# DABP (Attentive) supports only unary/binary factors, so the ternary suite runs
# the full AAAI family minus the two DABP variants. Optimal is kept in the set
# (it self-limits via the time cap and is only meaningfully attempted on the
# low-domain ternary benchmarks; see run_full_ternary.sh).
DABP_LABELS = {ATTENTIVE_LABEL, ATTENTIVE_NOSPLIT_LABEL, ATTENTIVE_SYMSPLIT_LABEL}
# what "--algorithms all" expands to for a ternary benchmark (mirrors binary
# "all": ALL_LABELS, i.e. no opt-in extras), minus DABP.
TERNARY_SUPPORTED_LABELS = set(ALL_LABELS) - DABP_LABELS
# what a ternary benchmark accepts when labels are named explicitly: every known
# label (including opt-in extras like DMS_split_at_1500) except DABP, matching the
# binary benchmarks which apply no gating to explicitly-named labels.
TERNARY_KNOWN_LABELS = set(KNOWN_LABELS) - DABP_LABELS


def supported_labels_for(benchmark: str) -> set[str]:
    """Algorithm labels a benchmark accepts (for SKIP/error messages)."""
    if benchmark == RANDOM_TERNARY_BENCHMARK:
        return set(RANDOM_TERNARY_LABELS)
    if benchmark in TERNARY_SUITE:
        return set(TERNARY_KNOWN_LABELS)
    return set(KNOWN_LABELS)


def labels_for_benchmark(
    benchmark: str, requested: set[str], *, all_requested: bool
) -> tuple[set[str], set[str]]:
    """Resolve benchmark-specific algorithm support.

    The binary-suite ``random_ternary`` is a targeted high-arity DMS+split
    experiment (only ``DMS_split_0.5``). The dedicated ternary suite
    (``problems_ternary``) instead runs the full AAAI family minus DABP. Every
    other benchmark keeps the existing label behavior.
    """
    if benchmark == RANDOM_TERNARY_BENCHMARK:
        if all_requested:
            return set(RANDOM_TERNARY_LABELS), set()
        resolved = requested & RANDOM_TERNARY_LABELS
        return resolved, requested - resolved
    if benchmark in TERNARY_SUITE:
        if all_requested:
            return set(TERNARY_SUPPORTED_LABELS), set()
        # explicit labels: allow any known label (incl. opt-in extras) except DABP
        resolved = requested & TERNARY_KNOWN_LABELS
        return resolved, requested - resolved
    return set(requested), set()


def run_engine_task(benchmark: str, seed: int, label: str, max_iter: int) -> list[dict]:
    fg = BENCHMARK_BUILDERS[benchmark](seed)
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
    fg = BENCHMARK_BUILDERS[benchmark](seed)
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
        menus = {v: sorted({int(branch1[v]), int(branch2[v])}) for v in var_names}
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
    fg = BENCHMARK_BUILDERS[benchmark](seed)
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
        for label in ENGINE_LABELS + EXTRA_ENGINE_LABELS:
            if label in labels:
                tasks.append(
                    (
                        "engine",
                        benchmark,
                        seed,
                        {"label": label, "max_iter": args.max_iter},
                    )
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


def _write_metadata(
    out_dir: Path, benchmark: str, args, labels: set[str], elapsed: float
) -> None:
    """write (or, in --append mode, union into) the per-benchmark metadata.

    Appending keeps the existing run parameters and only extends the recorded
    ``algorithms`` list with the newly appended labels, so a metadata file is
    never silently clobbered when adding a column to a completed benchmark.
    """
    meta_path = out_dir / f"{benchmark}_metadata.json"
    base: dict = {
        "benchmark": benchmark,
        "n_problems": args.n_problems,
        "seed_start": args.seed_start,
        "max_iter": args.max_iter,
        "merge_at": args.merge_at,
        "damping": DAMPING,
        "split_at_iters": list(SPLIT_AT_ITERS),
        "opt_time_limit_s": args.opt_time_limit,
        "algorithms": sorted(labels),
        "elapsed_s": round(elapsed, 1),
    }
    if args.append and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        merged = dict(existing)
        merged["algorithms"] = sorted(set(existing.get("algorithms", [])) | labels)
        appends = list(existing.get("appends", []))
        appends.append({"algorithms": sorted(labels), "elapsed_s": round(elapsed, 1)})
        merged["appends"] = appends
        meta_path.write_text(json.dumps(merged, indent=2))
    else:
        meta_path.write_text(json.dumps(base, indent=2))


def run_benchmark(benchmark: str, args, labels: set[str]) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{benchmark}_final_costs.csv"
    raw_path = out_dir / f"{benchmark}_raw_costs.csv"

    # append rows to existing CSVs (adding an algorithm column to a completed
    # benchmark) only when both files already exist; otherwise fall back to a
    # fresh write so a first run still produces headers.
    appending = args.append and final_path.exists() and raw_path.exists()
    mode = "a" if appending else "w"

    tasks = build_tasks(benchmark, args, labels)
    total = len(tasks)
    print(
        f"START {benchmark}: {total} tasks on {args.jobs} workers "
        f"(mode={'append' if appending else 'write'})",
        flush=True,
    )
    started = time.time()

    with (
        final_path.open(mode, newline="") as final_handle,
        raw_path.open(mode, newline="") as raw_handle,
    ):
        final_writer = csv.writer(final_handle)
        raw_writer = csv.writer(raw_handle)
        if not appending:
            final_writer.writerow(["algorithm", "seed", "final_cost", "anytime_cost"])
            raw_writer.writerow(["algorithm", "seed", "iteration", "cost"])

        def _write_rows(rows: list[dict]) -> None:
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

        # Self-healing execution: run the tasks in passes, each on a FRESH pool.
        # A worker that dies (OOM / native crash) surfaces as a per-future
        # exception -- with no max_tasks_per_child the executor cleanly marks the
        # pool broken instead of deadlocking on a recycle -- so the offending
        # tasks are collected and retried on a smaller pool. Only completed tasks
        # write rows, so retries never duplicate output.
        pending = list(tasks)
        done = 0
        for attempt in range(MAX_PASSES):
            if not pending:
                break
            jobs = args.jobs if attempt == 0 else max(1, args.jobs // 2)
            if attempt > 0:
                print(
                    f"  {benchmark}: retry pass {attempt} for {len(pending)} "
                    f"failed task(s) on {jobs} workers",
                    flush=True,
                )
            failed: list[tuple] = []
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_task, task): task for task in pending}
                for future in as_completed(futures):
                    kind, bench, seed, _ = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"FAILED {bench} seed={seed} kind={kind} "
                            f"(pass {attempt}): {exc!r}",
                            flush=True,
                        )
                        failed.append(futures[future])
                        continue
                    _write_rows(rows)
                    done += 1
                    if done % 10 == 0 or done == total:
                        elapsed = time.time() - started
                        print(
                            f"  {benchmark}: {done}/{total} tasks "
                            f"({elapsed / 60:.1f} min)",
                            flush=True,
                        )
            pending = failed

        if pending:
            print(
                f"WARNING {benchmark}: {len(pending)} task(s) still failing after "
                f"{MAX_PASSES} passes; their rows are missing",
                flush=True,
            )

    elapsed = time.time() - started
    _write_metadata(out_dir, benchmark, args, labels, elapsed)
    print(f"DONE {benchmark} in {elapsed / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="benchmark names, 'all' (binary suite) or 'all_ternary' (arity-3 suite)",
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=["all"], help="algorithm labels or 'all'"
    )
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--merge-at", type=int, default=200)
    parser.add_argument("--opt-time-limit", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument(
        "--append",
        action="store_true",
        help="append rows to existing {benchmark}_{final,raw}_costs.csv instead "
        "of overwriting them (for adding an algorithm column to a completed "
        "benchmark); metadata's algorithm list is unioned, not replaced",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="do not copy existing CSVs to experiments/aaai/backups before writing",
    )
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    args = parser.parse_args()

    if args.benchmarks == ["all"]:
        benchmarks = list(BENCHMARKS)
    elif args.benchmarks == ["all_ternary"]:
        benchmarks = list(TERNARY_BENCHMARKS)
    else:
        benchmarks = args.benchmarks
    unknown = set(benchmarks) - set(BENCHMARK_BUILDERS)
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)}")

    all_requested = args.algorithms == ["all"]
    labels = set(ALL_LABELS) if all_requested else set(args.algorithms)
    unknown = labels - set(KNOWN_LABELS)
    if unknown:
        raise SystemExit(
            f"unknown algorithms: {sorted(unknown)}; known: {KNOWN_LABELS}"
        )

    if args.merge_at < 2 or args.merge_at > args.max_iter:
        raise SystemExit("--merge-at must be in [2, --max-iter]")

    if not args.skip_backup:
        backup_dir = backup_existing_csvs(Path(args.out_dir), label="data_before_run")
        if backup_dir is not None:
            print(f"BACKUP existing CSVs -> {backup_dir}", flush=True)

    for benchmark in benchmarks:
        benchmark_labels, skipped = labels_for_benchmark(
            benchmark, labels, all_requested=all_requested
        )
        supported = supported_labels_for(benchmark)
        if skipped:
            print(
                f"SKIP {benchmark}: unsupported algorithm(s) {sorted(skipped)}; "
                f"supported: {sorted(supported)}",
                flush=True,
            )
        if not benchmark_labels:
            raise SystemExit(
                f"no supported algorithms requested for {benchmark}; "
                f"supported: {sorted(supported)}"
            )
        run_benchmark(benchmark, args, benchmark_labels)


if __name__ == "__main__":
    main()
