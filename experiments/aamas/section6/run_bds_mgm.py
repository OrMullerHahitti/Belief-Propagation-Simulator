"""DMS-k*DS-MGM: the split point of DMS-k*DS, no damping after the split, then MGM on two assignments.

every benchmark and seed splits at the same iteration as DMS-k*DS: K = t* + 1, where t* is the first
iteration of the lowest DMS cost within the first WINDOW library iterations of the paper folder's DMS
line (WINDOW = 1000 library iterations = the first 2000 paper iterations; nothing after the window is
looked at). the engine repeats DMS exactly until K, splits every factor 0.5/0.5 (transfer mode), turns
damping off and runs undamped min-sum on the split graph up to the common horizon.

phases (outputs in --out, default experiments/aamas/section6/split_at_best_mgm/):
  run     BP only. <bench>_bp.npz: costs [seed, iteration] and assignments [seed, iteration, variable]
          of every iteration, with t*, K and the variable order. a benchmark whose npz exists is skipped
  settle  per instance, the first step after the split (0 = the split step) from which every assignment
          equals the one two steps earlier up to the horizon: one fixed assignment or two alternating
          ones. settle.csv, and the distribution per benchmark in paper iterations
  merge   MGM-1 on the two assignments of steps K+N-2 and K+N-1, started from each of them, the better
          result kept (the MGM of MS-SCFG-MGM). --n N gives every instance N undamped steps, --n settle
          gives each instance its own settle step (--n-unsettled N for instances that never settle).
          <bench>_final_costs.csv and <bench>_raw_costs.csv (BP costs to step K+N-1, the MGM cost at K+N)

usage:
  uv run python experiments/aamas/section6/run_bds_mgm.py run [--benchmarks ...] [--seeds 50] [--jobs N]
  uv run python experiments/aamas/section6/run_bds_mgm.py settle
  uv run python experiments/aamas/section6/run_bds_mgm.py merge --n settle [--n-unsettled N]
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
sys.path.insert(0, str(HERE))
# the DMS-k*DS runner sets the float-table switch and the aaai code path; its t* gives both lines the same K
from run_split_at_best import BENCHES, DATA, t_star  # noqa: E402
from run_experiments import BENCHMARK_BUILDERS, DAMPING, _common_kwargs  # noqa: E402
from merge import mgm1_binary_merge, score_assignment  # noqa: E402
from problems import capture_original  # noqa: E402

from experiments.aamas.late_split.core import ReleasedDampingSplitEngine  # noqa: E402

OUT = HERE / "split_at_best_mgm"
LABEL = "DMS_split_at_best_MGM"
# library iterations searched for t* (the first 2000 paper iterations)
WINDOW = 1000
# library iterations per run (the paper's 4000 iterations)
HORIZON = 2000
# library steps of period <= 2 required at the end of the run to call an instance settled
TAIL = 100


def bp_task(args):
    """DMS up to K - 1, the split at K, undamped min-sum on the split graph up to the horizon."""
    bench, seed, split_iter = args
    fg = BENCHMARK_BUILDERS[bench](seed)
    var_names, _, _ = capture_original(fg)
    engine = ReleasedDampingSplitEngine(
        factor_graph=fg,
        damping_factor=DAMPING,
        split_at_iter=split_iter,
        split_factor=0.5,
        transfer_mode="transfer",
        **_common_kwargs(),
    )
    engine.convergence_monitor.reset()
    assignments = np.empty((HORIZON, len(var_names)), dtype=np.int8)
    for i in range(HORIZON):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        current = engine.assignments
        assignments[i] = [current[v] for v in var_names]
    costs = np.array([float(engine._snapshots[i].global_cost) for i in range(HORIZON)])
    return bench, seed, costs, assignments, var_names


def run_phase(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for bench in args.benchmarks:
        path = out / f"{bench}_bp.npz"
        if path.exists():
            print(f"SKIP {bench}: {path} exists", flush=True)
            continue
        t0 = time.time()
        ts = t_star(bench, WINDOW)
        seeds = list(range(args.seeds))
        jobs = [(bench, seed, ts[seed] + 1) for seed in seeds]
        print(f"START {bench}: {len(jobs)} runs on {args.jobs} workers", flush=True)
        results = {}
        with Pool(args.jobs) as pool:
            for i, (_, seed, costs, assignments, var_names) in enumerate(
                pool.imap_unordered(bp_task, jobs), 1
            ):
                results[seed] = (costs, assignments, var_names)
                if i % 10 == 0:
                    print(
                        f"  {bench}: {i}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                        flush=True,
                    )
        var_names = results[seeds[0]][2]
        if any(results[seed][2] != var_names for seed in seeds):
            raise RuntimeError(f"{bench}: variable order differs between seeds")
        costs = np.stack([results[seed][0] for seed in seeds])
        assignments = np.stack([results[seed][1] for seed in seeds])
        split = np.array([ts[seed] + 1 for seed in seeds])

        # before the split the run is DMS: it must equal the paper folder's DMS line (csv rounded to 4 decimals)
        raw = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
        dms = raw[raw.algorithm == "DMS"].pivot(
            index="seed", columns="iteration", values="cost"
        )
        prefix_ok = sum(
            np.allclose(
                costs[k, : split[k]],
                dms.loc[seed].to_numpy()[: split[k]],
                rtol=0,
                atol=1e-3,
            )
            for k, seed in enumerate(seeds)
        )
        np.savez_compressed(
            path,
            seeds=np.array(seeds),
            t_star=split - 1,
            split_iter=split,
            costs=costs,
            assignments=assignments,
            var_names=np.array(var_names),
        )
        print(
            f"DONE {bench} in {(time.time() - t0) / 60:.1f} min: prefix equals DMS on {prefix_ok}/{len(seeds)}, "
            f"median split at library {np.median(split):.0f}",
            flush=True,
        )


def settle_step(post: np.ndarray) -> int | None:
    """first step after the split from which every assignment equals the one two steps earlier, up to
    the horizon; None when the last TAIL steps are not of period <= 2."""
    # same[j]: the assignment of step j + 2 repeats that of step j
    same = np.all(post[2:] == post[:-2], axis=1)
    if not same[-TAIL:].all():
        return None
    broken = np.flatnonzero(~same)
    return int(broken[-1]) + 3 if broken.size else 2


def settle_phase(args) -> None:
    out = Path(args.out)
    rows = []
    for bench in args.benchmarks:
        z = np.load(out / f"{bench}_bp.npz")
        for k, seed in enumerate(z["seeds"]):
            split_iter = int(z["split_iter"][k])
            post = z["assignments"][k, split_iter:]
            s = settle_step(post)
            if s is None:
                kind = "none"
            else:
                kind = "fixed" if np.array_equal(post[s - 2], post[s - 1]) else "two"
            rows.append(
                dict(
                    benchmark=bench,
                    seed=int(seed),
                    t_star=split_iter - 1,
                    split_iter=split_iter,
                    settle_steps=s,
                    kind=kind,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "settle.csv", index=False)
    print(
        "undamped iterations needed after the split (paper iterations = 2 x library steps)"
    )
    for bench, g in df.groupby("benchmark", sort=False):
        settled = g[g.kind != "none"]
        paper = 2 * settled.settle_steps.astype(int)
        spread = (
            f"median {paper.median():.0f}, 90% {paper.quantile(0.9):.0f}, max {paper.max()}"
            if len(settled)
            else "none settled"
        )
        print(
            f"  {bench:19s} settled {len(settled)}/{len(g)} (one fixed assignment {(g.kind == 'fixed').sum()}, "
            f"two alternating {(g.kind == 'two').sum()}): {spread}"
        )


def merge_phase(args) -> None:
    out = Path(args.out)
    settle = pd.read_csv(out / "settle.csv") if args.n == "settle" else None
    for bench in args.benchmarks:
        z = np.load(out / f"{bench}_bp.npz")
        var_names = [str(v) for v in z["var_names"]]
        rows, raws = [], []
        for k, seed in enumerate(z["seeds"]):
            seed = int(seed)
            split_iter = int(z["split_iter"][k])
            if settle is None:
                n = int(args.n)
            else:
                s = settle[
                    (settle.benchmark == bench) & (settle.seed == seed)
                ].settle_steps.iloc[0]
                if pd.notna(s):
                    n = int(s)
                elif args.n_unsettled is not None:
                    n = args.n_unsettled
                else:
                    raise SystemExit(
                        f"{bench} seed {seed} never settles; pass --n-unsettled"
                    )
            if split_iter + n >= HORIZON:
                raise SystemExit(
                    f"{bench} seed {seed}: the MGM result would fall after the horizon"
                )

            names, factor_vars, tables = capture_original(
                BENCHMARK_BUILDERS[bench](seed)
            )
            if names != var_names:
                raise RuntimeError(
                    f"{bench} seed {seed}: variable order differs from the run"
                )
            costs = z["costs"][k]
            steps = (split_iter + n - 2, split_iter + n - 1)
            branch1, branch2 = (
                dict(zip(var_names, map(int, z["assignments"][k, step])))
                for step in steps
            )
            # the stored assignments must reproduce the recorded costs on the original tables
            for branch, step in zip((branch1, branch2), steps):
                if (
                    abs(score_assignment(branch, tables, factor_vars) - costs[step])
                    > 1e-6
                ):
                    raise RuntimeError(
                        f"{bench} seed {seed}: assignment of step {step} does not give its cost"
                    )
            merged = [
                mgm1_binary_merge(
                    branch1, branch2, start, var_names, factor_vars, tables
                )[0]
                for start in ("branch1", "branch2")
            ]
            mgm_cost = min(score_assignment(m, tables, factor_vars) for m in merged)
            line = np.concatenate([costs[: split_iter + n], [mgm_cost]])
            rows.append(
                dict(
                    algorithm=LABEL,
                    seed=seed,
                    t_star=split_iter - 1,
                    split_iter=split_iter,
                    undamped_steps=n,
                    dms_best=float(costs[split_iter - 1]),
                    branch1_cost=float(costs[steps[0]]),
                    branch2_cost=float(costs[steps[1]]),
                    final_cost=float(mgm_cost),
                )
            )
            raws.append(
                pd.DataFrame(
                    {
                        "algorithm": LABEL,
                        "seed": seed,
                        "iteration": np.arange(len(line)),
                        "cost": line,
                    }
                )
            )
        fin = pd.DataFrame(rows)
        fin.to_csv(out / f"{bench}_final_costs.csv", index=False)
        pd.concat(raws).to_csv(out / f"{bench}_raw_costs.csv", index=False)
        better_branch = np.minimum(fin.branch1_cost, fin.branch2_cost)
        print(
            f"{bench:19s} mean DMS best {fin.dms_best.mean():,.2f} -> MGM {fin.final_cost.mean():,.2f}; "
            f"MGM below the better of its two assignments on {(fin.final_cost < better_branch - 1e-9).sum()}/{len(fin)}; "
            f"mean undamped iterations {2 * fin.undamped_steps.mean():.0f} (paper)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=["run", "settle", "merge"])
    parser.add_argument("--benchmarks", nargs="+", default=BENCHES)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument(
        "--n",
        default="settle",
        help="undamped library steps after the split before MGM: an integer, or 'settle'",
    )
    parser.add_argument(
        "--n-unsettled",
        type=int,
        default=None,
        help="with --n settle: steps for instances that never settle",
    )
    args = parser.parse_args()
    {"run": run_phase, "settle": settle_phase, "merge": merge_phase}[args.phase](args)


if __name__ == "__main__":
    main()
