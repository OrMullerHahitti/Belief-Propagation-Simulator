"""exp0: correctness checks for everything the other experiments rely on.

(a) the reduction identity: the 0.5/0.5 SCFG and the original graph with Q = cavity + belief
    produce the same beliefs (up to a per-variable constant) at every iteration, damped or not;
(b) the fast engine reproduces propflow's BPEngine / SplitEngine / DampingEngine /
    DampingSCFGEngine / DampedMidRunSplitEngine (transfer mode) assignment trajectories on a
    float-table instance;
(c) the fast engine reproduces the recorded AAAI cost curves (data_cuda) iteration for iteration
    on the exact benchmark instances. two propflow implementation details are needed for that:
    compute_R casts Q to the cost table's dtype, so on the integer random_dense / random_sparse
    tables the factors see trunc(Q) until the factors are split (float clones), and messages are
    normalised only every graph_diameter iterations. with damping the truncated value depends on
    the last bit of 0.9*old + 0.1*new, so damped-and-truncated runs (DMS before any split) can only
    be reproduced statistically;
(d) the truncation artifact itself: DMS with truncation (what the AAAI runs did) against DMS in
    floats (what the theory describes) over all 50 seeds.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lab import (  # noqa: E402
    REPO,
    FastEngine,
    aaai_inst,
    bipartite_inst,
    coloring_inst,
    fg_diameter,
    random_inst,
    run_record,
)

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DATA = REPO / "experiments" / "aaai" / "data_cuda"
T = 2000
lines: list[str] = []


def say(s: str) -> None:
    print(s, flush=True)
    lines.append(s)


# -- (a) reduction identity ---------------------------------------------------------------------
def check_reduction() -> None:
    say(
        "== (a) reduction identity: SCFG(0.5) vs Q = cavity + belief on the original graph"
    )
    cases = [
        ("random dense n=30", random_inst(30, 10, 0.6, seed=1)),
        ("random sparse n=50", random_inst(50, 10, 0.1, seed=2)),
        ("coloring n=40", coloring_inst(40, density=0.15, seed=3)),
        ("bipartite 20+20", bipartite_inst(20, 20, 10, 0.3, seed=4)),
    ]
    worst = 0.0
    for name, inst in cases:
        for lam in (0.0, 0.5, 0.9):
            a = FastEngine(inst, split=0.5, lam=lam)
            b = FastEngine(inst, rule="cav+belief", lam=lam)
            dmax, mism = 0.0, 0
            for _ in range(400):
                a.step()
                b.step()
                ba = a.beliefs()
                bb = b.beliefs()
                ba -= ba.min(axis=1, keepdims=True)
                bb -= bb.min(axis=1, keepdims=True)
                dmax = max(dmax, float(np.abs(ba - bb).max()))
                mism += int((a.assignment() != b.assignment()).sum())
            worst = max(worst, dmax)
            say(
                f"  {name:20s} lam={lam:.1f}  max |belief diff| = {dmax:.2e}  assignment mismatches = {mism}"
            )
    say(f"  -> worst belief difference over all cases: {worst:.2e}")


# -- (b) propflow cross-check -------------------------------------------------------------------
def build_pf(inst):
    from propflow import FGBuilder
    from propflow.bp.factor_graph import FactorGraph
    from propflow.core.agents import FactorAgent, VariableAgent

    vs = [VariableAgent(f"x{i + 1}", domain=inst.m) for i in range(inst.n)]
    fs, edges = [], {}
    for k, (i, j) in enumerate(inst.edges):
        f = FactorAgent.create_from_cost_table(
            f"f{i + 1}_{j + 1}", cost_table=inst.C[k].copy()
        )
        fs.append(f)
        edges[f] = [vs[i], vs[j]]
    fg = FactorGraph(vs, fs, edges)
    return FGBuilder.build_with_unary_costs(
        fg, {f"x{i + 1}": inst.theta[i] for i in range(inst.n)}
    )


def check_propflow() -> None:
    say(
        "== (b) propflow cross-check on a float-table instance (assignment agreement per iteration)"
    )
    from propflow.bp.engine_base import BPEngine
    from propflow.bp.engines import DampingEngine, DampingSCFGEngine, SplitEngine

    sys.path.insert(0, str(REPO / "experiments" / "aaai" / "code"))
    from engines import DampedMidRunSplitEngine

    def run_pf(engine, iters):
        out = []
        for t in range(iters):
            engine.step(t)
            # a per-message constant shift (no effect in exact arithmetic) that stops the level of
            # the messages from growing like (degree - 1)^t: without it an undamped propflow run
            # reaches 1e17 in about 20 iterations on this instance and its decoded assignment is
            # float noise. the AAAI runs normalise every graph_diameter iterations for the same reason
            engine.normalize_inbox()
            a = engine.assignments
            out.append([a[f"x{i + 1}"] for i in range(len(a))])
        return np.array(out)

    def compare(tag, inst, factory, fast_kwargs, iters=60, split_at=None):
        pf = run_pf(factory(build_pf(inst)), iters)
        fa = run_record(FastEngine(inst, **fast_kwargs), iters, split_at=split_at)[
            "assigns"
        ]
        agree = float((pf == fa).mean())
        say(f"  {tag:32s} agreement = {agree:.4f}")
        return agree

    inst = random_inst(10, 4, 0.5, seed=11, tiebreak=1e-2)
    agree = [
        compare(
            "MS",
            inst,
            lambda fg: BPEngine(factor_graph=fg, normalize_messages=False),
            dict(),
        ),
        compare(
            "MS + split",
            inst,
            lambda fg: SplitEngine(
                factor_graph=fg, split_factor=0.5, normalize_messages=False
            ),
            dict(split=0.5),
        ),
        compare(
            "DMS",
            inst,
            lambda fg: DampingEngine(
                factor_graph=fg, damping_factor=0.9, normalize_messages=False
            ),
            dict(lam=0.9),
        ),
        compare(
            "DMS + split",
            inst,
            lambda fg: DampingSCFGEngine(
                factor_graph=fg,
                split_factor=0.5,
                damping_factor=0.9,
                normalize_messages=False,
            ),
            dict(split=0.5, lam=0.9),
        ),
        compare(
            "DMS, split at 10 (transfer)",
            inst,
            lambda fg: DampedMidRunSplitEngine(
                factor_graph=fg,
                damping_factor=0.9,
                split_at_iter=10,
                split_factor=0.5,
                transfer_mode="transfer",
                normalize_messages=False,
            ),
            dict(lam=0.9),
            split_at=10,
        ),
    ]
    say(f"  -> minimum agreement: {min(agree):.4f}")


# -- (c) AAAI cost curves -----------------------------------------------------------------------
def _exact_kwargs(label: str, d: int) -> tuple[dict, int | None]:
    if label == "MS":
        return dict(floor_q=True, norm_every=d), None
    if label == "DMS":
        return dict(lam=0.9, floor_q=True, norm_every=d), None
    if label == "MS_split_0.5":
        return dict(split=0.5, norm_every=d), None
    if label == "DMS_split_0.5":
        return dict(split=0.5, lam=0.9, norm_every=d), None
    if label == "DMS_0.5_split_0.5":
        return dict(split=0.5, lam=0.5, norm_every=d), None
    if label.startswith("DMS_split_at_"):
        return dict(lam=0.9, floor_q=True, norm_every=d), int(label.rsplit("_", 1)[1])
    raise ValueError(label)


def check_csv() -> None:
    say(
        "== (c) recorded AAAI cost curves (data_cuda) vs the fast engine in exact-propflow mode"
    )
    labels = (
        "MS",
        "MS_split_0.5",
        "DMS_split_0.5",
        "DMS_split_at_100",
        "DMS_split_at_1000",
        "DMS",
    )
    for bench in (
        "random_dense",
        "random_sparse",
        "graph_coloring",
        "scale_free",
        "meeting_scheduling",
    ):
        df = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
        for seed in (0, 1):
            inst = aaai_inst(bench, seed)
            d = fg_diameter(inst)
            int_tables = bench in ("random_dense", "random_sparse")
            for label in labels:
                rec = (
                    df[(df.algorithm == label) & (df.seed == seed)]
                    .sort_values("iteration")
                    .cost.values
                )
                if len(rec) == 0:
                    continue
                kwargs, split_at = _exact_kwargs(label, d)
                if not int_tables:
                    kwargs.pop(
                        "floor_q", None
                    )  # float tables: no truncation in propflow
                got = run_record(
                    FastEngine(inst, **kwargs),
                    len(rec),
                    split_at=split_at,
                    record_assign=False,
                )["costs"]
                bad = np.flatnonzero(np.abs(got - rec) > 1e-3)
                say(
                    f"  {bench:18s} seed={seed} {label:17s} mismatching iterations={len(bad):4d}/{len(rec)}"
                    f"  first={'none' if len(bad) == 0 else int(bad[0]):>4}  final fast={got[-1]:.1f} csv={rec[-1]:.1f}"
                )


# -- (d) the truncation artifact ----------------------------------------------------------------
def _trunc_task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    d = fg_diameter(inst)
    out = {}
    for mode, kw in (
        ("float", dict(lam=0.9)),
        ("trunc", dict(lam=0.9, floor_q=True, norm_every=d)),
    ):
        r = run_record(FastEngine(inst, **kw), T, record_assign=False)
        out[mode] = (float(r["costs"][-1]), float(r["costs"].min()))
    return bench, seed, out


def check_truncation(pool) -> None:
    say(
        "== (d) DMS with propflow's integer truncation vs DMS in floats, 50 seeds, 2000 iterations"
    )
    for bench in ("random_dense", "random_sparse"):
        df = pd.read_csv(DATA / f"{bench}_raw_costs.csv")
        csv_final = (
            df[(df.algorithm == "DMS") & (df.iteration == T - 1)]
            .sort_values("seed")
            .cost.values
        )
        res = pool.map(_trunc_task, [(bench, s) for s in range(50)])
        fin = {m: np.array([r[2][m][0] for r in res]) for m in ("float", "trunc")}
        best = {m: np.array([r[2][m][1] for r in res]) for m in ("float", "trunc")}
        say(
            f"  {bench}: final cost  csv DMS {csv_final.mean():.0f} +- {csv_final.std():.0f}   "
            f"trunc {fin['trunc'].mean():.0f} +- {fin['trunc'].std():.0f}   float {fin['float'].mean():.0f} +- {fin['float'].std():.0f}"
        )
        say(
            f"  {bench}: best cost   trunc {best['trunc'].mean():.0f}   float {best['float'].mean():.0f}   "
            f"paired final diff trunc-float {np.mean(fin['trunc'] - fin['float']):+.0f} "
            f"(se {np.std(fin['trunc'] - fin['float'], ddof=1) / np.sqrt(50):.0f})"
        )
        pd.DataFrame(
            dict(
                seed=range(50),
                csv_final=csv_final,
                trunc_final=fin["trunc"],
                float_final=fin["float"],
                trunc_best=best["trunc"],
                float_best=best["float"],
            )
        ).to_csv(OUT / f"exp0_truncation_{bench}.csv", index=False)


if __name__ == "__main__":
    check_reduction()
    check_propflow()
    check_csv()
    with Pool() as pool:
        check_truncation(pool)
    (OUT / "exp0_checks.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT / 'exp0_checks.txt'}")
