"""exp3: the two-solution (period-2) end state and how it depends on density.

random instances n=50, domain 10 (the AAAI random family) at densities 0.05 .. 1.0, 20 seeds,
MS + split (undamped) and DMS + split (0.9), 1000 iterations. for every run: the period of the
decoded assignment, and for a period-2 run the two layers x, y with
  - the flip fraction (variables that alternate),
  - edge classes by the number of alternating endpoints (0 / 1 / 2) with the mean edge cost of
    each class in layer x, against the mean edge cost of a 1-opt repaired assignment,
  - whether each layer is a best response to the other (layer-wise minimality of cost_2),
  - improving single moves inside each layer and the gain of greedy 1-opt from the layer,
  - after commitment: does the decoded assignment follow synchronous best response, and is
    cost_2(x_t, x_{t+1}) non-increasing.
the same on bipartite instances, where the two layers can be re-phased into two ordinary
assignments (left from x, right from y and vice versa).
outputs: results/exp3_random.csv, results/exp3_bipartite.csv, results/exp3_summary.md, plots/exp3_*.pdf
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
    FastEngine,
    bipartite_inst,
    detect_period,
    edge_flip_classes,
    freeze_time,
    greedy_1opt,
    improving_single_moves,
    random_inst,
    run_record,
    sync_best_response,
)
from plotting import LABELS, RESULTS, STYLE, new_fig, save  # noqa: E402

T = 1000
SEEDS = 20
DENSITIES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
ALGS = {"MS_split": dict(split=0.5), "DMS_split": dict(split=0.5, lam=0.9)}


def analyse(inst, r, rng):
    a = r["assigns"]
    sats = r["sats"]
    period = detect_period(a[-400:], pmax=64, window=200)
    x, y = a[-2], a[-1]
    row = dict(
        period=period,
        freeze=freeze_time(r["changes"]),
        final_commit=float(sats[-1]),
        cost_x=inst.cost(x),
        cost_y=inst.cost(y),
        cost2_half=inst.pair_cost(x, y) / 2,
        flip=float((x != y).mean()),
    )
    # edge classes and their mean cost in layer x
    cls = edge_flip_classes(inst, x, y)
    ec = inst.edge_costs(x)
    for c in (0, 1, 2):
        m = cls == c
        row[f"edges_class{c}"] = float(m.mean())
        row[f"cost_class{c}"] = float(ec[m].mean()) if m.any() else np.nan
    # layer-wise minimality: each layer a best response to the other
    row["x_is_br_of_y"] = float((sync_best_response(inst, y, iters=1) == x).mean())
    row["y_is_br_of_x"] = float((sync_best_response(inst, x, iters=1) == y).mean())
    row["single_moves_x"] = improving_single_moves(inst, x)
    row["single_moves_y"] = improving_single_moves(inst, y)
    gx, _ = greedy_1opt(inst, x, rng)
    row["greedy_from_x"] = inst.cost(gx)
    row["cost_greedy_class2"] = (
        float(inst.edge_costs(gx)[cls == 2].mean()) if (cls == 2).any() else np.nan
    )
    # after commitment: synchronous best response and monotone cost_2
    committed = np.flatnonzero(sats >= 0.99)
    if len(committed) > 5:
        t0 = int(committed[0])
        br_ok = [
            (sync_best_response(inst, a[t], iters=1) == a[t + 1]).all()
            for t in range(t0, T - 1)
        ]
        c2 = [inst.pair_cost(a[t], a[t + 1]) for t in range(t0, T - 1)]
        row["t_commit99"] = t0
        row["br_steps"] = float(np.mean(br_ok))
        row["cost2_increases"] = int(np.sum(np.diff(c2) > 1e-6))
    else:
        row["t_commit99"] = T
        row["br_steps"] = np.nan
        row["cost2_increases"] = -1
    return row


def task(args):
    family, density, seed = args
    rng = np.random.default_rng(1000 + seed)
    if family == "random":
        inst = random_inst(50, 10, density, seed=seed)
    else:
        inst = bipartite_inst(25, 25, 10, density, seed=seed)
    rows = []
    for alg, kw in ALGS.items():
        r = run_record(FastEngine(inst, **kw), T)
        row = analyse(inst, r, rng)
        row.update(family=family, density=density, seed=seed, alg=alg, edges=inst.E)
        if family == "bipartite":
            x, y = r["assigns"][-2], r["assigns"][-1]
            left = np.arange(inst.n) < 25
            z1 = np.where(left, x, y)
            z2 = np.where(left, y, x)
            row["cost_rephased_min"] = min(inst.cost(z1), inst.cost(z2))
            row["rephased_single_moves"] = min(
                improving_single_moves(inst, z1), improving_single_moves(inst, z2)
            )
        rows.append(row)
    return rows


def run_all() -> None:
    RESULTS.mkdir(exist_ok=True)
    with Pool() as pool:
        for family in ("random", "bipartite"):
            jobs = [(family, d, s) for d in DENSITIES for s in range(SEEDS)]
            rows = [row for rows in pool.map(task, jobs) for row in rows]
            pd.DataFrame(rows).to_csv(RESULTS / f"exp3_{family}.csv", index=False)
            print(f"{family}: done", flush=True)


def summarize() -> None:
    lines = [
        "# exp3: period-2 end states vs density (n=50, domain 10, 20 seeds, 1000 iterations)",
        "",
    ]
    for family in ("random", "bipartite"):
        df = pd.read_csv(RESULTS / f"exp3_{family}.csv")
        lines += [f"## {family} instances", ""]
        for alg in ALGS:
            d = df[df.alg == alg]
            lines += [
                f"### {LABELS[alg]}",
                "",
                "| density | edges | period 1 / 2 / other | flip fraction | class-2 edges | cost class 0 / 1 / 2 | 1-opt cost class 2 | "
                "layer cost | greedy from layer | improving single moves | BR steps | cost_2 increases | "
                + (
                    "rephased cost | rephased single moves |"
                    if family == "bipartite"
                    else ""
                ),
                "|---|---|---|---|---|---|---|---|---|---|---|---|"
                + ("---|---|" if family == "bipartite" else ""),
            ]
            for dens, g in d.groupby("density"):
                p2 = g[g.period == 2]
                extra = ""
                if family == "bipartite":
                    extra = (
                        f" {p2.cost_rephased_min.mean():.0f} | {p2.rephased_single_moves.mean():.1f} |"
                        if len(p2)
                        else " - | - |"
                    )
                lines.append(
                    f"| {dens} | {g.edges.mean():.0f} | {(g.period == 1).sum()} / {(g.period == 2).sum()} / {((g.period != 1) & (g.period != 2)).sum()} | "
                    f"{p2.flip.mean() if len(p2) else float('nan'):.2f} | {p2.edges_class2.mean() if len(p2) else float('nan'):.2f} | "
                    f"{p2.cost_class0.mean() if len(p2) else float('nan'):.0f} / {p2.cost_class1.mean() if len(p2) else float('nan'):.0f} / {p2.cost_class2.mean() if len(p2) else float('nan'):.0f} | "
                    f"{p2.cost_greedy_class2.mean() if len(p2) else float('nan'):.0f} | "
                    f"{g.cost_x.mean():.0f} | {g.greedy_from_x.mean():.0f} | {g.single_moves_x.mean():.1f} | "
                    f"{g.br_steps.mean():.3f} | {g.cost2_increases.mean():.1f} |"
                    + extra
                )
            lines.append("")
    (RESULTS / "exp3_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def plot() -> None:
    df = pd.read_csv(RESULTS / "exp3_random.csv")
    fig, axes = new_fig(3, 1, width=4.0)
    for alg in ALGS:
        d = df[df.alg == alg]
        g = d.groupby("density")
        axes[0, 0].plot(
            g.period.apply(lambda s: (s == 2).mean()).index,
            g.period.apply(lambda s: (s == 2).mean()).values,
            marker="o",
            ms=4,
            label=LABELS[alg],
            **STYLE[alg],
        )
        p2 = d[d.period == 2].groupby("density")
        if len(p2):
            axes[0, 1].plot(
                p2.flip.mean().index,
                p2.flip.mean().values,
                marker="o",
                ms=4,
                label=LABELS[alg],
                **STYLE[alg],
            )
            axes[0, 2].plot(
                p2.cost_class2.mean().index,
                p2.cost_class2.mean().values,
                marker="o",
                ms=4,
                label=f"{LABELS[alg]}: both endpoints alternate",
                **STYLE[alg],
            )
            axes[0, 2].plot(
                p2.cost_class0.mean().index,
                p2.cost_class0.mean().values,
                marker="s",
                ms=4,
                mfc="none",
                label=f"{LABELS[alg]}: neither endpoint alternates",
                **{**STYLE[alg], "ls": "--"},
            )
    axes[0, 0].set_ylabel("fraction of runs ending with period 2")
    axes[0, 1].set_ylabel("fraction of alternating variables")
    axes[0, 2].set_ylabel("mean edge cost in a layer")
    for ax in axes[0]:
        ax.set_xlabel("edge density")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=7)
    save(fig, "exp3_density")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_all()
    summarize()
    plot()
