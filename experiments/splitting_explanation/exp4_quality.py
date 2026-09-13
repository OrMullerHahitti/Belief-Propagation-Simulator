"""exp4: what kind of solution does the split freeze on, and why is it worse than DMS on random dense?

on random dense / random sparse (50 seeds, 2000 iterations, float semantics) take the final decoded
assignment of DMS, DMS + split, MS + split and DMS with a split at iteration K (50 .. 1500), and for
each: the cost, the number of improving single-variable moves, improving two-variable edge moves,
improving three-variable path moves (sampled induced paths), and the cost after greedy 1-opt local
search from it. references: greedy 1-opt from 5 random assignments, and DMS's decoded assignment at K
repaired by greedy 1-opt.
outputs: results/exp4.csv, results/exp4_summary.md, plots/exp4_split_at_k.pdf
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
    aaai_inst,
    greedy_1opt,
    improving_edge_moves,
    improving_path_moves,
    improving_single_moves,
    run_record,
)
from plotting import BENCH_TITLE, LABELS, RESULTS, STYLE, new_fig, save  # noqa: E402

T = 2000
SEEDS = 50
BENCHES = ("random_dense", "random_sparse")
KS = (50, 100, 300, 500, 1000, 1500)


def describe(inst, x, rng):
    imp, samp = improving_path_moves(inst, x, n_samples=2000, rng=rng)
    gx, moves = greedy_1opt(inst, x, rng)
    return dict(
        cost=inst.cost(x),
        single=improving_single_moves(inst, x),
        edge=improving_edge_moves(inst, x),
        path_improving=imp,
        path_sampled=samp,
        greedy_cost=inst.cost(gx),
        greedy_moves=moves,
    )


def task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    rng = np.random.default_rng(500 + seed)
    rows = []
    r = run_record(FastEngine(inst, lam=0.9), T)
    finals = {"DMS": r["assigns"][-1]}
    dms_at = {K: r["assigns"][K - 1] for K in KS}
    finals["DMS_split"] = run_record(FastEngine(inst, split=0.5, lam=0.9), T)[
        "assigns"
    ][-1]
    finals["MS_split"] = run_record(FastEngine(inst, split=0.5), T)["assigns"][-1]
    for K in KS:
        finals[f"split_at_{K}"] = run_record(FastEngine(inst, lam=0.9), T, split_at=K)[
            "assigns"
        ][-1]
    for name, x in finals.items():
        rows.append(dict(bench=bench, seed=seed, name=name, **describe(inst, x, rng)))
    for K in KS:
        rows.append(
            dict(
                bench=bench,
                seed=seed,
                name=f"DMS_at_{K}",
                **describe(inst, dms_at[K], rng),
            )
        )
    for k in range(5):
        x0 = rng.integers(0, inst.m, size=inst.n)
        rows.append(
            dict(
                bench=bench,
                seed=seed,
                name=f"random_start_{k}",
                **describe(inst, x0, rng),
            )
        )
    return rows


def run_all() -> None:
    RESULTS.mkdir(exist_ok=True)
    with Pool() as pool:
        rows = []
        for bench in BENCHES:
            rows += [
                row
                for rr in pool.map(task, [(bench, s) for s in range(SEEDS)])
                for row in rr
            ]
            print(f"{bench}: done", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "exp4.csv", index=False)


def summarize() -> None:
    df = pd.read_csv(RESULTS / "exp4.csv")
    lines = [
        "# exp4: final assignments and their neighbourhoods (50 seeds, 2000 iterations)",
        "",
    ]
    order = (
        ["DMS", "DMS_split", "MS_split"]
        + [f"split_at_{K}" for K in KS]
        + [f"DMS_at_{K}" for K in KS]
        + ["random_start"]
    )
    for bench in BENCHES:
        d = df[df.bench == bench].copy()
        d["group"] = d.name.str.replace(r"random_start_\d", "random_start", regex=True)
        lines += [
            f"## {BENCH_TITLE[bench]}",
            "",
            "| assignment | cost | improving single moves | improving edge moves | improving path moves (of sampled) | cost after greedy 1-opt | greedy moves |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in order:
            g = d[d.group == name]
            if len(g) == 0:
                continue
            lines.append(
                f"| {name} | {g.cost.mean():.0f} +- {g.cost.std():.0f} | {g.single.mean():.2f} | {g.edge.mean():.2f} | "
                f"{g.path_improving.mean():.1f} of {g.path_sampled.mean():.0f} | {g.greedy_cost.mean():.0f} | {g.greedy_moves.mean():.1f} |"
            )
        lines.append("")
    (RESULTS / "exp4_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def plot() -> None:
    df = pd.read_csv(RESULTS / "exp4.csv")
    fig, axes = new_fig(2, 1, width=4.4)
    for ax, bench in zip(axes[0], BENCHES):
        d = df[df.bench == bench]
        ks = np.array(KS)
        split_k = [d[d.name == f"split_at_{K}"].cost.mean() for K in KS]
        dms_k = [d[d.name == f"DMS_at_{K}"].cost.mean() for K in KS]
        dms_k_greedy = [d[d.name == f"DMS_at_{K}"].greedy_cost.mean() for K in KS]
        ax.plot(
            ks,
            split_k,
            marker="o",
            ms=4,
            color="tab:red",
            label="DMS, split at K (final)",
        )
        ax.plot(
            ks, dms_k, marker="s", ms=4, color="0.2", ls="--", label="DMS decoded at K"
        )
        ax.plot(
            ks,
            dms_k_greedy,
            marker="^",
            ms=4,
            color="tab:blue",
            ls="-.",
            label="DMS decoded at K, then greedy 1-opt",
        )
        ax.axhline(
            d[d.name == "DMS"].cost.mean(), color="0.2", ls=":", label="DMS final"
        )
        ax.axhline(
            d[d.name == "DMS_split"].cost.mean(),
            color="tab:red",
            ls=":",
            label=LABELS["DMS_split"],
        )
        rs = d[d.name.str.startswith("random_start")].greedy_cost.mean()
        ax.axhline(rs, color="0.6", ls="-", lw=0.8, label="greedy 1-opt from random")
        ax.set_xscale("log")
        ax.set_xlabel("split iteration K")
        ax.set_ylabel("mean cost")
        ax.set_title(BENCH_TITLE[bench], fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=7)
    save(fig, "exp4_split_at_k")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_all()
    summarize()
    plot()
