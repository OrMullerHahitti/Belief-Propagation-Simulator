"""exp5: damping sweep. with and without the split, damping lambda in {0, 0.2, ..., 0.98}, on random
dense / random sparse / graph coloring, 20 seeds, 2000 iterations: freeze time, commitment, final and
best cost, period. outputs: results/exp5.csv, results/exp5_summary.md, plots/exp5_damping.pdf
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
    detect_period,
    freeze_time,
    run_record,
    strict_freeze,
)
from plotting import BENCH_TITLE, RESULTS, new_fig, save  # noqa: E402

T = 2000
SEEDS = 20
BENCHES = ("random_dense", "random_sparse", "graph_coloring")
LAMS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    rows = []
    for split in (None, 0.5):
        for lam in LAMS:
            r = run_record(FastEngine(inst, split=split, lam=lam), T)
            rows.append(
                dict(
                    bench=bench,
                    seed=seed,
                    split=split is not None,
                    lam=lam,
                    freeze=freeze_time(r["changes"]),
                    t95=int(np.argmax(r["sats"] >= 0.95))
                    if (r["sats"] >= 0.95).any()
                    else T,
                    final_commit=float(r["sats"][-1]),
                    final=float(r["costs"][-1]),
                    best=float(r["costs"].min()),
                    period=detect_period(r["assigns"][-400:], pmax=64, window=200),
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
    pd.DataFrame(rows).to_csv(RESULTS / "exp5.csv", index=False)


def summarize() -> None:
    df = pd.read_csv(RESULTS / "exp5.csv")
    df["freeze"] = strict_freeze(df.freeze, T)
    lines = ["# exp5: damping sweep (20 seeds, 2000 iterations)", ""]
    for bench in BENCHES:
        for split in (False, True):
            d = df[(df.bench == bench) & (df.split == split)]
            lines += [
                f"## {BENCH_TITLE[bench]}, {'with' if split else 'without'} split",
                "",
                "| damping | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for lam, g in d.groupby("lam"):
                fr = g.freeze.values
                frozen = fr < T
                lines.append(
                    f"| {lam} | {frozen.mean() * 100:.0f}% | {np.median(fr[frozen]) if frozen.any() else float('nan'):.0f} | "
                    f"{np.median(g.t95):.0f} | {g.final_commit.mean():.2f} | {g.final.mean():.0f} +- {g.final.std():.0f} | {g.best.mean():.0f} | "
                    f"{(g.period == 1).sum()} / {(g.period == 2).sum()} / {((g.period != 1) & (g.period != 2)).sum()} |"
                )
            lines.append("")
    (RESULTS / "exp5_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def plot() -> None:
    df = pd.read_csv(RESULTS / "exp5.csv")
    df["freeze"] = strict_freeze(df.freeze, T)
    fig, axes = new_fig(3, 2, width=4.0, height=3.0)
    for col, bench in enumerate(BENCHES):
        for split, color, label in (
            (True, "tab:red", "with split"),
            (False, "0.2", "without split"),
        ):
            d = df[(df.bench == bench) & (df.split == split)]
            g = d.groupby("lam")
            med = g.freeze.median()
            q1, q3 = g.freeze.quantile(0.25), g.freeze.quantile(0.75)
            axes[0, col].plot(
                med.index, med.values, marker="o", ms=4, color=color, label=label
            )
            axes[0, col].fill_between(
                med.index, q1.values, q3.values, color=color, alpha=0.15, lw=0
            )
            axes[1, col].plot(
                g.final.mean().index,
                g.final.mean().values,
                marker="o",
                ms=4,
                color=color,
                label=label,
            )
        axes[0, col].set_title(BENCH_TITLE[bench], fontsize=10)
        axes[0, col].set_ylabel("freeze time (median, IQR)")
        axes[0, col].set_yscale("log")
        axes[1, col].set_ylabel("mean final cost")
        for ax in axes[:, col]:
            ax.set_xlabel("damping $\\lambda$")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "exp5_damping")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_all()
    summarize()
    plot()
