"""exp6: split ratio sweep. DMS (0.9) with an asymmetric split p / (1-p), p in {0.5, ..., 0.99}, on
random dense / random sparse, 20 seeds, 2000 iterations: freeze time, commitment, final and best cost,
period. p -> 1 approaches the unsplit graph (the weak clone carries almost nothing).
outputs: results/exp6.csv, results/exp6_summary.md, plots/exp6_split_ratio.pdf
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
BENCHES = ("random_dense", "random_sparse")
PS = (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    rows = []
    for p in PS:
        r = run_record(FastEngine(inst, split=p, lam=0.9), T)
        rows.append(
            dict(
                bench=bench,
                seed=seed,
                p=p,
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
    r = run_record(FastEngine(inst, lam=0.9), T)
    rows.append(
        dict(
            bench=bench,
            seed=seed,
            p=1.0,
            freeze=freeze_time(r["changes"]),
            t95=int(np.argmax(r["sats"] >= 0.95)) if (r["sats"] >= 0.95).any() else T,
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
    pd.DataFrame(rows).to_csv(RESULTS / "exp6.csv", index=False)


def summarize() -> None:
    df = pd.read_csv(RESULTS / "exp6.csv")
    df["freeze"] = strict_freeze(df.freeze, T)
    lines = [
        "# exp6: split ratio sweep, DMS 0.9 (20 seeds, 2000 iterations); p = 1 is the unsplit graph",
        "",
    ]
    for bench in BENCHES:
        d = df[df.bench == bench]
        lines += [
            f"## {BENCH_TITLE[bench]}",
            "",
            "| p | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for p, g in d.groupby("p"):
            fr = g.freeze.values
            frozen = fr < T
            lines.append(
                f"| {p} | {frozen.mean() * 100:.0f}% | {np.median(fr[frozen]) if frozen.any() else float('nan'):.0f} | "
                f"{np.median(g.t95):.0f} | {g.final_commit.mean():.2f} | {g.final.mean():.0f} +- {g.final.std():.0f} | {g.best.mean():.0f} | "
                f"{(g.period == 1).sum()} / {(g.period == 2).sum()} / {((g.period != 1) & (g.period != 2)).sum()} |"
            )
        lines.append("")
    (RESULTS / "exp6_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def plot() -> None:
    df = pd.read_csv(RESULTS / "exp6.csv")
    df["freeze"] = strict_freeze(df.freeze, T)
    fig, axes = new_fig(2, 2, width=4.0, height=3.0)
    for col, bench in enumerate(BENCHES):
        d = df[df.bench == bench]
        g = d.groupby("p")
        med = g.freeze.median()
        q1, q3 = g.freeze.quantile(0.25), g.freeze.quantile(0.75)
        axes[0, col].plot(med.index, med.values, marker="o", ms=4, color="tab:red")
        axes[0, col].fill_between(
            med.index, q1.values, q3.values, color="tab:red", alpha=0.15, lw=0
        )
        axes[1, col].plot(
            g.final.mean().index,
            g.final.mean().values,
            marker="o",
            ms=4,
            color="tab:red",
            label="final cost",
        )
        axes[1, col].plot(
            g.best.mean().index,
            g.best.mean().values,
            marker="s",
            ms=4,
            color="0.2",
            ls="--",
            label="best cost seen",
        )
        axes[0, col].set_title(BENCH_TITLE[bench], fontsize=10)
        axes[0, col].set_ylabel("freeze time (median, IQR)")
        axes[0, col].set_yscale("log")
        axes[1, col].set_ylabel("mean cost")
        for ax in axes[:, col]:
            ax.set_xlabel("split ratio p (1 = no split)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    save(fig, "exp6_split_ratio")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_all()
    summarize()
    plot()
