"""exp1: how fast does each algorithm freeze, and what does the message state look like while it does?

on the five AAAI benchmarks (50 seeds, 2000 iterations, float semantics) run MS, DMS, MS + split,
DMS + split and DMS(0.5) + split and record per iteration: the decoded cost, the fraction of
committed arcs (the factor forwards a single row of its table), the number of variables whose
decoded value changed, and the largest change of any Q message.

outputs: results/exp1_<bench>.npz, results/exp1_summary.md and plots/exp1_*.pdf
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

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
from plotting import BENCH_TITLE, LABELS, RESULTS, STYLE, new_fig, save  # noqa: E402

T = 2000
SEEDS = 50
BENCHES = (
    "random_dense",
    "random_sparse",
    "graph_coloring",
    "scale_free",
    "meeting_scheduling",
)
ALGS = {
    "MS": dict(),
    "DMS": dict(lam=0.9),
    "MS_split": dict(split=0.5),
    "DMS_split": dict(split=0.5, lam=0.9),
    "DMS05_split": dict(split=0.5, lam=0.5),
}


def task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    out = {}
    for alg, kw in ALGS.items():
        r = run_record(FastEngine(inst, **kw), T, record_dq=True)
        a = r["assigns"]
        out[alg] = dict(
            costs=r["costs"].astype(np.float32),
            sats=r["sats"].astype(np.float32),
            changes=r["changes"].astype(np.int16),
            dq=r["dq"].astype(np.float32),
            freeze=freeze_time(r["changes"]),
            period=detect_period(a[-400:], pmax=64, window=200),
            final=float(r["costs"][-1]),
            best=float(r["costs"].min()),
            t95=int(np.argmax(r["sats"] >= 0.95)) if (r["sats"] >= 0.95).any() else T,
        )
    return bench, seed, out


def run_all() -> None:
    RESULTS.mkdir(exist_ok=True)
    with Pool() as pool:
        for bench in BENCHES:
            res = pool.map(task, [(bench, s) for s in range(SEEDS)])
            res.sort(key=lambda r: r[1])
            arrays = {}
            for alg in ALGS:
                for key in ("costs", "sats", "changes", "dq"):
                    arrays[f"{alg}/{key}"] = np.stack([r[2][alg][key] for r in res])
                for key in ("freeze", "period", "final", "best", "t95"):
                    arrays[f"{alg}/{key}"] = np.array([r[2][alg][key] for r in res])
            np.savez_compressed(RESULTS / f"exp1_{bench}.npz", **arrays)
            print(f"{bench}: done", flush=True)


def summarize() -> None:
    lines = ["# exp1: freeze time, commitment and cost (50 seeds, 2000 iterations)", ""]
    for bench in BENCHES:
        z = np.load(RESULTS / f"exp1_{bench}.npz")
        lines += [
            f"## {BENCH_TITLE[bench]}",
            "",
            "| algorithm | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for alg in ALGS:
            fr = strict_freeze(z[f"{alg}/freeze"], T)
            frozen = fr < T
            per = z[f"{alg}/period"]
            lines.append(
                f"| {LABELS[alg]} | {frozen.mean() * 100:.0f}% | "
                f"{np.median(fr[frozen]) if frozen.any() else float('nan'):.0f} | "
                f"{np.median(z[f'{alg}/t95']):.0f} | {z[f'{alg}/sats'][:, -1].mean():.2f} | "
                f"{z[f'{alg}/final'].mean():.0f} +- {z[f'{alg}/final'].std():.0f} | {z[f'{alg}/best'].mean():.0f} | "
                f"{(per == 1).sum()} / {(per == 2).sum()} / {((per != 1) & (per != 2)).sum()} |"
            )
        lines.append("")
    (RESULTS / "exp1_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def plot() -> None:
    it = np.arange(1, T + 1)
    for bench in BENCHES:
        z = np.load(RESULTS / f"exp1_{bench}.npz")
        # one representative run (seed 0): cost, commitment, assignment changes
        fig, axes = new_fig(3, 1, width=4.0)
        for alg in ("MS", "DMS", "MS_split", "DMS_split"):
            axes[0, 0].plot(it, z[f"{alg}/costs"][0], label=LABELS[alg], **STYLE[alg])
            axes[0, 1].plot(it, z[f"{alg}/sats"][0], label=LABELS[alg], **STYLE[alg])
            axes[0, 2].plot(it, z[f"{alg}/changes"][0], label=LABELS[alg], **STYLE[alg])
        for ax in axes[0]:
            ax.set_xscale("log")
            ax.set_xlabel("iteration")
        axes[0, 0].set_ylabel("cost of the decoded assignment")
        axes[0, 1].set_ylabel("fraction of committed arcs")
        axes[0, 2].set_ylabel("variables that changed value")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{BENCH_TITLE[bench]}, seed 0", fontsize=10)
        save(fig, f"exp1_{bench}_example")

        # aggregated: fraction of runs frozen by iteration t, mean commitment, mean cost
        fig, axes = new_fig(3, 1, width=4.0)
        for alg in ALGS:
            fr = strict_freeze(z[f"{alg}/freeze"], T)
            # freeze == T means "never froze": keep those runs out of the count at the last iteration
            frozen_by = np.array([((fr < T) & (fr <= t)).mean() for t in it])
            axes[0, 0].plot(it, frozen_by, label=LABELS[alg], **STYLE[alg])
            axes[0, 1].plot(
                it, z[f"{alg}/sats"].mean(axis=0), label=LABELS[alg], **STYLE[alg]
            )
            axes[0, 2].plot(
                it, z[f"{alg}/costs"].mean(axis=0), label=LABELS[alg], **STYLE[alg]
            )
        for ax in axes[0]:
            ax.set_xscale("log")
            ax.set_xlabel("iteration")
        axes[0, 0].set_ylabel("fraction of runs already frozen")
        axes[0, 1].set_ylabel("mean fraction of committed arcs")
        axes[0, 2].set_ylabel("mean cost")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{BENCH_TITLE[bench]}, 50 seeds", fontsize=10)
        save(fig, f"exp1_{bench}_aggregate")

        # message change after the freeze: geometric at rate lambda
        fig, axes = new_fig(1, 1)
        ax = axes[0, 0]
        for alg in ("DMS", "DMS_split", "DMS05_split"):
            dq = z[f"{alg}/dq"][0]
            ax.plot(it, np.maximum(dq, 1e-12), label=LABELS[alg], **STYLE[alg])
        fr = int(strict_freeze(z["DMS_split/freeze"], T)[0])
        if 0 < fr < T:
            # clipped at the same floor as the data so the reference line does not stretch the axis
            ref = np.maximum(z["DMS_split/dq"][0][fr - 1] * 0.9 ** (it - fr), 1e-12)
            ax.plot(
                it[fr - 1 :],
                ref[fr - 1 :],
                color="k",
                lw=0.8,
                ls="--",
                label="$0.9^{t}$ from the freeze",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("largest change of any Q message")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{BENCH_TITLE[bench]}, seed 0", fontsize=10)
        save(fig, f"exp1_{bench}_message_change")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_all()
    summarize()
    plot()
