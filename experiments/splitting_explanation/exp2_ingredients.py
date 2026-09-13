"""exp2: which ingredient of the split does what?

by Theorem 1 the symmetric split equals min-sum on the original graph with the variable update
Q = cavity + belief = 2 cavity + (the recipient's own R). run the four variants on the original
graph:
  cav         Q = cavity                (plain min-sum)
  2cav        Q = 2 cavity              (doubling only)
  belief      Q = cavity + own R        (echo only)
  cav+belief  Q = 2 cavity + own R      (both = the split)
undamped and damped (0.9), on random dense / random sparse / graph coloring, 50 seeds, 2000
iterations. outputs: results/exp2_<bench>.npz, results/exp2_summary.md, plots/exp2_*.pdf
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
BENCHES = ("random_dense", "random_sparse", "graph_coloring")
RULES = ("cav", "2cav", "belief", "cav+belief")
LAMS = (0.0, 0.9)


def task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    out = {}
    for rule in RULES:
        for lam in LAMS:
            r = run_record(FastEngine(inst, rule=rule, lam=lam), T)
            out[(rule, lam)] = dict(
                sats=r["sats"].astype(np.float32),
                costs=r["costs"].astype(np.float32),
                freeze=freeze_time(r["changes"]),
                period=detect_period(r["assigns"][-400:], pmax=64, window=200),
                final=float(r["costs"][-1]),
                best=float(r["costs"].min()),
                t95=int(np.argmax(r["sats"] >= 0.95))
                if (r["sats"] >= 0.95).any()
                else T,
            )
    return bench, seed, out


def key(rule, lam):
    return f"{rule}/lam{lam:.1f}"


def run_all() -> None:
    RESULTS.mkdir(exist_ok=True)
    with Pool() as pool:
        for bench in BENCHES:
            res = pool.map(task, [(bench, s) for s in range(SEEDS)])
            res.sort(key=lambda r: r[1])
            arrays = {}
            for rule in RULES:
                for lam in LAMS:
                    k = key(rule, lam)
                    for f in ("sats", "costs"):
                        arrays[f"{k}/{f}"] = np.stack(
                            [r[2][(rule, lam)][f] for r in res]
                        )
                    for f in ("freeze", "period", "final", "best", "t95"):
                        arrays[f"{k}/{f}"] = np.array(
                            [r[2][(rule, lam)][f] for r in res]
                        )
            np.savez_compressed(RESULTS / f"exp2_{bench}.npz", **arrays)
            print(f"{bench}: done", flush=True)


def summarize() -> None:
    lines = ["# exp2: the two ingredients of the split (50 seeds, 2000 iterations)", ""]
    for bench in BENCHES:
        z = np.load(RESULTS / f"exp2_{bench}.npz")
        lines += [
            f"## {BENCH_TITLE[bench]}",
            "",
            "| update rule | damping | frozen within 2000 | median freeze | median t(commit >= 95%) | final commitment | final cost | best cost | period 1 / 2 / other |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for lam in LAMS:
            for rule in RULES:
                k = key(rule, lam)
                fr = strict_freeze(z[f"{k}/freeze"], T)
                frozen = fr < T
                per = z[f"{k}/period"]
                lines.append(
                    f"| {LABELS[rule]} | {lam:.1f} | {frozen.mean() * 100:.0f}% | "
                    f"{np.median(fr[frozen]) if frozen.any() else float('nan'):.0f} | {np.median(z[f'{k}/t95']):.0f} | "
                    f"{z[f'{k}/sats'][:, -1].mean():.2f} | {z[f'{k}/final'].mean():.0f} +- {z[f'{k}/final'].std():.0f} | "
                    f"{z[f'{k}/best'].mean():.0f} | {(per == 1).sum()} / {(per == 2).sum()} / {((per != 1) & (per != 2)).sum()} |"
                )
        lines.append("")
    (RESULTS / "exp2_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def plot() -> None:
    it = np.arange(1, T + 1)
    for bench in BENCHES:
        z = np.load(RESULTS / f"exp2_{bench}.npz")
        fig, axes = new_fig(2, 2, width=4.2, height=3.0)
        for col, lam in enumerate(LAMS):
            for rule in RULES:
                k = key(rule, lam)
                axes[0, col].plot(
                    it, z[f"{k}/sats"].mean(axis=0), label=LABELS[rule], **STYLE[rule]
                )
                fr = strict_freeze(z[f"{k}/freeze"], T)
                axes[1, col].plot(
                    it,
                    [((fr < T) & (fr <= t)).mean() for t in it],
                    label=LABELS[rule],
                    **STYLE[rule],
                )
            axes[0, col].set_title(
                f"{BENCH_TITLE[bench]}, damping {lam:.1f}", fontsize=10
            )
            axes[0, col].set_ylabel("mean fraction of committed arcs")
            axes[1, col].set_ylabel("fraction of runs already frozen")
            # undamped no run freezes; keep the same axis as the damped panel instead of a flat line at 0
            axes[1, col].set_ylim(-0.02, 1.02)
            for ax in axes[:, col]:
                ax.set_xscale("log")
                ax.set_xlabel("iteration")
        axes[0, 0].legend(frameon=False, fontsize=8)
        save(fig, f"exp2_{bench}")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run_all()
    summarize()
    plot()
