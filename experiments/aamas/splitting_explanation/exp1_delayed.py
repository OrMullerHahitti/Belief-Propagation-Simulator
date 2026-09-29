"""exp1_delayed: the exp1 records for the delayed split (DMS for k iterations, then a 0.5 split).

same benchmarks, seeds, horizon and per-iteration records as exp1_speed.py (cost, fraction of
committed arcs, variables that changed value, largest Q change), for every k of the paper's grid,
so the commitment fraction can be drawn around the split moment next to the exp1 lines.
the split is propflow's transfer mode (each clone inherits half of the current R messages, the
first step after the split is undamped), as in lab.FastEngine.split_now.

outputs: results/exp1_delayed_<bench>.npz (keys DMS_split_at_<k>/<record>) and
results/exp1_delayed_summary.md
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exp1_speed import BENCHES, SEEDS, T  # noqa: E402
from lab import (
    FastEngine,
    aaai_inst,
    detect_period,
    freeze_time,
    run_record,
    strict_freeze,
)  # noqa: E402
from plotting import BENCH_TITLE, RESULTS  # noqa: E402

# library iterations (the paper counts two per library iteration)
SPLIT_AT = (50, 100, 300, 500, 1000, 1500)
LAM = 0.9


def label(k: int) -> str:
    return f"DMS_split_at_{k}"


def task(args):
    bench, seed = args
    inst = aaai_inst(bench, seed)
    out = {}
    for k in SPLIT_AT:
        r = run_record(
            FastEngine(inst, lam=LAM), T, split_at=k, split_p=0.5, record_dq=True
        )
        a = r["assigns"]
        out[label(k)] = dict(
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
            for k in SPLIT_AT:
                alg = label(k)
                for key in ("costs", "sats", "changes", "dq"):
                    arrays[f"{alg}/{key}"] = np.stack([r[2][alg][key] for r in res])
                for key in ("freeze", "period", "final", "best", "t95"):
                    arrays[f"{alg}/{key}"] = np.array([r[2][alg][key] for r in res])
            np.savez_compressed(RESULTS / f"exp1_delayed_{bench}.npz", **arrays)
            print(f"{bench}: done", flush=True)


def summarize() -> None:
    lines = [
        "# exp1_delayed: the delayed split, same records as exp1 (50 seeds, 2000 iterations)",
        "",
    ]
    for bench in BENCHES:
        z = np.load(RESULTS / f"exp1_delayed_{bench}.npz")
        lines += [
            f"## {BENCH_TITLE[bench]}",
            "",
            "| split at | frozen within 2000 | median freeze | median t(commit >= 95%) | commitment before the split | final commitment | final cost | best cost | period 1 / 2 / other |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for k in SPLIT_AT:
            alg = label(k)
            fr = strict_freeze(z[f"{alg}/freeze"], T)
            frozen = fr < T
            per = z[f"{alg}/period"]
            sats = z[f"{alg}/sats"]
            lines.append(
                f"| {k} | {frozen.mean() * 100:.0f}% | "
                f"{np.median(fr[frozen]) if frozen.any() else float('nan'):.0f} | "
                f"{np.median(z[f'{alg}/t95']):.0f} | {sats[:, k - 1].mean():.2f} | {sats[:, -1].mean():.2f} | "
                f"{z[f'{alg}/final'].mean():.0f} +- {z[f'{alg}/final'].std():.0f} | {z[f'{alg}/best'].mean():.0f} | "
                f"{(per == 1).sum()} / {(per == 2).sum()} / {((per != 1) & (per != 2)).sum()} |"
            )
        lines.append("")
    (RESULTS / "exp1_delayed_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        run_all()
    summarize()
