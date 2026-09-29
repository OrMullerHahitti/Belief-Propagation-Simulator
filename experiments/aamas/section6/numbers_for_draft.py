"""every number the Section 6 draft quotes, read from the data folders (nothing typed by hand).

iterations are printed in paper units (two per library iteration).

inputs: experiments/aaai/data_paper_20260928, experiments/aamas/splitting_explanation/results,
        experiments/aamas/section6/split_at_best, experiments/aamas/section6/out/delayed_split_sweep.csv
output: experiments/aamas/section6/out/numbers.md (also printed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "aaai" / "data_paper_20260928"
EXP1 = HERE.parent / "splitting_explanation" / "results"
BEST = HERE / "split_at_best_fixed_horizon"  # DMS-BDS: split at each instance's own best DMS iteration, run to library iteration 2000
OUT = HERE / "out"
BENCHES = [
    ("random_sparse", "random sparse"),
    ("random_dense", "random dense"),
    ("scale_free", "scale free"),
    ("graph_coloring", "graph coloring"),
    ("meeting_scheduling", "meeting scheduling"),
]
LINES = [
    "MS",
    "MS_split_0.5",
    "MS_split_MGM_200",
    "MS_split_opt_200",
    "DMS",
    "DMS_split_0.5",
    "DMS_split_0.4_0.6",
    "DMS_split_at_500",
    "Attentive",
    "Attentive_NoSplit",
    "Optimal",
]
PAIRS = [
    ("DMS_split_0.5", "DMS"),
    ("MS_split_0.5", "MS"),
    ("MS_split_MGM_200", "MS_split_0.5"),
    ("MS_split_opt_200", "MS_split_0.5"),
    ("MS_split_opt_200", "MS_split_MGM_200"),
    ("DMS_split_0.5", "MS_split_opt_200"),
    ("DMS_split_0.4_0.6", "DMS_split_0.5"),
    ("DMS_split_at_500", "DMS_split_0.5"),
    ("DMS_split_at_500", "DMS"),
    ("Attentive", "DMS_split_0.5"),
    ("Attentive", "Attentive_NoSplit"),
]
T = 2000  # library iterations of exp1
MIN_QUIET = 100


def strict_freeze(fr: np.ndarray) -> np.ndarray:
    # exp1's rule: a freeze needs 100 quiet iterations before the horizon
    return np.where(fr <= T - MIN_QUIET, fr, T)


lines = []


def p(s: str = "") -> None:
    lines.append(s)
    print(s)


for bench, title in BENCHES:
    final = pd.read_csv(DATA / f"{bench}_final_costs.csv").pivot(
        index="seed", columns="algorithm", values="final_cost"
    )
    settling = pd.read_csv(DATA / f"{bench}_settling.csv").set_index("algorithm")
    p(f"## {title}")
    p()
    p(
        "| line | mean final cost | std | settled of 50 | median settling iteration (paper) |"
    )
    p("|---|---|---|---|---|")
    for a in LINES:
        if a not in final:
            continue
        col = final[a].dropna()
        if a in settling.index:
            s = settling.loc[a]
            st = f"{int(s.n_settled)} | {2 * s.settle_median:.0f}"
        else:
            st = "— | —"
        p(f"| {a} | {col.mean():,.2f} | {col.std():,.2f} | {st} |")
    p()
    p("| a | b | mean a | mean b | diff % | Wilcoxon p | a lower / ties / a higher |")
    p("|---|---|---|---|---|---|---|")
    for a, b in PAIRS:
        if a not in final or b not in final:
            continue
        d = (final[a] - final[b]).dropna()
        pv = (
            wilcoxon(final[a][d.index], final[b][d.index]).pvalue
            if (d != 0).any()
            else 1.0
        )
        p(
            f"| {a} | {b} | {final[a].mean():,.2f} | {final[b].mean():,.2f} | {100 * d.mean() / final[b].mean():+.2f} | {pv:.2g} | {(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()} |"
        )
    # merge lines: share of the MS-SCFG -> DMS-SCFG gap closed by the merge
    gap = final["MS_split_0.5"].mean() - final["DMS_split_0.5"].mean()
    for m in ("MS_split_MGM_200", "MS_split_opt_200"):
        closed = (final["MS_split_0.5"].mean() - final[m].mean()) / gap
        p(f"- {m} closes {100 * closed:.1f}% of the gap between MS-SCFG and DMS-SCFG")
    # exp1: fraction at a bound, freeze, period-2 tails
    z = np.load(EXP1 / f"exp1_{bench}.npz")
    zd = np.load(EXP1 / f"exp1_delayed_{bench}.npz")
    p()
    p(
        "exp1 (fraction of function-to-variable messages at a bound; freeze = 200 paper iterations without a value change):"
    )
    for alg, name in (
        ("MS", "MS"),
        ("DMS", "DMS"),
        ("MS_split", "MS-SCFG"),
        ("DMS_split", "DMS-SCFG"),
    ):
        sats = z[f"{alg}/sats"]
        fr = strict_freeze(z[f"{alg}/freeze"])
        frozen = fr < T
        per = z[f"{alg}/period"]
        p(
            f"- {name}: at a bound at the end {sats[:, -1].mean():.2f}, after 200 paper iterations {sats[:, 99].mean():.2f}; "
            f"frozen runs {frozen.sum()}/50, median freeze {2 * np.median(fr[frozen]) if frozen.any() else float('nan'):.0f}; "
            f"tail period 1/2/other {(per == 1).sum()}/{(per == 2).sum()}/{((per != 1) & (per != 2)).sum()}"
        )
    k = 500
    sats = zd[f"DMS_split_at_{k}/sats"]
    fr = strict_freeze(zd[f"DMS_split_at_{k}/freeze"])
    frozen = fr < T
    p(
        f"- DMS-kDS (k = {2 * k}): at a bound just before the split {sats[:, k - 1].mean():.2f}, 100 paper iterations after {sats[:, k + 49].mean():.2f}, "
        f"at the end {sats[:, -1].mean():.2f}; frozen {frozen.sum()}/50, median freeze {2 * np.median(fr[frozen]):.0f} "
        f"(= {2 * np.median(fr[frozen]) - 2 * k:.0f} after the split)"
    )
    # split at the best point
    b = pd.read_csv(BEST / f"{bench}_final_costs.csv")
    lower = (b.final_cost < b.cost_at_split - 1e-6).sum()
    higher = (b.final_cost > b.cost_at_split + 1e-6).sum()
    p(
        f"- split at the best point: DMS best {b.cost_at_split.mean():,.2f} (median t* {2 * b.t_star.median():.0f} paper iterations) -> "
        f"{b.final_cost.mean():,.2f} at iteration 4000; lower/same/higher {lower}/{50 - lower - higher}/{higher}; "
        f"prefix check {int(b.prefix_matches_dms.sum())}/50"
    )
    p()

p("## delayed split sweep (gain vs DMS-SCFG, all 50 instances; k in paper iterations)")
p()
sweep = pd.read_csv(OUT / "delayed_split_sweep.csv")
for _, title in BENCHES:
    sub = sweep[sweep.benchmark == title].sort_values("k")
    p(
        f"- {title}: "
        + ", ".join(
            f"k={int(r.k)}: {r.gain_pct:+.2f}% (p {r.p:.2g}, {int(r.wins)}/{int(r.ties)}/{int(r.losses)})"
            for r in sub.itertuples()
        )
    )

OUT.mkdir(exist_ok=True)
(OUT / "numbers.md").write_text("\n".join(lines) + "\n")
