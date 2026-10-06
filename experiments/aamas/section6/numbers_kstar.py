"""numbers for the Section 6 text after the 2026-10-06 runs, read from the result folders.

DMS-k^*DS: K = the best DMS point in the first 2000 paper iterations, damping kept (split_at_best_first2000/).
DMS-k^*DS-MGM: the same K, no damping after the split, MGM on two assignments (split_at_best_mgm_20261006/).
DABP and DABP-NoSplit, DMS-SCFG, MS-SCFG-MGM and the fixed k lines from data_paper_20261002.
iterations are printed in paper units (two per library iteration); every test is a paired Wilcoxon
signed-rank test over the 50 instances.

output: experiments/aamas/section6/out/numbers_kstar.md (also printed)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "aaai" / "data_paper_20261002"
KSTAR = HERE / "split_at_best_first2000"
KSTAR_OLD = HERE / "split_at_best_fixed_horizon"
MGM = HERE / "split_at_best_mgm_20261006"
OUT = HERE / "out" / "numbers_kstar.md"
BENCHES = [
    "random_sparse",
    "random_dense",
    "scale_free",
    "graph_coloring",
    "meeting_scheduling",
]
FIXED_K = [f"DMS_split_at_{k}" for k in (50, 100, 300, 500, 1000, 1500)]
# costs closer than this count as equal
TOL = 1e-6


def compare(a: pd.Series, b: pd.Series) -> str:
    """mean of a against mean of b, how often a is lower / equal / higher, Wilcoxon p."""
    d = (a - b).to_numpy()
    lower, higher = int((d < -TOL).sum()), int((d > TOL).sum())
    p = wilcoxon(a, b).pvalue if lower + higher else 1.0
    return (
        f"{a.mean():,.2f} vs {b.mean():,.2f} ({100 * (a.mean() - b.mean()) / b.mean():+.2f}%): "
        f"lower on {lower}, equal on {len(d) - lower - higher}, higher on {higher}, p = {p:.2g}"
    )


def main() -> None:
    out = []
    for bench in BENCHES:
        fin = pd.read_csv(DATA / f"{bench}_final_costs.csv").pivot(
            index="seed", columns="algorithm", values="final_cost"
        )
        ks = (
            pd.read_csv(KSTAR / f"{bench}_final_costs.csv")
            .set_index("seed")
            .sort_index()
        )
        old = (
            pd.read_csv(KSTAR_OLD / f"{bench}_final_costs.csv")
            .set_index("seed")
            .sort_index()
        )
        mg = (
            pd.read_csv(MGM / f"{bench}_final_costs.csv").set_index("seed").sort_index()
        )
        settle = pd.read_csv(MGM / "settle.csv")
        settle = settle[settle.benchmark == bench]
        settled = settle[settle.kind != "none"]
        fixed_means = {k: fin[k].mean() for k in FIXED_K if k in fin}
        # fixed k in paper iterations
        fixed_text = ", ".join(
            f"{2 * int(k.rsplit('_', 1)[1])}: {m:,.2f}" for k, m in fixed_means.items()
        )
        out += [
            f"## {bench}",
            f"- DMS-k^*DS split iteration (paper): median {2 * ks.t_star.median():.0f}, "
            f"range {2 * ks.t_star.min()}-{2 * ks.t_star.max()}; prefix equals DMS on {int(ks.prefix_matches_dms.sum())}/50",
            f"- DMS-k^*DS final vs its cost at the split: {compare(ks.final_cost, ks.cost_at_split)}",
            f"- DMS-k^*DS vs DMS-SCFG: {compare(ks.final_cost, fin['DMS_split_0.5'])}",
            f"- DMS-k^*DS vs the old version (K from all 4000 iterations): {compare(ks.final_cost, old.final_cost)}",
            f"- fixed k means: {fixed_text}; "
            f"DMS-k^*DS below every one: {bool(all(ks.final_cost.mean() < m for m in fixed_means.values()))}",
            f"- DMS-k^*DS-MGM undamped iterations needed (paper): settled {len(settled)}/50 "
            f"(one assignment {(settle.kind == 'fixed').sum()}, two alternating {(settle.kind == 'two').sum()}, "
            f"never {(settle.kind == 'none').sum()})"
            + (
                f", median {2 * settled.settle_steps.median():.0f}, max {2 * int(settled.settle_steps.max())}"
                if len(settled)
                else ""
            ),
            f"- DMS-k^*DS-MGM vs the DMS best at its split: {compare(mg.final_cost, mg.dms_best)}",
            f"- DMS-k^*DS-MGM vs DMS-k^*DS: {compare(mg.final_cost, ks.final_cost)}",
            f"- DMS-k^*DS-MGM vs DMS-SCFG: {compare(mg.final_cost, fin['DMS_split_0.5'])}",
            f"- DMS-k^*DS-MGM vs MS-SCFG-MGM: {compare(mg.final_cost, fin['MS_split_MGM_200'])}",
            f"- DABP vs DABP-NoSplit: {compare(fin['Attentive'], fin['Attentive_NoSplit'])}",
            "",
        ]
    text = "\n".join(out)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
