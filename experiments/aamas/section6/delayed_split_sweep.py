"""section 6, item 3: the delayed split over the whole k grid, all 50 instances per benchmark.

per benchmark and k: mean final cost of DMS_split_at_k, its gain against the split from the
start (DMS_split_0.5; negative = lower cost), the paired Wilcoxon p, wins/ties/losses, and the
number of runs whose cost settled. k is printed in paper iterations (two per library iteration).

inputs:  experiments/aaai/data_paper_20260928/<bench>_{final_costs,settling}.csv
outputs: experiments/aamas/section6/out/delayed_split_sweep.{csv,md,tex}
         experiments/aamas/section6/out/delayed_split_sweep.pdf (gain vs k, one line per benchmark)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "aaai" / "data_paper_20260928"
OUT = HERE / "out"
BENCHES = [
    ("random_sparse", "random sparse"),
    ("random_dense", "random dense"),
    ("scale_free", "scale free"),
    ("graph_coloring", "graph coloring"),
    ("meeting_scheduling", "meeting scheduling"),
]
KS = (50, 100, 300, 500, 1000, 1500)  # library units
BASE = "DMS_split_0.5"


def remove_frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def sweep() -> pd.DataFrame:
    rows = []
    for bench, title in BENCHES:
        final = pd.read_csv(DATA / f"{bench}_final_costs.csv").pivot(
            index="seed", columns="algorithm", values="final_cost"
        )
        settling = pd.read_csv(DATA / f"{bench}_settling.csv").set_index("algorithm")
        base = final[BASE]
        for k in KS:
            alg = f"DMS_split_at_{k}"
            if alg not in final:
                continue
            d = final[alg] - base
            rows.append(
                dict(
                    benchmark=title,
                    k=2 * k,
                    mean=final[alg].mean(),
                    base_mean=base.mean(),
                    dms_mean=final["DMS"].mean(),
                    gain_pct=100 * d.mean() / base.mean(),
                    p=wilcoxon(final[alg], base).pvalue if (d != 0).any() else 1.0,
                    wins=int((d < 0).sum()),
                    ties=int((d == 0).sum()),
                    losses=int((d > 0).sum()),
                    settled=int(settling.loc[alg, "n_settled"]),
                )
            )
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(OUT / "delayed_split_sweep.csv", index=False)
    ks = sorted(df.k.unique())
    md = [
        "| benchmark | DMS | DMS-SCFG | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|---|---|" + "---|" * len(ks),
    ]
    tex = [
        "\\begin{tabular}{l" + "r" * len(ks) + "}",
        "benchmark & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\ \\hline",
    ]
    for _, title in BENCHES:
        sub = df[df.benchmark == title].set_index("k")
        cells_md, cells_tex = [], []
        for k in ks:
            if k not in sub.index:
                cells_md.append("—")
                cells_tex.append("--")
                continue
            r = sub.loc[k]
            star = "*" if r.p < 0.01 else ""
            cells_md.append(f"{r.gain_pct:+.2f}%{star} (p {r.p:.2g})")
            cells_tex.append(f"${r.gain_pct:+.2f}\\%$" + ("$^{*}$" if star else ""))
        md.append(
            f"| {title} | {sub.dms_mean.iloc[0]:,.1f} | {sub.base_mean.iloc[0]:,.1f} | "
            + " | ".join(cells_md)
            + " |"
        )
        tex.append(f"{title} & " + " & ".join(cells_tex) + " \\\\")
    tex.append("\\end{tabular}")
    md.append("")
    md.append(
        "gain = mean final cost of DMS-kDS relative to DMS-SCFG (split from the start), 50 instances; * = paired Wilcoxon p < 0.01; k in paper iterations"
    )
    (OUT / "delayed_split_sweep.md").write_text("\n".join(md) + "\n")
    (OUT / "delayed_split_sweep.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


def plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for _, title in BENCHES:
        sub = df[df.benchmark == title].sort_values("k")
        (line,) = ax.plot(
            sub.k, sub.gain_pct, lw=1.0, marker="o", ms=3, mfc="none", label=title
        )
        sig = sub[sub.p < 0.01]
        ax.plot(
            sig.k, sig.gain_pct, ls="none", marker="o", ms=3, color=line.get_color()
        )
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 600, 1000, 2000, 3000])
    ax.set_xticklabels(["100", "200", "600", "1000", "2000", "3000"], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("split iteration $k$", fontsize=8)
    ax.set_ylabel("cost relative to DMS-SCFG (%)", fontsize=8)
    ax.legend(frameon=False, fontsize=6, title="filled: p < 0.01", title_fontsize=6)
    remove_frame(ax)
    fig.tight_layout()
    path = OUT / "delayed_split_sweep.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = sweep()
    write_tables(df)
    plot(df)
