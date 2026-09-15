"""Paper-level statistics from the per-seed final costs.

Two outputs per benchmark, next to the inputs:

  {benchmark}_heldout_k.csv
      delayed-split K chosen on seeds 0-24 (lowest mean final cost) and
      evaluated on seeds 25-49, paired against immediate splitting there.

  {benchmark}_key_comparisons.csv
      a fixed list of paired comparisons (final cost, n = 50): mean difference
      a - b, Wilcoxon p, Holm-adjusted p over the list, paired bootstrap 95%
      CI of the mean difference, wins / ties / losses of a over b, and the
      standardized effect d_z = mean(diff) / std(diff).

Example:
  uv run python experiments/aaai/code/key_comparisons.py --benchmarks all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SELECT_SEEDS = range(0, 25)
EVAL_SEEDS = range(25, 50)
BOOTSTRAP = 10_000
# (a, b): the question is "is a better (lower) than b?"
PAIRS = [
    ("DMS", "MS"),
    ("DMS_split_0.5", "DMS"),
    ("DMS_split_0.4_0.6", "DMS_split_0.5"),
    ("DMS_split_0.95", "DMS_split_0.5"),
    ("DMS_split_pulse", "DMS_split_0.5"),
    ("DMS_split_pulse", "DMS_split_0.95"),
    ("DMS_split_at_1000", "DMS_split_0.5"),
    ("DMS_split_at_300", "DMS_split_0.5"),
    ("DMS_split_at_100", "DMS_split_0.5"),
    ("Attentive", "DMS_split_0.5"),
    ("Attentive", "DMS_split_at_1000"),
    ("Attentive", "DMS_split_0.95"),
    ("Attentive", "DMS_split_pulse"),
    ("Attentive", "Attentive_NoSplit"),
    ("DMS", "Attentive_NoSplit"),
    ("MS_split_MGM_200", "MS_split_0.5"),
    ("MS_split_opt_200", "MS_split_MGM_200"),
    ("MS_split_MGM_200", "MS_split_MGM_inverted_200"),
    ("DMS_split_0.5", "MS_split_opt_200"),
    ("DMS", "MS_split_opt_200"),
]


def paired(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    diff = x - y
    n = len(diff)
    if np.allclose(diff, 0.0):
        w_p = 1.0
    else:
        w_p = float(stats.wilcoxon(x, y).pvalue)
    idx = rng.integers(0, n, size=(BOOTSTRAP, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = diff.std(ddof=1)
    return {
        "n": n,
        "mean_a": x.mean(),
        "mean_b": y.mean(),
        "mean_diff": diff.mean(),
        "wilcoxon_p": w_p,
        "ci_low": lo,
        "ci_high": hi,
        "wins": int((diff < -1e-6).sum()),
        "ties": int((np.abs(diff) <= 1e-6).sum()),
        "losses": int((diff > 1e-6).sum()),
        "d_z": diff.mean() / sd if sd > 0 else 0.0,
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def heldout_k(wide: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame | None:
    ks = sorted((int(c.rsplit("_", 1)[1]) for c in wide.columns if c.startswith("DMS_split_at_")))
    if not ks or "DMS_split_0.5" not in wide.columns:
        return None
    select = wide.loc[wide.index.isin(SELECT_SEEDS)]
    evaluate = wide.loc[wide.index.isin(EVAL_SEEDS)]
    means = {k: select[f"DMS_split_at_{k}"].mean() for k in ks}
    best = min(means, key=means.get)
    label = f"DMS_split_at_{best}"
    rows = []
    for other in ("DMS_split_0.5", "DMS", "Attentive"):
        if other not in wide.columns:
            continue
        pair = evaluate[[label, other]].dropna()
        row = paired(pair[label].to_numpy(), pair[other].to_numpy(), rng)
        row.update({"selected_k": best, "a": label, "b": other})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_final_costs.csv"))
    if args.benchmarks != ["all"]:
        files = [f for f in files if f.name.replace("_final_costs.csv", "") in args.benchmarks]
    rng = np.random.default_rng(0)
    for path in files:
        benchmark = path.name.replace("_final_costs.csv", "")
        df = pd.read_csv(path)
        wide = df.pivot(index="seed", columns="algorithm", values="final_cost")

        rows = []
        for a, b in PAIRS:
            if a not in wide.columns or b not in wide.columns:
                continue
            pair = wide[[a, b]].dropna()
            if len(pair) < 2:
                continue
            row = paired(pair[a].to_numpy(), pair[b].to_numpy(), rng)
            row.update({"a": a, "b": b})
            rows.append(row)
        comp = pd.DataFrame(rows)
        comp["holm_p"] = holm(list(comp["wilcoxon_p"]))
        cols = [
            "a",
            "b",
            "n",
            "mean_a",
            "mean_b",
            "mean_diff",
            "wilcoxon_p",
            "holm_p",
            "ci_low",
            "ci_high",
            "wins",
            "ties",
            "losses",
            "d_z",
        ]
        comp = comp[cols]
        comp.to_csv(data_dir / f"{benchmark}_key_comparisons.csv", index=False, float_format="%.6g")
        print(f"\n=== {benchmark}: key comparisons (a - b, final cost) ===")
        print(comp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

        held = heldout_k(wide, rng)
        if held is not None:
            cols = [
                "selected_k",
                "a",
                "b",
                "n",
                "mean_a",
                "mean_b",
                "mean_diff",
                "wilcoxon_p",
                "ci_low",
                "ci_high",
                "wins",
                "ties",
                "losses",
                "d_z",
            ]
            held = held[cols]
            held.to_csv(data_dir / f"{benchmark}_heldout_k.csv", index=False, float_format="%.6g")
            print(f"\n=== {benchmark}: delayed split, K selected on seeds 0-24, evaluated on 25-49 ===")
            print(held.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
