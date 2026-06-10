"""Aggregate the AAAI experiment results: means and statistical significance.

For each benchmark this writes, next to the inputs:
  {benchmark}_summary.csv       per-algorithm mean/std of final and anytime cost
  {benchmark}_significance.csv  pairwise paired t-tests and Wilcoxon signed-rank
                                tests (final and anytime metrics), following the
                                paired t-test methodology of Cohen et al. (2020)

Pairs are computed on the seeds both algorithms solved (Optimal rows are NaN
when branch and bound hit its time limit and are excluded pairwise).

Example:
  uv run python experiments/aaai/code/analyze_results.py
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("final_cost", "anytime_cost")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, group in df.groupby("algorithm", sort=False):
        row = {"algorithm": algorithm}
        for metric in METRICS:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row["n"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("final_cost_mean").reset_index(drop=True)


def significance(df: pd.DataFrame) -> pd.DataFrame:
    algorithms = list(df["algorithm"].unique())
    rows = []
    for metric in METRICS:
        wide = df.pivot(index="seed", columns="algorithm", values=metric)
        for a, b in itertools.combinations(algorithms, 2):
            paired = wide[[a, b]].dropna()
            n = len(paired)
            if n < 2:
                continue
            x, y = paired[a].to_numpy(), paired[b].to_numpy()
            diff = x - y
            t_stat, t_p = stats.ttest_rel(x, y)
            if np.allclose(diff, 0.0):
                w_p = 1.0
            else:
                _, w_p = stats.wilcoxon(x, y)
            rows.append(
                {
                    "metric": metric,
                    "algorithm_a": a,
                    "algorithm_b": b,
                    "n": n,
                    "mean_a": x.mean(),
                    "mean_b": y.mean(),
                    "mean_diff": diff.mean(),
                    "t_stat": t_stat,
                    "t_pvalue": t_p,
                    "wilcoxon_pvalue": w_p,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_final_costs.csv"))
    if args.benchmarks != ["all"]:
        files = [f for f in files if f.name.replace("_final_costs.csv", "") in args.benchmarks]
    if not files:
        raise SystemExit(f"no *_final_costs.csv files found in {data_dir}")

    for path in files:
        benchmark = path.name.replace("_final_costs.csv", "")
        df = pd.read_csv(path)

        summary = summarize(df)
        summary_path = data_dir / f"{benchmark}_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.4f")

        sig = significance(df)
        sig_path = data_dir / f"{benchmark}_significance.csv"
        sig.to_csv(sig_path, index=False, float_format="%.6g")

        print(f"\n=== {benchmark} (n problems per algorithm in 'n') ===")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"wrote {summary_path.name}, {sig_path.name}")


if __name__ == "__main__":
    main()
