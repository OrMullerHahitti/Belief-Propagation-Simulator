"""Settling statistics from the per-iteration cost curves.

For every benchmark and algorithm: how many of the 50 runs end with a constant
cost over the final ``--quiet`` iterations, and at which iteration the cost
last changed (the settling iteration) for the runs that did.

Writes {benchmark}_settling.csv next to the inputs:
  algorithm, n, n_settled, settle_median, settle_mean

Example:
  uv run python experiments/aaai/code/settling.py --benchmarks random_dense
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def settle_iteration(costs: np.ndarray, quiet: int) -> int | None:
    """last iteration at which the cost changed, or None if the final
    ``quiet`` iterations are not constant."""
    changes = np.flatnonzero(np.abs(np.diff(costs)) > 1e-6)
    last = int(changes[-1]) + 1 if len(changes) else 0
    if len(costs) - last < quiet:
        return None
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument("--quiet", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_raw_costs.csv"))
    if args.benchmarks != ["all"]:
        files = [f for f in files if f.name.replace("_raw_costs.csv", "") in args.benchmarks]
    for path in files:
        benchmark = path.name.replace("_raw_costs.csv", "")
        raw = pd.read_csv(path)
        rows = []
        for algorithm, group in raw.groupby("algorithm", sort=False):
            settles = []
            n = 0
            for _, run in group.groupby("seed"):
                costs = run.sort_values("iteration")["cost"].to_numpy()
                # merge rows hold only the pre-merge curve plus one value
                if len(costs) < args.quiet:
                    continue
                n += 1
                s = settle_iteration(costs, args.quiet)
                if s is not None:
                    settles.append(s)
            if n == 0:
                continue
            rows.append(
                {
                    "algorithm": algorithm,
                    "n": n,
                    "n_settled": len(settles),
                    "settle_median": float(np.median(settles)) if settles else np.nan,
                    "settle_mean": float(np.mean(settles)) if settles else np.nan,
                }
            )
        out = pd.DataFrame(rows).sort_values("settle_median")
        out_path = data_dir / f"{benchmark}_settling.csv"
        out.to_csv(out_path, index=False, float_format="%.1f")
        print(f"\n=== {benchmark} (quiet window {args.quiet}) ===")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
