"""Plot mean solution-cost curves for the AAAI experiments.

For each benchmark, two PDFs (in --plots-dir):
  {benchmark}_cost.pdf      mean per-iteration cost across the problem instances
  {benchmark}_anytime.pdf   mean anytime (best-so-far) cost

Curves shorter than the horizon (the merge variants stop at the merge point)
are extended with their last value. The Optimal mean (over instances where
branch and bound completed) is drawn as a horizontal reference line.

Example:
  uv run python experiments/aaai/code/plot_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.plot_helpers import remove_frame  # noqa: E402

LABELS = {
    "MS": "MS",
    "DMS": "DMS",
    "DMS_split_0.5": "DMS + split 0.5",
    "DMS_split_0.4_0.6": "DMS + split 0.4-0.6",
    "DMS_split_at_50": "DMS + split@50",
    "DMS_split_at_100": "DMS + split@100",
    "DMS_split_at_300": "DMS + split@300",
    "DMS_split_at_500": "DMS + split@500",
    "DMS_split_at_1000": "DMS + split@1000",
    "DMS_split_at_1500": "DMS + split@1500",
    "MS_split_0.5": "MS + split 0.5",
    "MS_split_MGM_200": "MS + split + MGM@200",
    "MS_split_opt_200": "MS + split + optimal@200",
    "Attentive": "Attentive (DABP)",
    "Attentive_NoSplit": "Attentive (DABP, no split)",
}
ORDER = list(LABELS)

# DABP variants whose curve is stretched onto the wall-clock axis, mapped to the
# per-iteration time-ratio column written by time_dabp.py.
STRETCH_RATIO_COLUMNS = {
    "Attentive": "ratio",
    "Attentive_NoSplit": "nosplit_ratio",
}


def load_ratios(data_dir: Path) -> dict[str, dict[str, float]]:
    """benchmark -> {algorithm -> DABP/DMS per-iteration time ratio}.

    Written by time_dabp.py; used to stretch each DABP curve onto a
    wall-clock-equivalent x-axis. A missing file, missing column, or non-finite
    ratio falls back to 1.0 (no stretch) for that algorithm.
    """
    path = data_dir / "dabp_timing.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    ratios: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        bench = str(row["benchmark"])
        for algorithm, column in STRETCH_RATIO_COLUMNS.items():
            if column not in row:
                continue
            try:
                ratio = float(row[column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(ratio) and ratio > 0:
                ratios.setdefault(bench, {})[algorithm] = ratio
    return ratios


def padded_runs(group: pd.DataFrame, horizon: int) -> np.ndarray:
    """rows = runs, columns = iterations 0..horizon-1, padded with last value."""
    runs = []
    for _, run in group.groupby("seed"):
        costs = run.sort_values("iteration")["cost"].to_numpy(dtype=float)
        if len(costs) < horizon:
            costs = np.concatenate(
                [costs, np.full(horizon - len(costs), costs[-1])]
            )
        runs.append(costs[:horizon])
    return np.asarray(runs)


def plot_benchmark(
    benchmark: str, data_dir: Path, plots_dir: Path, ratios: dict[str, dict[str, float]]
) -> None:
    raw = pd.read_csv(data_dir / f"{benchmark}_raw_costs.csv")
    final = pd.read_csv(data_dir / f"{benchmark}_final_costs.csv")
    horizon = int(raw["iteration"].max()) + 1
    bench_ratios = ratios.get(benchmark, {})

    optimal = final.loc[final["algorithm"] == "Optimal", "final_cost"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    for algorithm in ORDER:
        group = raw[raw["algorithm"] == algorithm]
        if group.empty:
            continue
        mean = padded_runs(group, horizon).mean(axis=0)
        # stretch each DABP variant onto a wall-clock-equivalent axis: iteration
        # k costs `stretch` plain-BP iterations, so plot it at x = stretch * k
        stretch = bench_ratios.get(algorithm, 1.0)
        xs = stretch * np.arange(len(mean))
        label = LABELS.get(algorithm, algorithm)
        if stretch != 1.0:
            label = f"{label} x{stretch:.1f}"
        ax.plot(xs, mean, lw=1.4, label=label)
    if len(optimal):
        ax.axhline(
            optimal.mean(),
            color="black",
            ls="--",
            lw=1.6,
            label=f"Optimal (n={len(optimal)})",
        )
    # keep the standard horizon so DABP is read as "where it reaches within the
    # wall-clock budget of `horizon` plain-BP iterations"
    ax.set_xlim(0, horizon - 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean solution cost")
    remove_frame(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    out = plots_dir / f"{benchmark}_cost.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    parser.add_argument(
        "--plots-dir", default=str(Path(__file__).resolve().parents[1] / "plots")
    )
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*_raw_costs.csv"))
    if args.benchmarks != ["all"]:
        files = [f for f in files if f.name.replace("_raw_costs.csv", "") in args.benchmarks]
    if not files:
        raise SystemExit(f"no *_raw_costs.csv files found in {data_dir}")

    ratios = load_ratios(data_dir)
    for path in files:
        plot_benchmark(
            path.name.replace("_raw_costs.csv", ""), data_dir, plots_dir, ratios
        )


if __name__ == "__main__":
    main()
