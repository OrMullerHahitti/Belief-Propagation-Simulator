"""Black-and-white mean cost-per-iteration plots for the AAAI experiments.

One PDF per benchmark (in --plots-dir): {benchmark}_cost_bw.pdf with the mean
per-iteration cost of every algorithm across the problem instances. Curves are
distinguished by line style, gray level and staggered hollow markers (print
friendly). Curves shorter than the horizon (the merge variants stop at the
merge point) are extended with their last value; the Optimal mean is a thin
horizontal reference line.

Example:
  uv run python experiments/aaai/code/plot_results_bw.py
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

from plot_results import LABELS, padded_runs  # noqa: E402
from utils.plot_helpers import remove_frame  # noqa: E402

# label -> (color, linestyle, marker); markers are hollow and staggered so
# overlapping curves stay readable in black and white
STYLES = {
    "DMS": ("black", "-", None),
    "DMS_split_0.5": ("black", "--", None),
    "DMS_split_0.4_0.6": ("black", ":", None),
    "DMS_split_at_50": ("black", "-", "o"),
    "DMS_split_at_100": ("black", "-", "s"),
    "DMS_split_at_300": ("black", "-", "^"),
    "DMS_split_at_500": ("black", "-", "v"),
    "DMS_split_at_1000": ("black", "-", "D"),
    "MS_split_0.5": ("0.45", "-", None),
    "MS_split_MGM_200": ("0.45", "--", "x"),
    "MS_split_MGM_inverted_200": ("0.45", ":", "d"),
    "MS_split_opt_200": ("0.45", "-.", "+"),
    "Attentive": ("0.0", "-", "*"),
}


def plot_benchmark(benchmark: str, data_dir: Path, plots_dir: Path) -> None:
    raw = pd.read_csv(data_dir / f"{benchmark}_raw_costs.csv")
    final = pd.read_csv(data_dir / f"{benchmark}_final_costs.csv")
    horizon = int(raw["iteration"].max()) + 1

    optimal = final.loc[final["algorithm"] == "Optimal", "final_cost"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    mark_step = max(horizon // 12, 1)
    for idx, (algorithm, (color, ls, marker)) in enumerate(STYLES.items()):
        group = raw[raw["algorithm"] == algorithm]
        if group.empty:
            continue
        runs = padded_runs(group, horizon)
        ax.plot(
            np.arange(horizon),
            runs.mean(axis=0),
            color=color,
            ls=ls,
            lw=1.3,
            marker=marker,
            markevery=(idx * mark_step // len(STYLES), mark_step),
            markersize=5,
            markerfacecolor="white",
            label=LABELS.get(algorithm, algorithm),
        )
    if len(optimal):
        ax.axhline(
            optimal.mean(),
            color="black",
            ls=(0, (1, 1)),
            lw=0.9,
            label=f"Optimal (n={len(optimal)})",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean solution cost")
    remove_frame(ax)
    # headroom so the legend never sits on the flat top curves
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    out = plots_dir / f"{benchmark}_cost_bw.pdf"
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
        files = [
            f for f in files if f.name.replace("_raw_costs.csv", "") in args.benchmarks
        ]
    if not files:
        raise SystemExit(f"no *_raw_costs.csv files found in {data_dir}")

    for path in files:
        plot_benchmark(path.name.replace("_raw_costs.csv", ""), data_dir, plots_dir)


if __name__ == "__main__":
    main()
