"""Plot mean solution-cost curves for the AAAI experiments.

For each benchmark, PDFs are written to --plots-dir:
  {benchmark}_cost.pdf       mean per-iteration cost across the problem instances
  {benchmark}_cost_zoom.pdf  close-up of the crowded lower-cost curve cluster,
                             written only when a separated cluster is detected

Curves shorter than the horizon (the merge variants stop at the merge point)
are extended with their last value. The Optimal mean (over instances where
branch and bound completed) is drawn as a horizontal reference line.

Example:
  uv run python experiments/aaai/code/plot_results.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    "MS_split_MGM_inverted_200": "MS + split + MGM inverted@200",
    "MS_split_opt_200": "MS + split + optimal@200",
    "Attentive": "Attentive (DABP)",
    "Attentive_NoSplit": "Attentive (DABP, no split)",
}
ORDER = list(LABELS)
COLORS = {
    "MS": "#8c8c8c",
    "DMS": "#0072B2",
    "DMS_split_0.5": "#009E73",
    "DMS_split_0.4_0.6": "#56B4E9",
    "DMS_split_at_50": "#E69F00",
    "DMS_split_at_100": "#D55E00",
    "DMS_split_at_300": "#CC79A7",
    "DMS_split_at_500": "#332288",
    "DMS_split_at_1000": "#AA4499",
    "DMS_split_at_1500": "#88CCEE",
    "MS_split_0.5": "#44AA99",
    "MS_split_MGM_200": "#117733",
    "MS_split_MGM_inverted_200": "#DDCC77",
    "MS_split_opt_200": "#882255",
    "Attentive": "#000000",
    "Attentive_NoSplit": "#BBBBBB",
}

# DABP variants whose curve is stretched onto the wall-clock axis, mapped to the
# per-iteration time-ratio column written by time_dabp.py.
STRETCH_RATIO_COLUMNS = {
    "Attentive": "ratio",
    "Attentive_NoSplit": "nosplit_ratio",
}
# Binary-menu merges whose single merged-cost point is stretched onto the
# wall-clock axis (drawn at x = merge_at + ratio), mapped to the per-iteration
# time-ratio column written by time_merges.py. Unlike DABP these are a one-shot
# merge, so only the final point shifts -- not the whole curve.
MERGE_STRETCH_COLUMNS = {
    "MS_split_MGM_200": "mgm_ratio",
    "MS_split_MGM_inverted_200": "mgm_ratio",
    "MS_split_opt_200": "bnb_ratio",
}
MERGE_STRETCH_ALGOS = set(MERGE_STRETCH_COLUMNS)
ZOOM_START_FRACTION = 0.75
ZOOM_MIN_CURVES = 5
ZOOM_MIN_OMITTED_CURVES = 2
ZOOM_GAP_MULTIPLIER = 4.0
ZOOM_GAP_RELATIVE = 0.03


@dataclass(frozen=True)
class CostCurve:
    """Mean plotted cost curve for one algorithm."""

    algorithm: str
    xs: np.ndarray
    ys: np.ndarray
    label: str


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


def load_merge_ratios(data_dir: Path) -> dict[str, dict[str, tuple[float, bool]]]:
    """benchmark -> {merge algorithm -> (stretch_ratio, complete)}.

    Written by time_merges.py; used to place each binary-menu merge's single
    merged-cost point at x = merge_at + ratio. ``complete`` is False when the
    optimal B&B hit its time cap (e.g. random_dense), which is flagged in the
    legend. A missing file/column or non-finite ratio means no stretch.
    """
    path = data_dir / "merge_timing.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    ratios: dict[str, dict[str, tuple[float, bool]]] = {}
    for _, row in df.iterrows():
        bench = str(row["benchmark"])
        complete = bool(int(row["bnb_complete"])) if "bnb_complete" in row else True
        for algorithm, column in MERGE_STRETCH_COLUMNS.items():
            if column not in row:
                continue
            try:
                ratio = float(row[column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(ratio) and ratio > 0:
                done = complete if algorithm == "MS_split_opt_200" else True
                ratios.setdefault(bench, {})[algorithm] = (ratio, done)
    return ratios


def padded_runs(group: pd.DataFrame, horizon: int) -> np.ndarray:
    """rows = runs, columns = iterations 0..horizon-1, padded with last value."""
    runs = []
    for _, run in group.groupby("seed"):
        costs = run.sort_values("iteration")["cost"].to_numpy(dtype=float)
        if len(costs) < horizon:
            costs = np.concatenate([costs, np.full(horizon - len(costs), costs[-1])])
        runs.append(costs[:horizon])
    return np.asarray(runs)


def _merge_curve(
    algorithm: str,
    group: pd.DataFrame,
    mean: np.ndarray,
    horizon: int,
    ratio: float,
    complete: bool,
) -> CostCurve:
    """Hold-flat-then-drop curve for a binary-menu merge stretched by its
    wall-clock ratio R (in DMS-equivalent iterations): the pre-merge cost is held
    flat from merge_at to merge_at+R, then drops to the merged cost at merge_at+R.
    If the drop lands beyond the horizon (e.g. B&B on random_dense), the merge
    never 'completes' on-screen and the curve stays at the pre-merge cost.
    """
    merge_at = int(group["iteration"].max())
    pre = mean[:merge_at]
    merged = float(mean[merge_at])
    drop_x = merge_at + ratio
    label = LABELS.get(algorithm, algorithm)
    if drop_x <= horizon - 1:
        xs = np.array(
            list(range(merge_at)) + [merge_at, drop_x, drop_x, horizon - 1],
            dtype=float,
        )
        ys = np.array(list(pre) + [pre[-1], pre[-1], merged, merged], dtype=float)
        label = f"{label} +{ratio:.0f}it"
    else:
        # merge completes past the plotted horizon: hold the pre-merge cost flat
        xs = np.array(list(range(merge_at)) + [horizon - 1], dtype=float)
        ys = np.array(list(pre) + [pre[-1]], dtype=float)
        label = f"{label} (merge @{drop_x:.0f}, off-axis)"
    if not complete:
        label += " *capped"
    return CostCurve(algorithm=algorithm, xs=xs, ys=ys, label=label)


def mean_cost_curves(
    raw: pd.DataFrame,
    horizon: int,
    bench_ratios: dict[str, float],
    bench_merge_ratios: dict[str, tuple[float, bool]] | None = None,
) -> list[CostCurve]:
    """Build the actual x/y curves used in the paper plot from raw CSV rows."""
    bench_merge_ratios = bench_merge_ratios or {}
    curves = []
    for algorithm in ORDER:
        group = raw[raw["algorithm"] == algorithm]
        if group.empty:
            continue
        mean = padded_runs(group, horizon).mean(axis=0)
        merge = bench_merge_ratios.get(algorithm)
        if algorithm in MERGE_STRETCH_ALGOS and merge is not None:
            curves.append(_merge_curve(algorithm, group, mean, horizon, *merge))
            continue
        stretch = bench_ratios.get(algorithm, 1.0)
        xs = stretch * np.arange(len(mean))
        label = LABELS.get(algorithm, algorithm)
        if stretch != 1.0:
            label = f"{label} x{stretch:.1f}"
        curves.append(CostCurve(algorithm=algorithm, xs=xs, ys=mean, label=label))
    return curves


def _tail_median(curve: CostCurve, x_min: int, x_max: int) -> float | None:
    mask = (curve.xs >= x_min) & (curve.xs <= x_max)
    if not np.any(mask):
        return None
    return float(np.nanmedian(curve.ys[mask]))


def select_zoom_curves(
    curves: list[CostCurve],
    horizon: int,
    *,
    min_curves: int = ZOOM_MIN_CURVES,
    min_omitted: int = ZOOM_MIN_OMITTED_CURVES,
) -> list[CostCurve]:
    """Select the crowded lower-cost cluster for a tail-window zoom plot.

    The split is derived from visible tail-window medians, so stretched DABP
    curves are clustered according to where they actually appear on the plot.
    """
    x_min = int(horizon * ZOOM_START_FRACTION)
    x_max = horizon - 1
    tail_values = []
    for curve in curves:
        value = _tail_median(curve, x_min, x_max)
        if value is not None and np.isfinite(value):
            tail_values.append((curve, value))

    if len(tail_values) < min_curves + min_omitted:
        return []

    ordered = sorted(tail_values, key=lambda item: item[1])
    values = np.asarray([value for _, value in ordered], dtype=float)
    gaps = np.diff(values)
    positive_gaps = gaps[gaps > 0]
    if len(positive_gaps) == 0:
        return []

    full_span = values[-1] - values[0]
    threshold = max(
        ZOOM_GAP_MULTIPLIER * float(np.median(positive_gaps)),
        ZOOM_GAP_RELATIVE * float(full_span),
    )
    for idx, gap in enumerate(gaps):
        prefix_len = idx + 1
        omitted = len(ordered) - prefix_len
        if prefix_len >= min_curves and omitted >= min_omitted and gap > threshold:
            selected = {curve.algorithm for curve, _ in ordered[:prefix_len]}
            return [curve for curve in curves if curve.algorithm in selected]
    return []


def _draw_curves(ax: plt.Axes, curves: list[CostCurve]) -> None:
    for curve in curves:
        ax.plot(
            curve.xs,
            curve.ys,
            color=COLORS.get(curve.algorithm),
            lw=1.4,
            label=curve.label,
        )


def _maybe_draw_optimal(ax: plt.Axes, optimal: pd.Series) -> None:
    if len(optimal):
        ax.axhline(
            optimal.mean(),
            color="black",
            ls="--",
            lw=1.6,
            label=f"Optimal (n={len(optimal)})",
        )


def _set_zoom_ylim(
    ax: plt.Axes, curves: list[CostCurve], x_min: int, x_max: int
) -> None:
    ys = []
    for curve in curves:
        mask = (curve.xs >= x_min) & (curve.xs <= x_max)
        if np.any(mask):
            ys.append(curve.ys[mask])
    if not ys:
        return
    values = np.concatenate(ys)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.05
    pad = 0.08 * span
    ax.set_ylim(lo - pad, hi + pad)


def plot_zoom_benchmark(
    benchmark: str,
    plots_dir: Path,
    horizon: int,
    curves: list[CostCurve],
) -> None:
    zoom_curves = select_zoom_curves(curves, horizon)
    if not zoom_curves:
        return

    x_min = int(horizon * ZOOM_START_FRACTION)
    x_max = horizon - 1
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_curves(ax, zoom_curves)
    ax.set_xlim(x_min, x_max)
    _set_zoom_ylim(ax, zoom_curves, x_min, x_max)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean solution cost")
    remove_frame(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    out = plots_dir / f"{benchmark}_cost_zoom.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_benchmark(
    benchmark: str,
    data_dir: Path,
    plots_dir: Path,
    ratios: dict[str, dict[str, float]],
    merge_ratios: dict[str, dict[str, tuple[float, bool]]] | None = None,
) -> None:
    raw = pd.read_csv(data_dir / f"{benchmark}_raw_costs.csv")
    final = pd.read_csv(data_dir / f"{benchmark}_final_costs.csv")
    horizon = int(raw["iteration"].max()) + 1
    bench_ratios = ratios.get(benchmark, {})
    bench_merge_ratios = (merge_ratios or {}).get(benchmark, {})

    optimal = final.loc[final["algorithm"] == "Optimal", "final_cost"].dropna()
    curves = mean_cost_curves(raw, horizon, bench_ratios, bench_merge_ratios)

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_curves(ax, curves)
    _maybe_draw_optimal(ax, optimal)
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
    plot_zoom_benchmark(benchmark, plots_dir, horizon, curves)


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

    ratios = load_ratios(data_dir)
    merge_ratios = load_merge_ratios(data_dir)
    for path in files:
        plot_benchmark(
            path.name.replace("_raw_costs.csv", ""),
            data_dir,
            plots_dir,
            ratios,
            merge_ratios,
        )


if __name__ == "__main__":
    main()
