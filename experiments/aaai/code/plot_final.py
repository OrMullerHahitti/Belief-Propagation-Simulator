"""Final AAAI plots: cost curves, zooms, ternary plots, and the tail-threshold
figure, all written under experiments/aaai/final_plots.

Differences from plot_results.py (the paper's final conventions):
  - x-axis counts iterations, not engine steps: one step = 2 iterations, so
    every plotted x is doubled and split/merge labels are renamed to match
    (@300 -> @600, merges @200 -> @400)
  - MS_split_opt is drawn as recorded (no wall-clock stretch): the merged cost
    drops at the merge point itself
  - late splits are reduced to @300 and @1000 (plus the best one when it is
    neither of those)
  - zoom plots exclude all DABP (Attentive) variants
  - one legend per figure, larger font, short labels (no timing suffixes),
    undamped variants listed before damped ones, no titles

Binary benchmarks are read from data_cuda into final_plots/; the three ternary
benchmarks are read from ternary_data into final_plots/<benchmark>/ each.

Example:
  uv run python experiments/aaai/code/plot_final.py
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

# one engine step is counted as 2 iterations on the paper's x-axis
SCALE = 2

# legend labels use iteration counts on the doubled axis
LABELS = {
    "MS": "MS",
    "MS_split_0.5": "MS s=.5",
    "MS_split_MGM_200": f"MGM@{200 * SCALE}",
    "MS_split_opt_200": f"Opt merge@{200 * SCALE}",
    "DMS": "DMS",
    "DMS_split_0.5": "DMS s=.5",
    "DMS_0.5_split_0.5": "DMS d=.5 s=.5",
    "DMS_split_0.4_0.6": "DMS s=.4-.6",
    "DMS_split_at_50": f"DMS @{50 * SCALE}",
    "DMS_split_at_100": f"DMS @{100 * SCALE}",
    "DMS_split_at_300": f"DMS @{300 * SCALE}",
    "DMS_split_at_500": f"DMS @{500 * SCALE}",
    "DMS_split_at_1000": f"DMS @{1000 * SCALE}",
    "DMS_split_at_1500": f"DMS @{1500 * SCALE}",
    "Attentive": "DABP",
    "Attentive_NoSplit": "DABP no-split",
}
# undamped variants first, then damped, then DABP (legend order = draw order)
ORDER = list(LABELS)
COLORS = {
    "MS": "#8c8c8c",
    "DMS": "#0072B2",
    "DMS_split_0.5": "#009E73",
    "DMS_0.5_split_0.5": "#999933",
    "DMS_split_0.4_0.6": "#56B4E9",
    "DMS_split_at_50": "#E69F00",
    "DMS_split_at_100": "#D55E00",
    "DMS_split_at_300": "#CC79A7",
    "DMS_split_at_500": "#332288",
    "DMS_split_at_1000": "#AA4499",
    "DMS_split_at_1500": "#88CCEE",
    "MS_split_0.5": "#44AA99",
    "MS_split_MGM_200": "#117733",
    "MS_split_opt_200": "#882255",
    "Attentive": "#000000",
    "Attentive_NoSplit": "#BBBBBB",
}

# DABP variants whose curve is stretched onto the wall-clock axis
STRETCH_RATIO_COLUMNS = {
    "Attentive": "ratio",
    "Attentive_NoSplit": "nosplit_ratio",
}
# only the MGM merges keep the wall-clock stretch; MS_split_opt is drawn as
# recorded, with the merged cost dropping at the merge point itself
MERGE_STRETCH_COLUMNS = {
    "MS_split_MGM_200": "mgm_ratio",
}
MERGE_STRETCH_ALGOS = set(MERGE_STRETCH_COLUMNS)

LATE_SPLIT_PREFIX = "DMS_split_at_"
LATE_SPLIT_KEEP = {"DMS_split_at_300", "DMS_split_at_1000"}

DABP_ALGOS = {"Attentive", "Attentive_NoSplit"}

ZOOM_START_FRACTION = 0.75
ZOOM_MIN_CURVES = 5
ZOOM_MIN_OMITTED_CURVES = 2
ZOOM_GAP_MULTIPLIER = 4.0
ZOOM_GAP_RELATIVE = 0.03
LEGEND_RIGHT_MARGIN = 0.68
LEGEND_KWARGS = {
    "fontsize": 13,
    "frameon": False,
    "loc": "center left",
    "bbox_to_anchor": (1.02, 0.5),
    "borderaxespad": 0.0,
    "columnspacing": 1.0,
    "handlelength": 2.2,
    "handletextpad": 0.5,
    "labelspacing": 0.55,
}

TERNARY_BENCHMARKS = [
    "meeting_scheduling_ternary",
    "random_dense_ternary",
    "random_sparse_ternary",
]


@dataclass(frozen=True)
class CostCurve:
    """Mean plotted cost curve for one algorithm (xs already on the doubled axis)."""

    algorithm: str
    xs: np.ndarray
    ys: np.ndarray
    label: str


def load_ratios(data_dir: Path) -> dict[str, dict[str, float]]:
    """benchmark -> {algorithm -> DABP/DMS per-iteration time ratio}."""
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


def load_merge_ratios(data_dir: Path) -> dict[str, dict[str, float]]:
    """benchmark -> {MGM merge algorithm -> wall-clock stretch ratio}."""
    path = data_dir / "merge_timing.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    ratios: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        bench = str(row["benchmark"])
        for algorithm, column in MERGE_STRETCH_COLUMNS.items():
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
    """rows = runs, columns = steps 0..horizon-1, padded with last value."""
    runs = []
    for _, run in group.groupby("seed"):
        costs = run.sort_values("iteration")["cost"].to_numpy(dtype=float)
        if len(costs) < horizon:
            costs = np.concatenate([costs, np.full(horizon - len(costs), costs[-1])])
        runs.append(costs[:horizon])
    return np.asarray(runs)


def select_algorithms(raw: pd.DataFrame) -> list[str]:
    """ORDER filtered to present algorithms, with late splits reduced to @300,
    @1000, and the best-performing one when it is neither of those."""
    present = [a for a in ORDER if a in set(raw["algorithm"])]
    lates = [a for a in present if a.startswith(LATE_SPLIT_PREFIX)]
    if not lates:
        return present
    final_means = {
        a: raw[raw["algorithm"] == a].groupby("seed")["cost"].last().mean()
        for a in lates
    }
    best = min(final_means, key=final_means.get)
    keep = LATE_SPLIT_KEEP | {best}
    return [a for a in present if not a.startswith(LATE_SPLIT_PREFIX) or a in keep]


def _merge_curve(
    algorithm: str,
    group: pd.DataFrame,
    mean: np.ndarray,
    horizon: int,
    ratio: float,
) -> CostCurve:
    """Hold-flat-then-drop curve for an MGM merge stretched by its wall-clock
    ratio (in DMS-equivalent steps), drawn on the doubled iteration axis."""
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
    else:
        # merge completes past the plotted horizon: hold the pre-merge cost flat
        xs = np.array(list(range(merge_at)) + [horizon - 1], dtype=float)
        ys = np.array(list(pre) + [pre[-1]], dtype=float)
    return CostCurve(algorithm=algorithm, xs=SCALE * xs, ys=ys, label=label)


def mean_cost_curves(
    raw: pd.DataFrame,
    algorithms: list[str],
    horizon: int,
    bench_ratios: dict[str, float],
    bench_merge_ratios: dict[str, float],
) -> list[CostCurve]:
    """Build the x/y curves for the final plots, xs on the doubled axis."""
    curves = []
    for algorithm in algorithms:
        group = raw[raw["algorithm"] == algorithm]
        if group.empty:
            continue
        mean = padded_runs(group, horizon).mean(axis=0)
        label = LABELS.get(algorithm, algorithm)
        ratio = bench_merge_ratios.get(algorithm)
        if algorithm in MERGE_STRETCH_ALGOS and ratio is not None:
            curves.append(_merge_curve(algorithm, group, mean, horizon, ratio))
            continue
        stretch = bench_ratios.get(algorithm, 1.0)
        xs = SCALE * stretch * np.arange(len(mean))
        curves.append(CostCurve(algorithm=algorithm, xs=xs, ys=mean, label=label))
    return curves


def _tail_median(curve: CostCurve, x_min: float, x_max: float) -> float | None:
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
    """Select the crowded lower-cost cluster for a tail-window zoom plot."""
    x_min = SCALE * int(horizon * ZOOM_START_FRACTION)
    x_max = SCALE * (horizon - 1)
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


def _legend_outside_right(ax: plt.Axes) -> None:
    ax.legend(**LEGEND_KWARGS)


def _save_plot_with_legend(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout(rect=(0.0, 0.0, LEGEND_RIGHT_MARGIN, 1.0))
    fig.savefig(out, dpi=150, bbox_inches="tight")


def _maybe_draw_optimal(ax: plt.Axes, optimal: pd.Series) -> None:
    if len(optimal):
        ax.axhline(optimal.mean(), color="black", ls="--", lw=1.6, label="Optimal")


def _set_zoom_ylim(
    ax: plt.Axes, curves: list[CostCurve], x_min: float, x_max: float
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
    # zoom plots exclude all DABP variants
    candidates = [c for c in curves if c.algorithm not in DABP_ALGOS]
    zoom_curves = select_zoom_curves(candidates, horizon)
    if not zoom_curves:
        return

    x_min = SCALE * int(horizon * ZOOM_START_FRACTION)
    x_max = SCALE * (horizon - 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_curves(ax, zoom_curves)
    ax.set_xlim(x_min, x_max)
    _set_zoom_ylim(ax, zoom_curves, x_min, x_max)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean solution cost")
    remove_frame(ax)
    _legend_outside_right(ax)
    out = plots_dir / f"{benchmark}_cost_zoom.pdf"
    _save_plot_with_legend(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_benchmark(
    benchmark: str,
    data_dir: Path,
    plots_dir: Path,
    ratios: dict[str, dict[str, float]],
    merge_ratios: dict[str, dict[str, float]],
) -> None:
    raw = pd.read_csv(data_dir / f"{benchmark}_raw_costs.csv")
    final = pd.read_csv(data_dir / f"{benchmark}_final_costs.csv")
    horizon = int(raw["iteration"].max()) + 1
    algorithms = select_algorithms(raw)

    optimal = final.loc[final["algorithm"] == "Optimal", "final_cost"].dropna()
    curves = mean_cost_curves(
        raw,
        algorithms,
        horizon,
        ratios.get(benchmark, {}),
        merge_ratios.get(benchmark, {}),
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_curves(ax, curves)
    _maybe_draw_optimal(ax, optimal)
    # keep the standard horizon so DABP is read as "where it reaches within the
    # wall-clock budget of `horizon` plain-BP steps"
    ax.set_xlim(0, SCALE * (horizon - 1))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean solution cost")
    remove_frame(ax)
    _legend_outside_right(ax)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"{benchmark}_cost.pdf"
    _save_plot_with_legend(fig, out)
    plt.close(fig)
    print(f"wrote {out}")
    plot_zoom_benchmark(benchmark, plots_dir, horizon, curves)


# ---------------------------------------------------------------------------
# tail-threshold figure (three-variable chain), restyled to the cost-plot
# conventions: same MS/DMS labels and colors, one big legend, no title
# ---------------------------------------------------------------------------

TAIL_THRESHOLD = 8.0
TAIL_ITERATIONS = 25
# long enough for DMS s=.5 (lambda=.9) to visibly flatten at the top
# (it crosses k=8 at ~64 and reaches ~99.9% of the plateau by ~152)
TAIL_ITERATIONS_LONG = 175

# (key, restyled label, split, damping_factor, colour, linestyle) - undamped first
TAIL_SCENARIOS = [
    ("MS", "MS", False, 0.0, "#8c8c8c", "-"),
    ("MS s=.5", "MS s=.5", True, 0.0, "#44AA99", "-"),
    ("DMS (λ=.5)", "DMS (λ=.5)", False, 0.5, "#0072B2", "-"),
    ("DMS s=.5 (λ=.5)", "DMS s=.5 (λ=.5)", True, 0.5, "#009E73", "-"),
    ("DMS (λ=.9)", "DMS (λ=.9)", False, 0.9, "#0072B2", "--"),
    ("DMS s=.5 (λ=.9)", "DMS s=.5 (λ=.9)", True, 0.9, "#009E73", "--"),
]
# the original notebook figure: its own labels, colours, and ordering
TAIL_SCENARIOS_ORIGINAL = [
    ("1. DMS s=.5 (lambda=.5)", True, 0.5, "#0072B2", "-"),
    ("2. DMS (lambda=.5)", False, 0.5, "#E69F00", "--"),
    ("3. MS s=.5", True, 0.0, "#009E73", ":"),
    ("4. MS", False, 0.0, "#D55E00", "-."),
    ("5. DMS (lambda=.9)", False, 0.9, "#CC79A7", "-"),
    ("6. DMS s=.5 (lambda=.9)", True, 0.9, "#332288", "--"),
]

# the tail figure keeps its own compact legend; the huge cost-plot legend
# overwhelms an 8x5 single-example figure
TAIL_LEGEND_KWARGS = {
    "fontsize": 10,
    "frameon": False,
    "loc": "upper left",
    "bbox_to_anchor": (1.02, 1.0),
}


def plot_tail_threshold(plots_dir: Path) -> None:
    import numpy as _np

    from propflow import FactorAgent, VariableAgent
    from propflow.bp.engines import RDampingEngine
    from propflow.policies.splitting import split_specific_factors
    from propflow.utils.fg_utils import FGBuilder

    # full F12 table; split runs hand this same table to split_specific_factors
    c12_full = _np.array([[8, 20], [32, 16]]).T
    # F23 is built so it always sends a constant 10 into X2
    c23_unary = _np.array([[10, 10], [0, 10]])

    def run_engine(split: bool, damping_factor: float, iterations: int):
        x1 = VariableAgent("X1", domain=2)
        x2 = VariableAgent("X2", domain=2)
        x3 = VariableAgent("X3", domain=2)
        f12 = FactorAgent.create_from_cost_table("F12", cost_table=c12_full.copy())
        f23 = FactorAgent.create_from_cost_table("F23", cost_table=c23_unary.copy())
        graph = FGBuilder.build_from_edges(
            variables=[x1, x2, x3],
            factors=[f12, f23],
            edges={f12: [x1, x2], f23: [x2, x3]},
        )
        if split:
            split_specific_factors(graph, [f12])
        engine = RDampingEngine(factor_graph=graph, damping_factor=damping_factor)
        for i in range(iterations):
            engine.step(i)
        return engine

    def belief_deltas(engine, var: str = "X1") -> list[float]:
        deltas = []
        for snap in engine.snapshots:
            belief = snap.beliefs.get(var)
            deltas.append(float(belief[0] - belief[1]) if belief is not None else 0.0)
        return deltas

    # run each scenario once at the long horizon; the short figures are just
    # the first TAIL_ITERATIONS points of the same trajectory
    deltas_by_run = {
        (split, damping): belief_deltas(
            run_engine(split, damping, TAIL_ITERATIONS_LONG)
        )
        for _, _, split, damping, _, _ in TAIL_SCENARIOS
    }

    def render(scenarios, out_name: str, iterations: int) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, split, damping, color, style in scenarios:
            deltas = deltas_by_run[(split, damping)][:iterations]
            ax.plot(
                range(len(deltas)),
                deltas,
                style,
                color=color,
                label=label,
                linewidth=1.6,
            )
        ax.axhline(
            y=TAIL_THRESHOLD,
            color="red",
            linestyle=":",
            linewidth=1.2,
            label=f"k={TAIL_THRESHOLD:g}",
        )
        ax.set_xlim(0, iterations - 1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Belief delta at X1 (b[0] - b[1])")
        ax.grid(True, alpha=0.3)
        remove_frame(ax)
        ax.legend(**TAIL_LEGEND_KWARGS)

        out = plots_dir / out_name
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    plots_dir.mkdir(parents=True, exist_ok=True)
    restyled = [
        (label, split, damping, color, style)
        for _, label, split, damping, color, style in TAIL_SCENARIOS
    ]
    render(restyled, "split_tail_threshold.pdf", TAIL_ITERATIONS)
    render(
        TAIL_SCENARIOS_ORIGINAL, "split_tail_threshold_original.pdf", TAIL_ITERATIONS
    )
    render(restyled, "split_tail_threshold_long.pdf", TAIL_ITERATIONS_LONG)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(root / "data_cuda"))
    parser.add_argument("--ternary-data-dir", default=str(root / "ternary_data"))
    parser.add_argument("--plots-dir", default=str(root / "final_plots"))
    parser.add_argument("--skip-tail", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ternary_dir = Path(args.ternary_data_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # binary benchmarks from the cuda data, straight into final_plots/
    ratios = load_ratios(data_dir)
    merge_ratios = load_merge_ratios(data_dir)
    for path in sorted(data_dir.glob("*_raw_costs.csv")):
        benchmark = path.name.replace("_raw_costs.csv", "")
        plot_benchmark(benchmark, data_dir, plots_dir, ratios, merge_ratios)

    # ternary benchmarks, one subdirectory each (no timing files -> no stretch)
    for benchmark in TERNARY_BENCHMARKS:
        if not (ternary_dir / f"{benchmark}_raw_costs.csv").exists():
            print(f"skipping {benchmark}: no raw costs in {ternary_dir}")
            continue
        plot_benchmark(benchmark, ternary_dir, plots_dir / benchmark, {}, {})

    if not args.skip_tail:
        plot_tail_threshold(plots_dir)


if __name__ == "__main__":
    main()
