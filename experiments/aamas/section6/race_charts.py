"""section 6 race charts: mean cost and fraction of messages at a bound, animated over iterations.

one self-contained html file (no network), both charts driven by one timeline, a benchmark picker,
a k picker for the delayed split and a bar/line switch per chart.

the numbers are the ones behind the paper figures:
  - cost: plot_final.py --section6 --data-dir data_paper_20260928
          --extra-raw experiments/aamas/section6/split_at_best_fixed_horizon
    (mean over the 50 instances, runs padded with their last value, DABP on the wall-clock axis)
  - bound: mechanism_figure.py (exp1 and exp1_delayed, mean of `sats` over the 50 instances)
frame f is paper iteration 2f; cost uses library iteration f, bound uses sats[f - 1] (as the figures do).

inputs: experiments/aaai/data_paper_20260928, experiments/aamas/section6/split_at_best_fixed_horizon,
        experiments/aamas/splitting_explanation/results
output: experiments/aamas/section6/out/race_charts.html and out/bound_line_race.html (bound line race only)

usage: uv run python experiments/aamas/section6/race_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COST_DIR = ROOT / "experiments" / "aaai" / "data_paper_20260928"
BEST_DIR = HERE / "split_at_best_fixed_horizon"
BOUND_DIR = HERE.parent / "splitting_explanation" / "results"
TEMPLATE = HERE / "race_charts_template.html"
OUT = HERE / "out" / "race_charts.html"
SOLO_OUT = HERE / "out" / "bound_line_race.html"

BENCHES = [
    ("random_sparse", "Random sparse"),
    ("random_dense", "Random dense"),
    ("scale_free", "Scale free"),
    ("graph_coloring", "Graph coloring"),
    ("meeting_scheduling", "Meeting scheduling"),
]
HORIZON = 2000  # library iterations; the figures run to paper iteration 3998
LIBRARY_KS = [50, 100, 300, 500, 1000, 1500]
DEFAULT_K = 1000  # paper units, the line in the paper figures

# line key in the page -> algorithm name in the cost data
COST_LINES = {
    "MS": "MS",
    "MS-SCFG": "MS_split_0.5",
    "MS-SCFG-opt": "MS_split_opt_200",
    "DMS": "DMS",
    "DMS-SCFG": "DMS_split_0.5",
    "DMS-BDS": "DMS_split_at_best",
    "DABP": "Attentive",
}
# line key in the page -> key prefix in the exp1 npz
BOUND_LINES = {
    "MS": "MS",
    "MS-SCFG": "MS_split",
    "DMS": "DMS",
    "DMS-SCFG": "DMS_split",
}


def padded_mean(group: pd.DataFrame) -> np.ndarray:
    """mean over seeds per library iteration, each run padded with its last value (plot_final.padded_runs)."""
    wide = group.pivot(index="seed", columns="iteration", values="cost")
    wide = wide.reindex(columns=range(HORIZON)).ffill(axis=1)
    if wide.isna().any().any():
        raise ValueError(
            f"{group['algorithm'].iloc[0]}: a run has no cost at iteration 0"
        )
    if len(wide) != 50:
        raise ValueError(
            f"{group['algorithm'].iloc[0]}: {len(wide)} seeds, expected 50"
        )
    return wide.to_numpy(dtype=float).mean(axis=0)


def dabp_ratio(bench: str) -> float:
    timing = pd.read_csv(COST_DIR / "dabp_timing.csv").set_index("benchmark")
    return float(timing.loc[bench, "ratio"])


def on_wall_clock(mean: np.ndarray, ratio: float) -> np.ndarray:
    """DABP's own iteration i is drawn at library iteration ratio * i; read it back at every library iteration."""
    frames = np.arange(HORIZON, dtype=float)
    return np.interp(frames / ratio, np.arange(len(mean), dtype=float), mean)


def rounded(values: np.ndarray, decimals: int) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), decimals) for v in values]


def cost_block(bench: str) -> tuple[dict, list[int], int]:
    wanted = set(COST_LINES.values()) | {f"DMS_split_at_{k}" for k in LIBRARY_KS}
    raw = pd.read_csv(COST_DIR / f"{bench}_raw_costs.csv")
    raw = raw[raw["algorithm"].isin(wanted)]
    best = pd.read_csv(BEST_DIR / f"{bench}_raw_costs.csv")
    raw = pd.concat([raw, best], ignore_index=True)
    means = {name: padded_mean(group) for name, group in raw.groupby("algorithm")}
    ratio = dabp_ratio(bench)
    decimals = 1 if means["DMS"].max() > 1000 else 3
    lines = {}
    for key, algorithm in COST_LINES.items():
        mean = means[algorithm]
        if key == "DABP":
            mean = on_wall_clock(mean, ratio)
        lines[key] = rounded(mean, decimals)
    ks = [k for k in LIBRARY_KS if f"DMS_split_at_{k}" in means]
    kds = {str(2 * k): rounded(means[f"DMS_split_at_{k}"], decimals) for k in ks}
    return (
        {
            "lines": lines,
            "kds": kds,
            "decimals": decimals,
            "dabp_ratio": round(ratio, 2),
        },
        ks,
        decimals,
    )


def bound_block(bench: str, ks: list[int]) -> dict:
    z = np.load(BOUND_DIR / f"exp1_{bench}.npz")
    zd = np.load(BOUND_DIR / f"exp1_delayed_{bench}.npz")

    def aligned(sats: np.ndarray) -> list[float | None]:
        if sats.shape != (50, HORIZON):
            raise ValueError(f"{bench}: sats shape {sats.shape}")
        mean = sats.astype(float).mean(axis=0)
        # frame f = paper iteration 2f shows sats[f - 1]; nothing is sent before the first iteration
        return [None] + rounded(mean[: HORIZON - 1], 4)

    lines = {key: aligned(z[f"{prefix}/sats"]) for key, prefix in BOUND_LINES.items()}
    kds = {str(2 * k): aligned(zd[f"DMS_split_at_{k}/sats"]) for k in ks}
    return {"lines": lines, "kds": kds}


def main() -> None:
    benches = []
    for bench, title in BENCHES:
        cost, cost_ks, _ = cost_block(bench)
        bound_ks = [
            k
            for k in LIBRARY_KS
            if f"DMS_split_at_{k}/sats"
            in np.load(BOUND_DIR / f"exp1_delayed_{bench}.npz")
        ]
        ks = [k for k in cost_ks if k in bound_ks]
        cost["kds"] = {key: v for key, v in cost["kds"].items() if int(key) // 2 in ks}
        benches.append(
            {
                "key": bench,
                "title": title,
                "ks": [2 * k for k in ks],
                "cost": cost,
                "bound": bound_block(bench, ks),
            }
        )
        print(
            f"{bench}: k (paper) {[2 * k for k in ks]}, DABP ratio {cost['dabp_ratio']}"
        )
    data = {"frames": HORIZON, "default_k": DEFAULT_K, "benches": benches}
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text()
    if html.count("/*__DATA__*/null") != 1:
        raise ValueError(
            "template must contain exactly one /*__DATA__*/null placeholder"
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html.replace("/*__DATA__*/null", payload))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")

    # the same page with only the line race of the messages at a bound, no text around it
    solo = {**data, "solo": True, "benches": [{k: v for k, v in b.items() if k != "cost"} for b in benches]}
    solo_html = html.replace("<title>Section 6 race charts</title>", "<title>Messages at a bound</title>")
    SOLO_OUT.write_text(solo_html.replace("/*__DATA__*/null", json.dumps(solo, separators=(",", ":"), allow_nan=False)))
    print(f"wrote {SOLO_OUT} ({SOLO_OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
