"""Plot paired mean original costs from the frozen sparse confirmation data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

from experiments.aaai.code.utils.plot_helpers import remove_frame


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/aaai_derived_control_20260915/paper_confirmation"
OUT = SOURCE.parent / "cost_iteration_plot"
METHODS = ("split05_d09", "split095_pulse64_256_d09")
COLORS = ("#454B54", "#2456A6")


def main() -> None:
    """Validate saved trajectory prefixes, render the comparison, and export."""
    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    contract = {
        "question": "How does original cost evolve under the pulse and usual baseline?",
        "takeaway": "The pulse has lower mean terminal cost on the 32 held-out sparse graphs.",
        "family": "line",
        "variant": "two mean curves with a separate expanded vertical scale",
        "renderer": "Matplotlib scientific research figure, PNG and vector PDF",
        "data": (
            "32 paired seeds 6000--6031; 10000 completed updates per trajectory; "
            "no smoothing or best-so-far transformation"
        ),
        "panels": "Same curves and iteration range; left full observed cost range, right expanded cost scale",
        "palette": {
            "baseline": COLORS[0],
            "pulse": COLORS[1],
        },
        "non_color_encoding": "baseline dashed, pulse solid; named legend",
        "footprint": "12 by 4.2 inches; inspect exported PNG",
        "presentation": "side-by-side panels; axes and shared legend only; no titles or notes",
    }
    (OUT / "chart_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    inputs = {}
    values = {}
    metrics = {
        (r["method"], int(r["seed"]), int(r["horizon"])): float(r["terminal_cost"])
        for r in csv.DictReader((SOURCE / "metrics.csv").open())
    }
    for method in METHODS:
        series = []
        for seed in range(6000, 6032):
            path = SOURCE / "trajectories" / f"random_sparse_{seed}_{method}.csv"
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1))
            np.testing.assert_array_equal(data[:, 0], np.arange(1, 10001))
            if not np.isfinite(data[:, 1]).all():
                raise ValueError(f"nonfinite costs in {path}")
            for horizon in (2000, 10000):
                np.testing.assert_allclose(
                    data[horizon - 1, 1],
                    metrics[method, seed, horizon],
                    atol=1e-9,
                    rtol=0,
                )
            series.append(data[:, 1])
        values[method] = np.stack(series).mean(axis=0)
    np.testing.assert_array_equal(values[METHODS[0]][:64], values[METHODS[1]][:64])
    iterations = np.arange(1, 10001)
    np.savetxt(
        OUT / "mean_cost_by_iteration.csv",
        np.column_stack([iterations, *(values[m] for m in METHODS)]),
        delimiter=",",
        header="completed_updates,baseline_mean_cost,pulse_mean_cost",
        comments="",
    )
    provenance = {
        "source_trajectories_sha256": inputs,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_count": 32,
        "seeds": list(range(6000, 6032)),
        "graph_variables": 50,
        "domain": 10,
        "damping": 0.9,
        "metric": "arithmetic mean of actual original-objective cost after each completed update",
        "final_2000": {m: float(values[m][1999]) for m in METHODS},
        "final_10000": {m: float(values[m][9999]) for m in METHODS},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#272C33",
            "text.color": "#272C33",
            "xtick.color": "#535A63",
            "ytick.color": "#535A63",
            "axes.edgecolor": "#A7ACB3",
            "pdf.fonttype": 42,
        }
    )
    for horizon in (2000, 10000):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.15, top=0.88, wspace=0.25)
        labels = ("Fixed 0.5 split", "0.5 → 0.95 → 0.5 split")
        for ax in axes:
            for method, color, style, label in zip(
                METHODS, COLORS, ("--", "-"), labels
            ):
                ax.plot(
                    iterations[:horizon],
                    values[method][:horizon],
                    color=color,
                    linestyle=style,
                    linewidth=1.8,
                    label=label,
                    zorder=3,
                )
            remove_frame(ax)
            ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(0, horizon)
            ax.set_xticks(np.linspace(0, horizon, 5))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.set_xlabel("Iteration", labelpad=9)
            ax.set_ylabel("Mean cost", labelpad=9)
        final = np.array([values[m][horizon - 1] for m in METHODS])
        upper = max(values[m][63:1000].max() for m in METHODS)
        axes[1].set_ylim(
            np.floor(final.min() / 50) * 50 - 50, np.ceil(upper / 50) * 50 + 25
        )
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            ncol=2,
            fontsize=10,
            handlelength=3,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                OUT / f"cost_by_iteration_{horizon}.{suffix}",
                dpi=180,
                facecolor="white",
            )
        plt.close(fig)
    print(json.dumps(provenance["final_2000"]))
    print(OUT)


if __name__ == "__main__":
    main()
