"""Plot the split-vs-no-split belief gap on the three-variable tail chain.

Reproduces the convergence figure from `notebooks/split_tail_example.ipynb`:
the belief delta at X1 over iterations for six scenarios (split / no split
crossed with damping 0.0 / 0.5 / 0.9), against the threshold k = 8 that the
gap must cross for the chain to settle on (b, b).

Chain: X1 -- F12 -- X2 -- F23 -- X3, where F23 is a unary-style constraint
that drives a constant 10 into X2. Split runs put the same full table through
`split_specific_factors`, so both graphs encode the same objective and any
difference in the gap is a message-passing effect.

PDF is written to --plots-dir:
  split_tail_threshold.pdf

Example:
  uv run python experiments/aaai/code/plot_split_tail_threshold.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from propflow import FactorAgent, VariableAgent  # noqa: E402
from propflow.bp.engines import RDampingEngine  # noqa: E402
from propflow.core.components import CostTable  # noqa: E402
from propflow.policies.splitting import split_specific_factors  # noqa: E402
from propflow.utils.fg_utils import FGBuilder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.plot_helpers import remove_frame  # noqa: E402

# full F12 table; the split runs hand this same table to split_specific_factors,
# which halves it across the two clones
C12_FULL: CostTable = np.array([[8, 20], [32, 16]]).T
# F23 is built so it always sends a constant 10 into X2
C23_UNARY: CostTable = np.array([[10, 10], [0, 10]])

THRESHOLD = 8.0
ITERATIONS = 25

# (label, split, damping_factor, colour, linestyle) in the notebook's order
SCENARIOS = [
    ("1. DMS s=.5 (lambda=.5)", True, 0.5, "#0072B2", "-"),
    ("2. DMS (lambda=.5)", False, 0.5, "#E69F00", "--"),
    ("3. MS s=.5", True, 0.0, "#009E73", ":"),
    ("4. MS", False, 0.0, "#D55E00", "-."),
    ("5. DMS (lambda=.9)", False, 0.9, "#CC79A7", "-"),
    ("6. DMS s=.5 (lambda=.9)", True, 0.9, "#332288", "--"),
]


def run_tail_engine(*, split: bool, damping_factor: float, iterations: int):
    """Build a fresh tail chain and step it, so runs never share graph state."""
    x1 = VariableAgent("X1", domain=2)
    x2 = VariableAgent("X2", domain=2)
    x3 = VariableAgent("X3", domain=2)

    f12 = FactorAgent.create_from_cost_table("F12", cost_table=C12_FULL.copy())
    f23 = FactorAgent.create_from_cost_table("F23", cost_table=C23_UNARY.copy())

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
    """Belief gap b[0] - b[1] at `var` for each recorded iteration."""
    deltas = []
    for snap in engine.snapshots:
        belief = snap.beliefs.get(var)
        deltas.append(float(belief[0] - belief[1]) if belief is not None else 0.0)
    return deltas


def plot_threshold(plots_dir: Path, iterations: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, split, damping, color, style in SCENARIOS:
        engine = run_tail_engine(
            split=split, damping_factor=damping, iterations=iterations
        )
        deltas = belief_deltas(engine)
        ax.plot(
            range(len(deltas)), deltas, style, color=color, label=label, linewidth=1.6
        )

    ax.axhline(
        y=THRESHOLD,
        color="red",
        linestyle=":",
        linewidth=1.2,
        label=f"Threshold k={THRESHOLD:g}",
    )
    ax.set_xlim(0, iterations - 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Belief delta at X1 (b[0] - b[1])")
    ax.grid(True, alpha=0.3)
    remove_frame(ax)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "split_tail_threshold.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir", default=str(Path(__file__).resolve().parents[1] / "plots")
    )
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()

    out = plot_threshold(Path(args.plots_dir), args.iterations)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
