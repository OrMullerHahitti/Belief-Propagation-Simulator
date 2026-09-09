"""Plot the split-vs-no-split belief gap on the tail chain with a [17, 0] unary.

Companion to `plot_split_tail_unary13.py` with the same F12 table but a unary
past the flip point: the belief delta at X1 over iterations for six scenarios
(split / no split crossed with damping 0.0 / 0.5 / 0.9).

Chain: X1 -- F12 -- X2 -- F23 -- X3. Each split half of F12 carries
[[0, 20], [30, 8]], so the full table is [[0, 40], [60, 16]]. F23 is the same
kind of unary-style constraint used before -- a binary table over (X2, X3)
whose row minima make it drive a constant [17, 0] into X2, because X3 is a leaf
and always returns a zero Q-message.

17 is just past the flip point of 16: (a, a) now costs 0 + 17 = 17 against
(b, b) at 16 + 0 = 16, so the optimum moves to (b, b) -- the opposite of the
[13, 0] case, where (a, a) still won at 13 versus 16. The delta is plotted as
b[1] - b[0] to match that figure; beliefs are costs, so this run goes negative
as X1 leans to b.

--delta picks the subtraction order: b1-b0 (default, negative here) or b0-b1
(positive here). PDF is written to --plots-dir; the default 60-iteration run at
the default delta keeps the plain name, anything else gets a suffix:
  split_tail_unary17.pdf              (60 iterations, b[1] - b[0])
  split_tail_unary17_it100.pdf
  split_tail_unary17_it300.pdf
  split_tail_unary17_b0b1.pdf         (60 iterations, b[0] - b[1])
  split_tail_unary17_b0b1_it100.pdf
  split_tail_unary17_b0b1_it300.pdf

Example:
  uv run python experiments/aaai/code/plot_split_tail_unary17.py
  uv run python experiments/aaai/code/plot_split_tail_unary17.py --iterations 300
  uv run python experiments/aaai/code/plot_split_tail_unary17.py --delta b0-b1
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

# each split half is [[0, 20], [30, 8]]; the split runs hand the doubled table
# to split_specific_factors, which halves it back across the two clones
C12_SPLIT_HALF: CostTable = np.array([[0, 20], [30, 8]])
C12_FULL = C12_SPLIT_HALF * 2

# F23 is built so it always sends a constant [17, 0] into X2
C23_UNARY: CostTable = np.array([[17, 17], [0, 17]])

ITERATIONS = 60

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


DELTAS = {"b1-b0": (1, 0), "b0-b1": (0, 1)}


def belief_deltas(engine, var: str = "X1", delta: str = "b1-b0") -> list[float]:
    """Belief gap at `var` for each recorded iteration, in the given order."""
    lo, hi = DELTAS[delta]
    deltas = []
    for snap in engine.snapshots:
        belief = snap.beliefs.get(var)
        deltas.append(float(belief[lo] - belief[hi]) if belief is not None else 0.0)
    return deltas


def output_name(iterations: int, delta: str = "b1-b0") -> str:
    """Plain name for the default horizon and delta; suffixes for anything else."""
    stem = "split_tail_unary17"
    if delta != "b1-b0":
        stem += "_b0b1"
    if iterations != ITERATIONS:
        stem += f"_it{iterations}"
    return f"{stem}.pdf"


def plot_tail_unary17(plots_dir: Path, iterations: int, delta: str = "b1-b0") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, split, damping, color, style in SCENARIOS:
        engine = run_tail_engine(
            split=split, damping_factor=damping, iterations=iterations
        )
        deltas = belief_deltas(engine, delta=delta)
        ax.plot(
            range(len(deltas)), deltas, style, color=color, label=label, linewidth=1.6
        )

    lo, hi = DELTAS[delta]
    ax.set_xlim(0, iterations - 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Belief delta at X1 (b[{lo}] - b[{hi}])")
    ax.grid(True, alpha=0.3)
    remove_frame(ax)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / output_name(iterations, delta)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir", default=str(Path(__file__).resolve().parents[1] / "plots")
    )
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--delta", choices=sorted(DELTAS), default="b1-b0")
    args = parser.parse_args()

    out = plot_tail_unary17(Path(args.plots_dir), args.iterations, args.delta)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
