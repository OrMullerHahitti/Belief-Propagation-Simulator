"""Generate snapshot visualizations for random factor graphs.

This utility spins up a random factor graph, runs belief propagation with
snapshots enabled, and produces the new visualizations added to
``SnapshotVisualizer``. Figures are saved under the ``figures/`` directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

from propflow import BPEngine, FGBuilder
from propflow.configs import CTFactories
from propflow.snapshots import SnapshotVisualizer


def run_random_engine(
    *,
    num_vars: int = 10,
    domain_size: int = 5,
    density: float = 0.6,
    max_iter: int = 20,
) -> SnapshotVisualizer:
    """Run BP on a random factor graph and wrap the snapshots in a visualizer."""
    np.random.seed(1234)

    factor_graph = FGBuilder.build_random_graph(
        num_vars=num_vars,
        domain_size=domain_size,
        ct_factory=CTFactories.RANDOM_INT,
        ct_params={"low": 0, "high": 7},
        density=density,
    )

    engine = BPEngine(factor_graph=factor_graph, use_bct_history=True)
    engine.run(max_iter=max_iter)

    records = list(engine.snapshots)
    if not records:
        raise RuntimeError("Engine did not capture any snapshots; check configuration.")
    return SnapshotVisualizer(records)


def main() -> None:
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    visualizer = run_random_engine()

    global_cost_path = figures_dir / "global_cost.png"
    visualizer.plot_global_cost(
        show=False,
        savepath=str(global_cost_path),
        rolling_window=3,
    )
    print(f"Saved global cost trajectory to {global_cost_path}")

    q_norms_path = figures_dir / "message_norms_q.png"
    visualizer.plot_message_norms(
        message_type="Q",
        show=False,
        savepath=str(q_norms_path),
    )
    print(f"Saved Q message norms to {q_norms_path}")

    r_norms_path = figures_dir / "message_norms_r.png"
    visualizer.plot_message_norms(
        message_type="R",
        show=False,
        savepath=str(r_norms_path),
    )
    print(f"Saved R message norms to {r_norms_path}")

    heatmap_path = figures_dir / "assignment_heatmap.png"
    visualizer.plot_assignment_heatmap(
        show=False,
        savepath=str(heatmap_path),
    )
    print(f"Saved assignment heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
