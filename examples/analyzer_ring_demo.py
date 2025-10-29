"""Example: Capture and visualize snapshots on a 4-variable ring graph using SnapshotVisualizer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from propflow import BPEngine, FGBuilder, SnapshotVisualizer
from propflow.configs import CTFactories
from propflow.snapshots import SnapshotsConfig
from propflow.snapshots.utils import get_snapshot

RESULTS_DIR = Path("results")
PLOT_PATH = RESULTS_DIR / "ring_argmin.png"
COMBINED_PLOT_PATH = RESULTS_DIR / "ring_argmin_combined.png"


def build_ring(num_vars: int = 4, domain_size: int = 3):
    """Construct a ring factor graph using the random integer cost table factory."""
    return FGBuilder.build_cycle_graph(
        num_vars=num_vars,
        domain_size=domain_size,
        ct_factory=CTFactories.RANDOM_INT.value,
        ct_params={"low": 0, "high": 5},
    )


def run_engine(max_steps: int = 12):
    """Run a basic BP engine on the ring with snapshot capture enabled."""
    np.random.seed(0)

    fg = build_ring()

    # Configure snapshot capture
    snapshot_cfg = SnapshotsConfig(
        compute_jacobians=False,
        compute_block_norms=False,
        compute_cycles=False,
        retain_last=None,  # Keep all snapshots
        save_each_step=False,  # Don't auto-save to disk
    )

    engine = BPEngine(factor_graph=fg, snapshots_config=snapshot_cfg)
    engine.run(max_iter=max_steps)

    # Collect all snapshots
    snapshots = [
        get_snapshot(engine, i)
        for i in range(len(engine.history.step_costs))
    ]
    return snapshots


def generate_plot(snapshots: list, *, show: bool = False) -> None:
    """Generate an argmin trajectory plot for all variables in the ring."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create visualizer from snapshots
    visualizer = SnapshotVisualizer(snapshots)

    # Get argmin series
    series = visualizer.argmin_series()
    print("Argmin series per variable:")
    for var, values in series.items():
        print(f"  {var}: {values}")

    # Generate plots
    visualizer.plot_argmin_per_variable(
        show=show,
        savepath=str(PLOT_PATH),
        combined_savepath=str(COMBINED_PLOT_PATH),
    )
    print(f"Saved per-variable plot to {PLOT_PATH}")
    print(f"Saved combined plot to {COMBINED_PLOT_PATH}")


def main() -> None:
    snapshots = run_engine()
    generate_plot(snapshots)
    print(f"Captured {len(snapshots)} snapshots")


if __name__ == "__main__":
    main()
