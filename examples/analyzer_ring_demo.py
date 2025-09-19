"""Example: run EngineSnapshotRecorder and SnapshotVisualizer on a 4-variable ring graph."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from analyzer.snapshot_recorder import EngineSnapshotRecorder
from analyzer.snapshot_visualizer import SnapshotVisualizer
from propflow.bp.engines import BPEngine
from propflow.configs.global_config_mapping import CTFactory
from propflow.utils.fg_utils import FGBuilder

RESULTS_DIR = Path("results")
SNAPSHOT_PATH = RESULTS_DIR / "ring_snapshots.json"
PLOT_PATH = RESULTS_DIR / "ring_argmin.png"
COMBINED_PLOT_PATH = RESULTS_DIR / "ring_argmin_combined.png"


def build_ring(num_vars: int = 4, domain_size: int = 3):
    """Construct a ring factor graph using the random integer cost table factory."""
    return FGBuilder.build_cycle_graph(
        num_vars=num_vars,
        domain_size=domain_size,
        ct_factory=CTFactory.random_int,
        ct_params={"low": 0, "high": 5},
    )


def run_engine(max_steps: int = 12) -> list[dict]:
    """Run a basic BP engine on the ring and capture per-step snapshots."""
    np.random.seed(0)

    fg = build_ring()
    engine = BPEngine(factor_graph=fg)

    recorder = EngineSnapshotRecorder(engine)
    recorder.record_run(max_steps=max_steps)
    return recorder.snapshots


def save_snapshots(snapshots: Sequence[dict]) -> None:
    """Persist snapshots to JSON for later inspection."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(list(snapshots), handle, indent=2)


def generate_plot(snapshots: Sequence[dict], *, show: bool = False) -> None:
    """Generate an argmin trajectory plot for all variables in the ring."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    viz = SnapshotVisualizer.from_object(list(snapshots))
    series = viz.argmin_series()
    print("Argmin series per variable:")
    for var, values in series.items():
        print(f"  {var}: {values}")

    viz.plot_argmin_per_variable(
        show=show,
        savepath=str(PLOT_PATH),
        combined_savepath=str(COMBINED_PLOT_PATH),
    )
    print(f"Saved per-variable plot to {PLOT_PATH}")
    print(f"Saved combined plot to {COMBINED_PLOT_PATH}")


def main() -> None:
    snapshots = run_engine()
    save_snapshots(snapshots)
    generate_plot(snapshots)
    print(f"Stored snapshots at {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
