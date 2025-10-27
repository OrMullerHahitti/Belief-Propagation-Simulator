"""Demonstration of snapshot analysis on a small 3-variable ring factor graph.

NOTE: This example demonstrates core analysis features from propflow.snapshots.
Some specialized features from the legacy analyzer module (like scc_greedy_neutral_cover)
are no longer available. The new snapshot module focuses on convergence analysis,
Jacobian computation, and cycle detection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from propflow import BPEngine, FGBuilder
from propflow.configs import create_random_int_table
from propflow.snapshots import SnapshotAnalyzer, AnalysisReport, SnapshotsConfig
from propflow.snapshots.utils import get_snapshot


def build_ring_3var():
    """Build a simple 3-variable ring factor graph."""
    return FGBuilder.build_cycle_graph(
        num_vars=3,
        domain_size=2,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 2},
    )


def main() -> None:
    np.random.seed(42)

    # Build and run engine with snapshots
    fg = build_ring_3var()
    snapshot_cfg = SnapshotsConfig(
        compute_jacobians=True,
        compute_block_norms=True,
        compute_cycles=True,
        max_cycle_len=6,
    )

    engine = BPEngine(factor_graph=fg, snapshots_config=snapshot_cfg)
    engine.run(max_iter=20)

    # Collect snapshots
    snapshots = [
        get_snapshot(engine, i)
        for i in range(len(engine.history.step_costs))
    ]

    # Analyze snapshots
    analyzer = SnapshotAnalyzer(snapshots)
    report = AnalysisReport(analyzer)

    out_dir = Path("results/analysis_ring")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export analysis as JSON
    summary = report.to_json(step_idx=len(snapshots) - 1)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("✓ Analysis summary exported to summary.json")
    print(f"  - Block norms: {summary.get('block_norms', {})}")
    print(f"  - Cycles found: {summary.get('cycle_metrics', {}).get('num_cycles', 0)}")
    print(f"  - Nilpotent index: {summary.get('nilpotent_index', None)}")

    # Show belief trajectories
    beliefs = analyzer.beliefs_per_variable()
    print("\nBelief trajectories (argmin per variable):")
    for var, trajectory in beliefs.items():
        print(f"  {var}: {trajectory}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
