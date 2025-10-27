"""Large random graph analysis with comprehensive snapshot export.

This example demonstrates snapshot analysis on a larger randomly-generated
factor graph, computing convergence metrics and exporting analysis results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from propflow import BPEngine, FGBuilder
from propflow.configs import create_random_int_table
from propflow.snapshots import SnapshotAnalyzer, AnalysisReport, SnapshotsConfig
from propflow.snapshots.utils import get_snapshot


def main() -> None:
    # Generate a larger random factor graph
    np.random.seed(0)

    fg = FGBuilder.build_random_graph(
        num_vars=200,
        domain_size=2,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 10},
        density=0.05,  # 5% factor density for connectivity
    )

    # Run with snapshot capture
    snapshot_cfg = SnapshotsConfig(
        compute_jacobians=True,
        compute_block_norms=True,
        compute_cycles=True,
        max_cycle_len=4,
        retain_last=None,  # Keep all snapshots
    )

    engine = BPEngine(factor_graph=fg, snapshots_config=snapshot_cfg)
    print("Running BP on 200-variable random graph...")
    engine.run(max_iter=50)

    # Collect snapshots
    snapshots = [
        get_snapshot(engine, i)
        for i in range(len(engine.history.step_costs))
    ]
    print(f"✓ Captured {len(snapshots)} snapshots")

    # Analyze
    analyzer = SnapshotAnalyzer(snapshots)
    report = AnalysisReport(analyzer)

    out_dir = Path("results/analysis_random")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export final step analysis
    final_step = len(snapshots) - 1
    summary = report.to_json(step_idx=final_step)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("✓ Analysis exported to", out_dir)
    print(f"  - Nilpotent index: {summary.get('nilpotent_index', None)}")
    print(f"  - Cycles found: {summary.get('cycle_metrics', {}).get('num_cycles', 0)}")

    # Show belief convergence
    beliefs = analyzer.beliefs_per_variable()
    converged_vars = sum(1 for traj in beliefs.values() if len(set(traj[-5:])) == 1)
    print(f"  - Variables converged: {converged_vars}/{len(beliefs)}")


if __name__ == "__main__":  # pragma: no cover
    main()
