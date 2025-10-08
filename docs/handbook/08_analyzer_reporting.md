# Analyzer and Reporting Workflows

This chapter explains how to turn raw belief propagation snapshots into actionable analytics. It builds on the runtime snapshot infrastructure provided by `EngineSnapshotRecorder`, and shows how to drive the reporting pipeline (parsers, `SnapshotAnalyzer`, and `AnalysisReport`). Use this document whenever you want to certify neutral covers, inspect Jacobians, or export CSV/plot artefacts for downstream tools.

## 1. Capturing Snapshots

1. Attach a recorder to the engine before running the simulation:
   ```python
   from analyzer.snapshot_recorder import EngineSnapshotRecorder

   recorder = EngineSnapshotRecorder(engine)
   snapshots = recorder.record_run(max_steps=50)
   recorder.to_json("results/run.json")
   ```
2. The JSON output contains per-step message payloads (Q/R flows, neutral flags, argmin indices), assignments, and global cost signals. The reporting layer consumes this format directly.

## 2. Parsing Snapshot Payloads

Use the reporting parser to convert raw JSON into typed snapshot records. The parser enforces schema integrity (step ordering, neutral counts) so downstream logic can rely on consistent structures.

```python
from analyzer.reporting import load_snapshots, parse_snapshots

raw = load_snapshots("results/run.json")
records = parse_snapshots(raw)
```

If you already captured snapshots through the in-engine `SnapshotManager`, adapt them with `from_engine_snapshot_manager`.

## 3. Running the Snapshot Analyzer

Create a `SnapshotAnalyzer` to reconstruct beliefs, build Jacobians, and query neutrality certificates. Register factor cost tables when neutrality thresholds or multi-label gap certificates are required.

```python
import numpy as np
from analyzer.reporting import SnapshotAnalyzer

analyzer = SnapshotAnalyzer(records, max_cycle_len=8)
for factor_name, cost_table in factor_costs.items():
    analyzer.register_factor_cost(factor_name, np.asarray(cost_table, dtype=float))

belief_series = analyzer.beliefs_per_variable()
jacobian0 = analyzer.jacobian(step_idx=0)
cover0, residual0 = analyzer.scc_greedy_neutral_cover(0, alpha={})
nilpotent0 = analyzer.nilpotent_index(0)
block_norms0 = analyzer.block_norms(0)
```

Key insights supplied by the analyzer include:

- **Difference coordinates** for both Q and R messages (binary gaps vs. recentered vectors).
- **Jacobian matrices** (dense or sparse automatically) with dependency digraph extraction.
- **Neutral certificates**: binary thresholds, multi-label gaps, SCC-based greedy covers.
- **Nilpotent bounds**: actual index `L` (when it exists) and longest-path bound `L_dag`.
- **Cycle metrics**: counts, aligned hops, and neutral coverage signals.
- **Split-ratio recommendations**: heuristic `α` values easing slack violations.

## 4. Generating Reports and Artefacts

The `AnalysisReport` helper packages common export paths. It collates analyzer outputs into JSON/CSV/plot bundles and optionally draws the dependency digraph.

```python
from analyzer.reporting import AnalysisReport

report = AnalysisReport(analyzer)
summary = report.to_json(step_idx=0)
report.to_csv("results/analysis", step_idx=0)
report.plots("results/analysis", step_idx=0, include_graph=True)
```

The summary JSON includes belief trajectories, neutral covers, nilpotent bounds, block norms, cycle statistics, split-ratio suggestions, and spectral radius estimates (dense matrices under the hood for small systems).

## 5. Command Line Interface

The `bp-analyze` CLI mirrors the API workflow so you can process snapshot files without writing Python glue.

```
bp-analyze \
  --snapshots results/run.json \
  --out results/analysis \
  --step 0 \
  --compute-jac \
  --cover \
  --plot
```

Frequently used flags:

- `--snapshots`: Input JSON path from `EngineSnapshotRecorder`.
- `--out`: Output directory (folders are created automatically).
- `--step`: Step index to analyze (default 0).
- `--compute-jac`: Export the dense Jacobian as `jacobian.csv`.
- `--cover`: Include neutral cover JSON alongside the summary.
- `--plot`: Produce argmin trajectories (and dependency graph when combined with `--cover`).

## 6. Worked Examples

Two reference scripts live under `examples/`:

- `analysis_ring_scfg.py`: Three-variable ring showcasing neutral covers, nilpotent index, and plot generation. Useful for validating SCFG-style damping choices and the `L + 1` finite-time bound.
- `analysis_large_random.py`: Large random system demonstrating sparse Jacobian handling and CSV-only outputs (ideal for offline processing or CI smoke tests).

Run them from the repository root with the active environment:

```
uv run python examples/analysis_ring_scfg.py
uv run python examples/analysis_large_random.py
```

Both examples deposit artefacts under `results/`, mirroring the layout created by `AnalysisReport`.

## 7. Best Practices

- Capture snapshots with the same deterministic seeds used during experiments to make comparisons reproducible.
- Register factor cost tables whenever neutrality or split-ratio diagnostics are required; the analyzer falls back to observed messages otherwise.
- On large graphs, prefer sparse Jacobians (`n >= 100` triggers the sparse path automatically) and disable spectral radius estimation if you only need DAG bounds.
- Integrate the CLI into CI pipelines to sanity check new configurations before promotion.

With these tools, you can fully reproduce the empirical analysis presented in the PropFlow convergence paper and extend it to new domains.
