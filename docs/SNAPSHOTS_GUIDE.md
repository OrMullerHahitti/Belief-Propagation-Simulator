# PropFlow Snapshots Guide

Snapshots are lightweight per-step records captured automatically by
`BPEngine.step()` and `BPEngine.run()`. They are the current source of truth for
history, visualisation, and analysis workflows.

## What Gets Captured

Each `EngineSnapshot` includes:

| Field | Meaning |
| --- | --- |
| `step` | Iteration index. |
| `lambda_` | Engine damping factor when available, otherwise `0.0`. |
| `dom` | Domain labels for each variable. |
| `N_var`, `N_fac` | Variable and factor neighbourhoods. |
| `Q`, `R` | Variable-to-factor and factor-to-variable message dictionaries. |
| `beliefs` | Current belief vectors by variable name. |
| `assignments` | Current assignment by variable name. |
| `global_cost` | Cost for the current assignment when computable. |
| `metadata` | Engine type, graph diameter, node counts, and message counts. |
| `cost_tables`, `cost_labels` | Current factor tables and axis ordering for diagnostics. |
| `jacobians`, `cycles`, `winners`, `min_idx` | Optional analysis fields, usually populated by post-processing rather than capture. |

## Capture and Access

```python
from propflow import BPEngine, FGBuilder
from propflow.configs import create_random_int_table

graph = FGBuilder.build_cycle_graph(
    num_vars=6,
    domain_size=3,
    ct_factory=create_random_int_table,
    ct_params={"low": 0, "high": 5},
)

engine = BPEngine(graph)
engine.run(max_iter=50)

snapshots = list(engine.snapshots)
latest = engine.latest_snapshot()
by_step = engine.snapshot_map

print(latest.step, latest.global_cost, latest.assignments)
```

No `use_bct_history` or recorder flag is required in the current implementation.

## Analyze

`SnapshotAnalyzer` consumes the in-memory `engine.snapshots` list.

```python
from propflow.snapshots import AnalysisReport, SnapshotAnalyzer

analyzer = SnapshotAnalyzer(engine.snapshots)

belief_series = analyzer.beliefs_per_variable()
delta_q, delta_r = analyzer.difference_coordinates(step_idx=len(engine.snapshots) - 1)
jacobian = analyzer.jacobian(step_idx=len(engine.snapshots) - 1)
block_norms = analyzer.block_norms(step_idx=len(engine.snapshots) - 1)
cycle_metrics = analyzer.cycle_metrics(step_idx=len(engine.snapshots) - 1)

report = AnalysisReport(analyzer)
summary = report.to_json(step_idx=len(engine.snapshots) - 1)
report.to_csv("results/analysis", step_idx=len(engine.snapshots) - 1)
```

`block_norms()` currently returns the infinity norms for the internal `A`, `B`,
and `P` blocks. `cycle_metrics()` returns a compact dictionary with cycle count
information.

## Visualize

```python
from propflow.snapshots import SnapshotVisualizer

viz = SnapshotVisualizer(engine.snapshots)

viz.plot_global_cost(show=True)
viz.plot_message_norms(message_type="Q", show=True)
viz.plot_message_norms(message_type="R", show=True)
viz.plot_assignment_heatmap(show=True)
viz.plot_argmin_per_variable(layout="combined", show=True)
viz.plot_cost_tables(show=True)

# Single cost table, using a factor name from the snapshot metadata.
fig = viz.plot_cost_tables(factor="f12", show=False)
```

For Backtrack Cost Tree work, `plot_bct()` reconstructs the BCT from recorded
Q/R messages and cost tables:

```python
creator = viz.plot_bct("x1", steps_back=5, show=False)
for cost_key, coeff in creator.cost_contributions().items():
    print(cost_key, coeff)
```

## Persist

Snapshots contain NumPy arrays, so convert only the fields you need before JSON
serialization.

```python
import json
from pathlib import Path

out_dir = Path("results/snapshots")
out_dir.mkdir(parents=True, exist_ok=True)

for snap in engine.snapshots:
    payload = {
        "step": snap.step,
        "global_cost": snap.global_cost,
        "assignments": snap.assignments,
        "metadata": snap.metadata,
    }
    (out_dir / f"snapshot_{snap.step:04d}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
```

Use `AnalysisReport.to_csv()` when you want structured analyzer output; it writes
`beliefs.csv` and `metrics.json` into the target directory.

## Common Uses

- Debug non-convergence by plotting global cost and assignment heatmaps.
- Compare policy engines by running the same graph with different engines and
  plotting cost trajectories from their snapshots.
- Inspect large messages with `plot_message_norms()`.
- Verify local dynamics with `SnapshotAnalyzer.jacobian()` and `block_norms()`.
