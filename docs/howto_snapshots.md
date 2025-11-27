# Snapshot How-To (Updated)

This guide explains how to work with PropFlow's lightweight snapshot system.

## 1. Capture Snapshots

Snapshots are captured automatically by `BPEngine`. Construct the engine and
run it as normal:

```python
from propflow import BPEngine, FGBuilder

graph = FGBuilder.build_cycle_graph(num_vars=6, domain_size=3, ct_factory="random_int", ct_params={"low": 0, "high": 5})
engine = BPEngine(graph, use_bct_history=True)
engine.run(max_iter=50)
```

Every call to `run` or `step` records an `EngineSnapshot`. Access them through
`engine.snapshots`, `engine.snapshot_map`, or `engine.latest_snapshot()`.

## 2. Inspect Data

```python
latest = engine.latest_snapshot()
print(latest.global_cost)
print(latest.assignments)
print(latest.metadata)
```

Each snapshot contains message dictionaries (`Q`, `R`), beliefs, assignments,
metadata, and placeholders for Jacobians/cycle metrics.

## 3. Persist Snapshots

Serialise snapshots manually:

```python
import json
from pathlib import Path

out_dir = Path("snapshots")
out_dir.mkdir(exist_ok=True)
for snap in engine.snapshots:
    payload = {
        "step": snap.step,
        "global_cost": snap.global_cost,
        "assignments": snap.assignments,
    }
    (out_dir / f"snapshot_{snap.step:04d}.json").write_text(json.dumps(payload, indent=2))
```

## 4. Analyse Snapshots

```python
from propflow.snapshots import SnapshotAnalyzer

analyzer = SnapshotAnalyzer(engine.snapshots)
block_norms = analyzer.block_norms(engine.latest_snapshot().step)
```

## 5. Visualise

```python
from propflow.snapshots import SnapshotVisualizer

viz = SnapshotVisualizer(engine.snapshots)
fig, payload = viz.plot_global_cost(show=False, return_data=True)

# Inspect cost tables (rows/cols labeled with variable names)
viz.show_cost_tables(factor="F12", step=10)   # pretty-printed single factor
viz.show_cost_tables(show=True, annotate=True)  # grid plot for all factors at last step
```

## 6. BCT Data

`SnapshotVisualizer.plot_bct()` reconstructs Backtrack Cost Trees directly from
the recorded Q/R messages and factor cost tables. Internally it uses
`propflow.utils.tools.bct.SnapshotBCTBuilder` to build the message DAG and
returns a `BCTCreator` instance that can visualise the tree or report per-leaf
coefficients via `cost_contributions()`. Provide `steps_back` or an explicit
`iteration` to anchor the root on earlier snapshots.

---

Snapshot analysis already exposes Jacobians, block norms, and cycle metrics;
future releases will extend the visual tooling further.
