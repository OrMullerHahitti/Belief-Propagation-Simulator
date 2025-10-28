# PropFlow Snapshot Pipeline How-To

This guide walks through the complete snapshot pipeline in PropFlow from the top down.  
It starts with the big-picture architecture, then dives into individual components, and
finishes with concrete instructions for capturing, storing, and analysing simulation snapshots.

---

## 1. Architecture Overview

```
FactorGraph + BPEngine
        │
        │  (SnapshotConfig attached)
        ▼
SnapshotManager.capture_step(...)
        │
        ├─▶ In-memory ring buffer of SnapshotRecord objects
        └─▶ Optional filesystem export (meta.json + NPZ blobs + manifest)
```

- **BPEngine** orchestrates message passing on a factor graph. When initialised with a
  `SnapshotsConfig`, it instantiates a `SnapshotManager`.
- **SnapshotManager** is invoked at every engine step, reconstructing a `SnapshotData`
  view of the graph, messages, and runtime metrics. Optional analyses (Jacobians, cycle
  metrics, block norms) are computed on demand.
- **Persistence layer** writes JSON metadata and binary NPZ blobs per step, plus an
  `index.json` that summarises the run for downstream tooling.
- **Analyzer/reporting stack** (under `src/analyzer/reporting`) can ingest either the
  in-memory records or the exported artefacts to compute diagnostics, generate plots,
  and drive notebooks.

---

## 2. Component Breakdown

### 2.1 SnapshotsConfig (`src/propflow/snapshots/types.py`)

- Controls which analyses run (Jacobians, block norms, cycle metrics).
- Configures retention (`retain_last` in-memory slots) and persistence behaviour
  (`save_each_step`, `save_dir`).
- Sets cycle enumeration limits and optional numeric gain estimates.

### 2.2 SnapshotData (`src/propflow/snapshots/types.py`)

Structured view of a single engine step:

- Graph structure: domains, variable → factor neighbours, factor → variable neighbours.
- Message tables: `Q` (variable→factor) and `R` (factor→variable).
- Runtime context: damping factor, unary potentials, beliefs, assignments, global cost.
- Metadata: engine name, graph diameter, history mode, convergence/performance summaries, capture timestamps.

### 2.3 SnapshotRecord (`src/propflow/snapshots/types.py`)

- Wraps `SnapshotData` plus optional Jacobian artefacts, cycle metrics, inferred winners
  per message, argmin indices, and capture timestamp.
- The manager buffers records and exposes convenience helpers via
  `propflow.snapshots.utils`.

### 2.4 Builder (`src/propflow/snapshots/builder.py`)

- `build_snapshot_from_engine(step_idx, step, engine)` retrieves domains, neighbourhoods,
  Q/R messages, cost accessors, beliefs, assignments, global cost, and metadata.
- Handles normalisation of Q messages, scalarisation of beliefs/assignments, and safe
  conversion of NumPy objects to built-ins for serialisation.

### 2.5 SnapshotManager (`src/propflow/snapshots/manager.py`)

- Maintains a configurable ring buffer of `SnapshotRecord` instances.
- Computes Jacobians (`A`, `P`, `B`), block norms, winners, and cycle metrics as required.
- Persists each step to disk with:
  - `meta.json`: structured context (`context`, `graph`, `messages`, `runtime`,
    `analysis`, `metadata`).
  - `messages_q.npz`, `messages_r.npz`, `unary.npz`: compressed numeric blobs.
  - Optional `A.npz`, `P.npz`, `B.npz` for Jacobians.
- Disk writes happen only when you call `save_step(..., save=True)` or when
  `SnapshotsConfig.save_each_step=True`, which internally passes `save=True`. Otherwise
  snapshots reside solely in the in-memory buffer attached to the engine.
- Maintains an `index.json` manifest summarising available steps, lambda values,
  message counts, contraction bounds, and cycle statistics.

### 2.6 Snapshot Storage Layout

Each saved step directory (`step_XXXX/`) contains:

| File | Description | Key Contents |
| ---- | ----------- | ------------ |
| `meta.json` | Human-readable metadata structured into `context`, `graph`, `messages`, `runtime`, `analysis`, `metadata`. | Step index, lambda, timestamps, domain lists, adjacency, NPZ index maps, beliefs, assignments, global cost, block norms, cycle metrics, winners, recorded-at timestamp. |
| `messages_q.npz` | Compressed NumPy archive for Q messages. | Arrays keyed by sanitised `var->factor` labels; `meta.json["messages"]["Q"]["index"]` maps back to original edge names. |
| `messages_r.npz` | Compressed NumPy archive for R messages. | Arrays keyed by `factor->var` labels with the same index indirection. |
| `unary.npz` | Compressed unary potentials. | Optional arrays per variable (empty NPZ omitted). |
| `A.npz`, `P.npz`, `B.npz` | Jacobian blocks when `compute_jacobians` is enabled. | CSR matrices saved via `scipy.sparse.save_npz`. |

At the run root, `index.json` summarises all persisted steps:

```json
{
  "generated_at": "...",
  "steps": [
    {
      "step": 5,
      "dir": "step_0005",
      "timestamp": "...",
      "lambda": 0.9,
      "cost": 12.5,
      "messages": {"Q": 18, "R": 18},
      "has_jacobians": true,
      "has_cycles": true,
      "block_norm_upper": 0.87,
      "num_cycles": 6,
      "aligned_hops_total": 2
    }
  ]
}
```

You can recover the exact message arrays by opening the NPZ files with `numpy.load`
and using the index map in `meta.json` to translate back to graph edge names.

### 2.7 Analyzer & Reporting Utilities

- `analyzer.reporting` parses stored snapshots and computes derived analytics,
  producing JSON/CSV outputs or visualisations.
- `analyzer.snapshot_visualizer` and notebooks (e.g.
  `notebooks/snapshot_cycle_lemniscate_demo.ipynb`) demonstrate plotting belief argmin
  trajectories and inspecting metadata.

---

## 3. Step-by-Step Usage

### 3.1 Enable Snapshots on an Engine

```python
from pathlib import Path
from propflow.bp.engine_base import BPEngine
from propflow.snapshots.types import SnapshotsConfig
from propflow.utils.fg_utils import FGBuilder
from propflow.configs import create_random_int_table

graph = FGBuilder.build_cycle_graph(
    num_vars=6,
    domain_size=3,
    ct_factory=create_random_int_table,
    ct_params={"low": 1, "high": 7},
)

snapshot_cfg = SnapshotsConfig(
    compute_jacobians=True,
    compute_cycles=True,
    include_detailed_cycles=True,
    retain_last=50,
    save_each_step=True,
    save_dir=str(Path("results/snapshots_run")),
)

engine = BPEngine(
    factor_graph=graph,
    snapshots_config=snapshot_cfg,
    use_bct_history=True,  # enables step-level beliefs/assignments/cost tracking
)
```

### 3.2 Run the Simulation

```python
engine.run(max_iter=10)
```

- The `SnapshotManager` automatically captures each step.
- `results/snapshots_run/index.json` lists the exported steps.
- `results/snapshots_run/step_0009/meta.json` (for step 9) contains structured metadata,
  while `messages_q.npz` / `messages_r.npz` store the raw arrays.

### 3.3 Inspect In-Memory Snapshots

```python
from propflow.snapshots.utils import latest_snapshot

record = latest_snapshot(engine)
print(record.data.metadata["graph_diameter"])
print(record.jacobians.block_norms if record.jacobians else "No Jacobians")
```

- `record.data.beliefs` and `record.data.assignments` mirror the history tracker when
  BCT mode is on.
- `record.cycles` summarises simple cycles and contraction hints if enabled.

Need to persist a subset of steps after a run? Use the engine’s saver helpers:

```python
target_dir = Path("results/snapshots_run")
latest = engine.latest_snapshot()
if latest:
    json_file = engine.save_snapshot.save_json(
        target_dir / f"snapshot_step_{latest.data.step:04d}.json",
        step=latest.data.step,
    )
    csv_file = engine.save_snapshot.save_csv(
        target_dir / "snapshot_summary.csv",
        step=latest.data.step,
    )
    print("Saved JSON to:", json_file)
    print("Updated CSV summary:", csv_file)
```

The saver works with any retained step (or all of them when `step` is omitted) and keeps the in-memory cache untouched.

### 3.4 Load Stored Snapshots for Analysis

```python
import json
from pathlib import Path

base = Path("results/snapshots_run")
manifest = json.loads((base / "index.json").read_text())
for entry in manifest["steps"]:
    step_dir = base / entry["dir"]
    meta = json.loads((step_dir / "meta.json").read_text())
    print(entry["step"], meta["analysis"]["block_norms"])
```

- To access the raw message values:

```python
import numpy as np

q_info = meta["messages"]["Q"]
with np.load(step_dir / q_info["file"]) as q_npz:
    for edge, key in q_info["index"].items():
        values = q_npz[key]
        print(edge, values)
```

- Metadata is organised for streaming; large arrays stay in NPZ form.
- `analysis.winners` is grouped by factor→variable edges for quick lookup of winning
  assignments.

### 3.5 Feed Snapshots into the Analyzer

```python
from analyzer.reporting import SnapshotAnalyzer, parse_snapshots
from analyzer.reporting.snapshot_parser import load_snapshots

raw = load_snapshots("path/to/engine_recorder_output.json")  # or convert meta.json payloads
typed = parse_snapshots(raw)
analyzer = SnapshotAnalyzer.from_records(typed)
report = analyzer.analysis_report(step_idx=len(typed) - 1)
print(report.to_json(step_idx=len(typed) - 1))
```

- Use `_from_engine_snapshot_manager` to adapt in-memory records if needed:

```python
from analyzer.reporting.snapshot_parser import from_engine_snapshot_manager

record = latest_snapshot(engine)
typed_record = from_engine_snapshot_manager(record)
```

### 3.6 Visualise Results (Notebook)

- Open `notebooks/snapshot_cycle_lemniscate_demo.ipynb`.
- It demonstrates:
  1. Building cycle and lemniscate graphs.
  2. Running the engine with snapshots enabled.
  3. Inspecting `index.json`/`meta.json`.
  4. Plotting factor graph layouts and cost trajectories.

---

## 4. Testing the Pipeline

- Unit tests under `tests/test_snapshots_module.py` verify:
  - Builder metadata enrichments (beliefs, assignments, costs, convergence/performance summaries).
  - Manager persistence (NPZ blobs, structured metadata, manifest entries).
  - Winner serialisation and helper utilities.
- To run the focused suite:

```bash
pytest tests/test_snapshots_module.py
```

- For end-to-end confidence, execute the full regression suite (`pytest -q`) once
  core changes are in place.

---

## 5. Tips & Best Practices

- **Keep NPZ files**: They drastically reduce storage and load times compared to JSON
  arrays, especially for dense message tensors and Jacobian matrices.
- **Control retention**: Use `retain_last` to cap in-memory footprint when running long
  simulations, relying on disk snapshots for historical inspection.
- **Reproducibility**: Seed cost-table factories (e.g., via `np.random.seed`) so saved
  runs can be compared across machines.
- **Analyzer alignment**: The metadata layout aligns with the analyzer’s expectations,
  removing the need for recomputation of Jacobians or cycle metrics during reporting.
- **Notebooks for prototyping**: The new demo notebook offers a template for bespoke
  diagnostics—duplicate it and adjust graph construction or plotting routines for
  project-specific analyses.

---

With these pieces in place you can capture rich per-step state, persist efficient
artefacts, and feed them into PropFlow’s analytics stack for deeper insight into
belief-propagation behaviour. Happy analysing!
