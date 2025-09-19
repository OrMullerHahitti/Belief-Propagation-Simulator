# Engine Snapshot Recorder Guide

The snapshot recorder captures everything you need to inspect a Belief Propagation (BP) run without modifying engine internals. Use it when you want to debug message flow, measure convergence, or build custom analytics dashboards.

## Quick Start

1. Ensure your PropFlow environment is installed (via `uv sync` or `pip install -e .[dev]`).
2. Instantiate your BP engine (any subclass with `step`, `assignments`, and `calculate_global_cost`).
3. Wrap the engine with the recorder:

```python
from analyzer.snapshot_recorder import EngineSnapshotRecorder

recorder = EngineSnapshotRecorder(engine)
recorder.record_run(max_steps=50)
```

### Example

- `examples/analyzer_ring_demo.py` builds a 4-variable ring, records 12 iterations, saves the snapshot JSON, and produces both per-variable and combined argmin plots using `SnapshotVisualizer` (the combined figure is written with a `_combined` suffix by default).

## What Gets Captured

Each recorded iteration contains:

- `step`: zero-based iteration index.
- `messages`: every message sent during the iteration with
  - sender and recipient names
  - the raw `values` array as a Python list
  - `argmin_index` pointing to the minimal entry
  - a `neutral` flag when multiple entries tie for the minimum
  - `flow` indicating variable→factor or factor→variable direction
- `assignments`: current variable assignments (`engine.assignments`).
- `cost`: global assignment cost (`engine.calculate_global_cost`).
- `neutral_messages`: count of neutral messages in the step.
- `step_neutral`: `True` if every message is neutral (flat update).

## Running With Convergence Control

```python
snapshots = recorder.record_run(
    max_steps=200,
    break_on_convergence=True,
)
```

If the engine exposes `_is_converged`, the run stops early when it reports convergence. Set `reset=False` to append to an existing trace.

## Persisting to JSON

```python
recorder.to_json("results/my_run.json")
```

The recorder writes JSON that can be loaded in pandas, notebooks, or visualisation tools. Arrays are stored as lists for maximum compatibility.

## Integrating With Simulations

You can call `record_run` inside higher-level experiments and retain the recorder’s `snapshots` list for reporting. For batch runs, instantiate a new recorder per engine to keep traces isolated.

## Interpreting Neutral Messages

Neutral messages typically indicate plateaus: multiple assignments sharing the same cost. Spotting successive `step_neutral=True` entries helps diagnose stalled runs or symmetry in the factor graph.

## Extending

The recorder is intentionally small: subclass it or wrap `_capture_step` if you need extra metadata (e.g., timing, custom metrics). Messages are ordinary dictionaries, so downstream consumers can add fields in place.
