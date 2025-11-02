# PropFlow Snapshots: Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [What Are Snapshots](#what-are-snapshots)
3. [Snapshot Data Structure](#snapshot-data-structure)
4. [Configuration](#configuration)
5. [Capturing and Accessing Snapshots](#capturing-and-accessing-snapshots)
6. [Analyzing Snapshots](#analyzing-snapshots)
7. [Visualizing Snapshots](#visualizing-snapshots)
8. [Exporting and Persisting](#exporting-and-persisting)
9. [Use Cases and Examples](#use-cases-and-examples)

---

## Overview

**Snapshots** are periodic captures of the complete simulation state at each iteration of a belief propagation algorithm. They provide a detailed window into how messages, beliefs, and costs evolve throughout a run, enabling deep analysis of convergence dynamics, cycle behavior, and message-passing patterns.

Snapshots are:
- **Lightweight**: Stored efficiently in memory or on disk
- **Flexible**: Configurable to capture only the data you need
- **Composable**: Can be combined with analysis tools for convergence, Jacobian, and cycle metrics
- **Portable**: Can be persisted and reloaded for later analysis

### Why Use Snapshots?

- **Debug algorithm behavior**: Understand how beliefs and costs evolve step-by-step
- **Analyze convergence**: Detect when and how quickly the algorithm converges
- **Study message patterns**: Examine variable-to-factor and factor-to-variable messages
- **Investigate cycles**: Find feedback loops that affect convergence
- **Compare configurations**: Run the same problem with different engine settings and compare trajectories
- **Track assignments**: Monitor how variable assignments change over time

---

## What Are Snapshots

A **snapshot** is a frozen view of the belief propagation algorithm's state at a single iteration. Unlike traditional "history" which stores only final results, snapshots capture:

- **Message values**: All Q (variable→factor) and R (factor→variable) messages
- **Runtime state**: Current variable assignments and beliefs
- **Graph structure**: Variable and factor neighbors
- **Metadata**: Damping factors, cost values, convergence metrics

### Types of Information Captured

```
EngineSnapshot
├── Core state
│   ├── Messages: Q (variable→factor) and R (factor→variable) arrays
│   ├── Beliefs/Assignments: Current variable marginals and argmins
│   ├── Graph topology: Variable/factor neighbourhoods and domains
│   ├── Costs: Global cost plus optional factor cost tables and labels
│   └── Metadata: Engine configuration hints, timestamps, damping factor
├── Jacobians (optional)
│   ├── Matrices A, P, B: Linearised message dependencies
│   └── Block norms: Convergence certification metrics
├── CycleMetrics (optional)
│   ├── Cycle count: Total number of feedback loops
│   ├── Aligned hops: Cycles amenable to contraction analysis
│   └── Details: Per-cycle properties when deeper analysis is enabled
├── Winners (optional)
│   └── Factor-variable assignment preferences
└── Min indices (optional)
    └── Argmin for each Q message
```

---

## EngineSnapshot Structure

Each :class:`~propflow.snapshots.EngineSnapshot` holds the complete state captured
after a single belief propagation iteration. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Iteration number |
| `lambda_` | `float` | Damping factor at this step |
| `dom` | `Dict[str, List[str]]` | Variable domain labels: `{var_name: ["0", "1", ...]}` |
| `N_var` | `Dict[str, List[str]]` | Variable neighborhoods: `{var: [factor_neighbors]}` |
| `N_fac` | `Dict[str, List[str]]` | Factor neighborhoods: `{factor: [variable_neighbors]}` |
| `Q` | `Dict[(str, str), ndarray]` | Variable→factor messages: `{(var, factor): [msg_values]}` |
| `R` | `Dict[(str, str), ndarray]` | Factor→variable messages: `{(factor, var): [msg_values]}` |
| `cost_tables` | `Dict[str, ndarray]` | Optional factor cost tensors used for diagnostics |
| `cost_labels` | `Dict[str, List[str]]` | Variable ordering for each stored cost table |
| `unary` | `Dict[str, ndarray]` | Unary potential per variable (usually zeros) |
| `beliefs` | `Dict[str, ndarray]` | Current belief vectors per variable |
| `assignments` | `Dict[str, int]` | Current assignment (argmin) per variable |
| `global_cost` | `float` (optional) | Total cost across all factors |
| `metadata` | `Dict[str, Any]` | Additional info: engine type, convergence status, etc. |
| `jacobians` | `Jacobians` or `None` | Optional A/P/B matrices with index maps and block norms |
| `cycles` | `CycleMetrics` or `None` | Optional cycle counts and contraction metadata |
| `min_idx` | `Dict[(str, str), int]` or `None` | Optional argmin indices per Q message |
| `captured_at` | `datetime` | Timestamp (UTC) when the snapshot was created |


## Engine Integration

Snapshots are captured automatically by :class:`propflow.bp.engine_base.BPEngine`.
Every call to :meth:`engine.step` appends an :class:`EngineSnapshot` to
``engine.snapshots``; no additional configuration object is required. The
engine exposes convenience helpers such as :meth:`engine.latest_snapshot` and
the ``engine.snapshot_map`` dictionary for step-based lookups.

```python
from propflow import BPEngine, DampingEngine

engine = DampingEngine(
    factor_graph=graph,
    damping_factor=0.9,
    use_bct_history=True,  # Optional: retain per-step message traces for BCT tools
)

engine.run(max_iter=100)

latest = engine.latest_snapshot()
print(len(engine.snapshots))
```

Set ``use_bct_history=True`` when constructing an engine if you need message
trajectories for Backtrack Cost Tree analysis or external tooling.

---

## Capturing and Accessing Snapshots

### During Simulation

Snapshots are automatically captured at each iteration when configured:

```python
engine = BPEngine(factor_graph=graph)
engine.run(max_iter=100)

# At this point, engine has captured up to 100 snapshots (or retain_last worth)
```

### Accessing Snapshots After Simulation

```python
from propflow.snapshots.utils import (
    get_snapshot,
    latest_snapshot,
    latest_jacobians,
    latest_cycles,
    latest_winners,
)

# Get a specific step's snapshot
snapshot_at_step_5 = get_snapshot(engine, 5)
print(snapshot_at_step_5.step)        # 5
print(snapshot_at_step_5.assignments) # {"x1": 0, "x2": 1, ...}
print(snapshot_at_step_5.global_cost) # 42.5

# Get the most recent snapshot
latest = latest_snapshot(engine)
print(latest.step)

# Get analysis artifacts from the latest snapshot
jac = latest_jacobians(engine)
cycles = latest_cycles(engine)
winners = latest_winners(engine)
```

### Collecting All Snapshots

```python
# Gather all snapshots captured during the run
all_snapshots = list(engine.snapshots)
print(f"Total snapshots: {len(all_snapshots)}")
```

---

## Analyzing Snapshots

### 1. Belief and Assignment Trajectories

Track how variable beliefs and assignments evolve:

```python
from propflow.snapshots import SnapshotAnalyzer

# Create analyzer from snapshots
analyzer = SnapshotAnalyzer(all_snapshots)

# Get belief trajectories (argmin over messages)
beliefs = analyzer.beliefs_per_variable()
print(beliefs["x1"])  # [0, 0, 1, 1, 2, 2, ...] - assignment over time

# Or manually extract from snapshots
beliefs_manual = {}
for var in ["x1", "x2"]:
    beliefs_manual[var] = [
        snap.assignments.get(var)
        for snap in all_snapshots
    ]
```

### 2. Convergence Analysis (BCT)

Analyze how each variable's belief evolved and when it converged:

```python
from propflow.snapshots import SnapshotVisualizer

visualizer = SnapshotVisualizer(all_snapshots)

# Create and return a BCT creator object
bct_creator = visualizer.plot_bct("x1", show=True)

# Analyze convergence for a variable
analysis = bct_creator.analyze_convergence("x1")
print(f"Variable x1 converged: {analysis['converged']}")
print(f"Final belief: {analysis['final_belief']}")
print(f"Total change: {analysis['total_change']}")
print(f"Convergence iteration: {analysis['convergence_iteration']}")

# Compare multiple variables
comparison = bct_creator.compare_variables(["x1", "x2", "x3"])
print(comparison["summary"]["all_converged"])

# Export detailed analysis
bct_creator.export_analysis("bct_analysis.json")
```

### 3. Jacobian and Block Norms

Examine linearized dynamics and convergence bounds:

```python
from propflow.snapshots.utils import latest_jacobians

# Get Jacobian for latest snapshot
jac = latest_jacobians(engine)

if jac:
    # Check convergence certification via block norms
    norms = jac.block_norms
    print(f"||BPA||_inf = {norms['||BPA||_inf']:.4f}")
    print(f"||B||_inf = {norms['||B||_inf']:.4f}")
    print(f"||PA||_inf = {norms['||PA||_inf']:.4f}")
    print(f"||M||_inf_upper = {norms['||M||_inf_upper']:.4f}")

    # If ||M||_inf_upper < 1.0, convergence is certified
    if norms['||M||_inf_upper'] < 1.0:
        print("✓ Convergence certified!")

    # Access raw matrices (sparse CSR format)
    print(jac.A.shape, jac.P.shape, jac.B.shape)
```

### 4. Cycle Analysis

Investigate feedback loops in the message-passing graph:

```python
from propflow.snapshots.utils import latest_cycles

cycles = latest_cycles(engine)

if cycles:
    print(f"Total cycles: {cycles.num_cycles}")
    print(f"Cycles with aligned hops: {cycles.aligned_hops_total}")
    print(f"Contraction certified: {cycles.has_certified_contraction}")

    # Per-cycle details (if enabled in config)
    if cycles.details:
        for i, detail in enumerate(cycles.details[:5]):
            print(f"Cycle {i}: length={detail['length']}, aligned={detail['aligned']}")
```

### 5. Analysis Report

Generate a comprehensive summary:

```python
from propflow.snapshots import AnalysisReport

report = AnalysisReport(analyzer)

# Get summary at a specific step
summary_at_last = report.to_json(step_idx=len(all_snapshots) - 1)
print(summary_at_last["block_norms"])
print(summary_at_last["cycle_metrics"])
```

---

## Visualizing Snapshots

### 1. Belief Trajectories

Plot how variable assignments evolve:

```python
from propflow.snapshots import SnapshotVisualizer

visualizer = SnapshotVisualizer(all_snapshots)

# Get all variable names
variables = visualizer.variables()
print(f"Variables: {variables}")

# Plot trajectories for a subset
visualizer.plot_argmin_per_variable(
    vars_filter=variables[:6],
    figsize=(10, 12),
    show=True,
    savepath="belief_trajectories.png"
)
```

### 2. Backtrack Cost Trees (BCT)

Visualize how costs backtrack through iterations:

```python
# Plot BCT for a single variable
bct_creator = visualizer.plot_bct(
    "x1",
    iteration=None,  # Use -1 (last iteration)
    show=True,
    savepath="bct_x1.png"
)

# The returned BCTCreator can be reused for analysis
analysis = bct_creator.analyze_convergence("x1")
```

### 3. Argmin Series

Extract and manually plot belief trajectories:

```python
series = visualizer.argmin_series(vars_filter=["x1", "x2"])
# series = {"x1": [0, 0, 1, 1, 2, ...], "x2": [1, 1, 1, 2, 2, ...]}

import matplotlib.pyplot as plt

for var, trajectory in series.items():
    plt.plot(range(len(trajectory)), trajectory, label=var, marker="o")

plt.xlabel("Iteration")
plt.ylabel("Assignment")
plt.legend()
plt.show()
```

---

## Exporting and Persisting

### Save Individual Snapshots

Snapshots live in memory under ``engine.snapshots``; serialise them manually or
persist snapshots by serialising ``engine.snapshots`` to JSON for reproducibility
turn-key, disk-backed workflow.

```python
from pathlib import Path
import json

out_dir = Path("snapshot_output")
out_dir.mkdir(exist_ok=True)

for snapshot in engine.snapshots:
    payload = {
        "step": snapshot.step,
        "global_cost": snapshot.global_cost,
        "assignments": snapshot.assignments,
        "metadata": snapshot.metadata,
    }
    (out_dir / f"snapshot_step_{snapshot.step:04d}.json").write_text(json.dumps(payload, indent=2))
```

### Snapshot Recorder Helper

Manual serialisation is sufficient in most cases; store snapshots as JSON if you
need to revisit them later.

### Example Figures

.. figure:: figures/global_cost.png
   :alt: Global cost trajectory example
   :width: 65%

   Example global cost trajectory produced by ``tools/generate_snapshot_figures.py``.

.. figure:: figures/message_norms_q.png
   :alt: Q message norms example
   :width: 65%

   L2 norms of selected Q messages across iterations.

.. figure:: figures/message_norms_r.png
   :alt: R message norms example
   :width: 65%

   L∞ norms of selected R messages across iterations.

.. figure:: figures/assignment_heatmap.png
   :alt: Assignment heatmap example
   :width: 65%

   Assignment heatmap highlighting per-variable argmin trajectories.

### Snapshot Directory Structure

```
results/snapshots/
├── index.json                    # Manifest of all saved steps
├── step_0000/
│   ├── meta.json                # Metadata, analysis results
│   ├── messages_q.npz           # Q messages
│   ├── messages_r.npz           # R messages
│   ├── unary.npz                # Unary potentials
│   ├── A.npz, P.npz, B.npz      # Jacobian matrices
├── step_0001/
└── ...
```

### BCT Export

```python
# Export complete BCT analysis as JSON
bct_creator.export_analysis("bct_complete_analysis.json")

# File structure:
# {
#   "metadata": {
#     "damping_factor": 0.9,
#     "total_variables": 5,
#     "total_steps": 100
#   },
#   "variable_analyses": {
#     "x1": {
#       "variable": "x1",
#       "total_iterations": 100,
#       "initial_belief": 2.5,
#       "final_belief": 0.1,
#       "converged": true,
#       "convergence_iteration": 45,
#       ...
#     },
#     ...
#   },
#   "global_data": { ... }
# }
```

---

## Use Cases and Examples

### Use Case 1: Debug Non-Convergence

**Problem**: Algorithm runs but beliefs don't stabilize.

**Solution**:

```python
visualizer = SnapshotVisualizer(all_snapshots)
variables = visualizer.variables()

# Check belief trajectories
series = visualizer.argmin_series(vars_filter=variables[:3])
for var, traj in series.items():
    if len(set(traj[-10:])) > 1:  # Last 10 still oscillating?
        print(f"⚠ {var} is still oscillating!")
        visualizer.plot_argmin_per_variable(vars_filter=[var], show=True)

# Check cycle metrics
cycles = latest_cycles(engine)
if cycles and cycles.num_cycles > 0 and not cycles.has_certified_contraction:
    print(f"⚠ Found {cycles.num_cycles} cycles, contraction not certified")
    print(f"  Aligned hops: {cycles.aligned_hops_total}")
```

### Use Case 2: Compare Two Engine Configurations

**Problem**: Which damping factor converges faster?

**Solution**:

```python
from propflow.snapshots import SnapshotVisualizer
import matplotlib.pyplot as plt

# Run two experiments
configs = [0.7, 0.9]
results = {}

for damp in configs:
    engine = DampingEngine(
        factor_graph=graph,
        damping_factor=damp,
    )
    engine.run(max_iter=100)

    results[damp] = list(engine.snapshots)

# Compare cost trajectories
fig, ax = plt.subplots()
for damp, snaps in results.items():
    costs = [s.global_cost for s in snaps if s.global_cost is not None]
    ax.plot(range(len(costs)), costs, label=f"damp={damp}")

ax.set_xlabel("Iteration")
ax.set_ylabel("Global Cost")
ax.legend()
ax.grid()
plt.show()
```

### Use Case 3: Validate Convergence Bounds

**Problem**: Need proof that algorithm will converge.

**Solution**:

```python
from propflow.snapshots import SnapshotAnalyzer

engine = BPEngine(factor_graph=graph)
engine.run(max_iter=100)

analyzer = SnapshotAnalyzer(engine.snapshots)
latest_step = engine.latest_snapshot().step
block_norms = analyzer.block_norms(latest_step)

if block_norms:
    M_upper = block_norms["||M||_inf_upper"]
    if M_upper < 1.0:
        print(f"✓ Convergence proven! ||M||_inf_upper = {M_upper:.4f} < 1.0")
    else:
        print(f"✗ Convergence not proven. ||M||_inf_upper = {M_upper:.4f} >= 1.0")
```

### Use Case 4: Analyze Per-Variable Convergence

**Problem**: Some variables converge faster than others; why?

**Solution**:

```python
bct_creator = visualizer.plot_bct("x1", show=False)

# Get analysis for all variables
all_analyses = {}
for var in visualizer.variables():
    all_analyses[var] = bct_creator.analyze_convergence(var)

# Rank by convergence speed
sorted_vars = sorted(
    all_analyses.items(),
    key=lambda item: item[1]["convergence_iteration"] or float("inf")
)

print("Convergence ranking (fastest to slowest):")
for var, analysis in sorted_vars:
    conv_iter = analysis["convergence_iteration"]
    status = "✓" if analysis["converged"] else "✗"
    print(f"{status} {var}: iteration {conv_iter}")
```

### Use Case 5: Study Message Patterns

**Problem**: Understand which factors send large messages.

**Solution**:

```python
# Analyze message magnitudes
latest = latest_snapshot(engine)
data = latest

# Factor-to-variable message magnitudes
r_magnitudes = {}
for (factor, var), r_msg in data.R.items():
    magnitude = float(np.linalg.norm(r_msg))
    key = f"{factor}->{var}"
    r_magnitudes[key] = magnitude

# Find largest messages
sorted_msgs = sorted(
    r_magnitudes.items(),
    key=lambda x: x[1],
    reverse=True
)

print("Top 10 largest R-messages:")
for msg, mag in sorted_msgs[:10]:
    print(f"  {msg}: {mag:.2f}")
```

---

## Advanced Topics

### Accessing Raw Matrices

If you need the Jacobian matrices for custom analysis:

```python
jac = latest_jacobians(engine)

# jac.A: R -> Q dependencies (sparse CSR matrix)
# jac.P: Projection for min-sum operator (sparse CSR matrix)
# jac.B: Q -> R dependencies (sparse CSR matrix)

# Convert to dense for small matrices
if jac.A.shape[0] < 100:
    A_dense = jac.A.toarray()
    print(A_dense)

# Or work directly with sparse format
from scipy.sparse import linalg
eigenvalues = linalg.eigsh(jac.A.T @ jac.A, k=1)[0]
```

### Custom Analysis with SnapshotAnalyzer

```python
from propflow.snapshots import SnapshotAnalyzer

analyzer = SnapshotAnalyzer(all_snapshots)

# Compute difference coordinates (for linearization analysis)
delta_q, delta_r = analyzer.difference_coordinates(step_idx=50)

# Construct Jacobian in difference coordinates
jac_matrix = analyzer.jacobian(step_idx=50)
```

---

## Summary

Snapshots provide a comprehensive window into belief propagation dynamics:

| Task | Tool |
|------|------|
| Track variable beliefs over time | `SnapshotVisualizer.argmin_series()` |
| Visualize belief trajectories | `SnapshotVisualizer.plot_argmin_per_variable()` |
| Analyze convergence | `BCTCreator.analyze_convergence()` |
| Prove convergence (bounds) | Check `Jacobians.block_norms["||M||_inf_upper"]` |
| Find feedback loops | `CycleMetrics` from snapshot |
| Compare configurations | Run multiple engines, collect snapshots, compare |
| Export for later analysis | `manager.save_step()` or `bct_creator.export_analysis()` |

Start with **configuration**, move to **visualization** (is algorithm converging?), then **analysis** (why/why not?), and finally **export** results for reporting.
