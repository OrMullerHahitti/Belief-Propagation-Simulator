# PropFlow User Guide

PropFlow is a research-oriented Python package for building factor graphs,
running belief propagation variants, comparing engine policies, and inspecting
per-step snapshots.

## Contents

1. [Runtime Model](#runtime-model)
2. [Build Factor Graphs](#build-factor-graphs)
3. [Run Engines](#run-engines)
4. [Inspect Results](#inspect-results)
5. [Compare Configurations](#compare-configurations)
6. [Snapshots and Analysis](#snapshots-and-analysis)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)
9. [Quick Reference](#quick-reference)

## Runtime Model

PropFlow follows a layered runtime:

1. **Agents**: `VariableAgent` and `FactorAgent` own domains, mailboxes, and
   message-computation hooks.
2. **Factor graph**: `FactorGraph` connects variables to factors in a bipartite
   NetworkX graph and initializes factor cost tables.
3. **Computator**: `MinSumComputator`, `MaxSumComputator`,
   `SumProductComputator`, or `MaxProductComputator` defines the message math.
4. **Engine**: `BPEngine` and engine variants run the synchronous BP loop.
5. **Simulator**: `Simulator` runs many engine/graph combinations and aggregates
   cost histories.
6. **Snapshots**: `EngineSnapshot` records per-step messages, beliefs,
   assignments, costs, metadata, and cost-table diagnostics.

## Build Factor Graphs

### Prefer `FGBuilder`

`FGBuilder` creates valid graphs with initialized cost tables. Its cost-table
factory can be a callable, a key such as `"random_int"`, or a function from
`CTFactories`.

```python
from propflow import FGBuilder
from propflow.configs import CTFactories

graph = FGBuilder.build_random_graph(
    num_vars=20,
    domain_size=4,
    ct_factory=CTFactories.RANDOM_INT,
    ct_params={"low": 0, "high": 25},
    density=0.3,
    seed=123,
)
```

Current builder helpers:

- `FGBuilder.build_cycle_graph(num_vars, domain_size, ct_factory, ct_params)`
- `FGBuilder.build_random_graph(..., density, seed=None)`
- `FGBuilder.build_lemniscate_graph(...)`
- `FGBuilder.build_with_unary_costs(base_graph, unary_costs)`

Built-in factory helpers:

- `create_random_int_table(n, domain, low=0, high=10)`
- `create_uniform_float_table(n, domain, low=0.0, high=1.0)`
- `create_poisson_table(n, domain, rate=1.0, strength=None)`
- `CTFactories.RANDOM_INT`, `CTFactories.UNIFORM`, `CTFactories.POISSON`
- `get_ct_factory("random_int" | "uniform_float" | "poisson")`

### Manual Graph Assembly

Use manual construction when the helper topologies are not enough. The important
invariant is that each factor key in `edges` maps to an ordered list of variables;
that order defines the tensor axes of the factor's cost table.

```python
import numpy as np

from propflow import FactorAgent, FactorGraph, VariableAgent

def prefer_matching(num_vars: int, domain_size: int, **_):
    table = np.full((domain_size,) * num_vars, 5.0)
    np.fill_diagonal(table, 0.0)
    return table

x = VariableAgent(name="x", domain=3)
y = VariableAgent(name="y", domain=3)
f_xy = FactorAgent(name="f_xy", domain=3, ct_creation_func=prefer_matching)

graph = FactorGraph(
    variable_li=[x, y],
    factor_li=[f_xy],
    edges={f_xy: [x, y]},
)
```

Manual-build checklist:

- Every factor appears once in `factor_li`.
- Every factor appears as a key in `edges`.
- Each `edges[factor]` value is an ordered variable list.
- Factory functions accept `(num_vars, domain_size, **kwargs)`.
- Use `FactorAgent.create_from_cost_table(name, cost_table)` when you already
  have a fixed NumPy table.

## Run Engines

The base engine runs synchronous BP: variables compute Q messages, factors
compute R messages, costs and snapshots are recorded, then convergence checks
run at graph-diameter intervals.

```python
from propflow import BPEngine, MinSumComputator

engine = BPEngine(
    factor_graph=graph,
    computator=MinSumComputator(),
)
engine.run(max_iter=100)
```

Computators:

- `MinSumComputator`: minimize additive cost, the default.
- `MaxSumComputator`: maximize additive utility.
- `SumProductComputator`: multiplicative combination with sum reduction.
- `MaxProductComputator`: multiplicative combination with max reduction.

Engine variants exported from `propflow`:

- `DampingEngine(damping_factor=0.9)`: damps variable-to-factor Q messages.
- `RDampingEngine(damping_factor=0.9)`: damps factor-to-variable R messages.
- `QRDampingEngine(q_damping_factor=..., r_damping_factor=...)`: damps both
  directions independently.
- `DiffusionEngine(alpha=0.3)`: spatially blends messages with same-neighborhood
  messages at the same iteration.
- `SplitEngine(split_factor=0.5)`: splits each factor at initialization.
- `MidRunSplitEngine(split_at_iter=..., transfer_mode="reset" | "transfer")`:
  injects splitting during a run.
- `CostReductionOnceEngine(reduction_factor=0.5)`: reduces factor costs at
  startup and discounts outgoing factor messages.
- `DampingSCFGEngine`: combines damping and factor splitting.
- `DampingCROnceEngine`: combines damping and one-time cost reduction.
- `TRWEngine`: tree-reweighted Min-Sum with sampled or explicit factor rhos.
- `DampingTRWEngine`: combines damping and TRW.
- `MessagePruningEngine`: initializes a pruning policy for message filtering.

Convergence can be configured with `ConvergenceConfig`:

```python
from propflow import DampingEngine
from propflow.policies import ConvergenceConfig

config = ConvergenceConfig(
    min_iterations=10,
    belief_threshold=1e-6,
    patience=5,
)

engine = DampingEngine(
    factor_graph=graph,
    damping_factor=0.85,
    convergence_config=config,
)
engine.run(max_iter=500)
```

## Inspect Results

After a run, use:

```python
print(engine.assignments)
print(engine.calculate_global_cost())
print(engine.graph.global_cost)
print(engine.iteration_count)
print(engine.get_beliefs())
```

`engine.calculate_global_cost()` evaluates the current assignment against the
original factor tables. `engine.graph.global_cost` evaluates against the
current factor tables, which may have been modified by policies such as
splitting, cost reduction, or TRW scaling.

`engine.history` is a read-only compatibility view backed by snapshots. Useful
fields include:

```python
costs = engine.history.costs
assignments_by_step = engine.history.assignments
beliefs_by_step = engine.history.beliefs
bct_payload = engine.history.get_bct_data()
```

For convergence status, inspect the monitor:

```python
summary = engine.convergence_monitor.get_convergence_summary()
print(summary.get("converged", False))
```

## Compare Configurations

`Simulator` runs every configured engine against every graph. Engine configs use
`{"class": EngineClass, ...kwargs}`.

```python
from propflow import BPEngine, DampingEngine, FGBuilder, Simulator, SplitEngine
from propflow.configs import create_random_int_table

graphs = [
    FGBuilder.build_cycle_graph(
        num_vars=10,
        domain_size=5,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 100},
    )
    for _ in range(5)
]

configs = {
    "baseline": {"class": BPEngine},
    "damped-0.9": {"class": DampingEngine, "damping_factor": 0.9},
    "split-0.6": {"class": SplitEngine, "split_factor": 0.6},
}

simulator = Simulator(configs, log_level="INFO", seed=42)
results = simulator.run_simulations(graphs, max_iter=100)
simulator.plot_results(verbose=True)
```

`results` is `dict[str, list[list[float]]]`: engine name to one cost trajectory
per graph.

## Snapshots and Analysis

Snapshots are captured automatically by `BPEngine.step()` and `BPEngine.run()`.
No `use_bct_history` flag is needed in the current implementation.

```python
from propflow.snapshots import AnalysisReport, SnapshotAnalyzer, SnapshotVisualizer

snapshots = list(engine.snapshots)
latest = engine.latest_snapshot()

print(latest.step)
print(latest.global_cost)
print(latest.assignments)
print(latest.metadata)

visualizer = SnapshotVisualizer(snapshots)
fig, payload = visualizer.plot_global_cost(show=False, return_data=True)
visualizer.plot_message_norms(message_type="R", show=False)
visualizer.plot_assignment_heatmap(show=False)

analyzer = SnapshotAnalyzer(snapshots)
print(analyzer.beliefs_per_variable())
print(analyzer.block_norms(step_idx=len(snapshots) - 1))
print(analyzer.cycle_metrics(step_idx=len(snapshots) - 1))

report = AnalysisReport(analyzer)
summary = report.to_json(step_idx=len(snapshots) - 1)
report.to_csv("results/analysis", step_idx=len(snapshots) - 1)
```

To persist a lightweight snapshot trace:

```python
import json
from pathlib import Path

out = Path("results/run_001_snapshots.json")
out.parent.mkdir(parents=True, exist_ok=True)
payload = [
    {
        "step": snap.step,
        "assignments": snap.assignments,
        "global_cost": snap.global_cost,
        "metadata": snap.metadata,
    }
    for snap in engine.snapshots
]
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

The installed CLI is `bp-sim`; it currently provides version/help output only.
There is no `bp-analyze` command in the current package.

## Common Patterns

### Build, Run, Plot

```python
from propflow import DampingEngine, FGBuilder
from propflow.configs import CTFactories
from propflow.snapshots import SnapshotVisualizer

graph = FGBuilder.build_random_graph(
    num_vars=20,
    domain_size=10,
    ct_factory=CTFactories.RANDOM_INT,
    ct_params={"low": 100, "high": 200},
    density=0.3,
    seed=7,
)

engine = DampingEngine(factor_graph=graph, damping_factor=0.9)
engine.run(max_iter=100)

viz = SnapshotVisualizer(engine.snapshots)
viz.plot_global_cost(show=True)
```

### Unary Costs

```python
import numpy as np

from propflow import FGBuilder

graph_with_priors = FGBuilder.build_with_unary_costs(
    base_graph=graph,
    unary_costs={
        "x1": np.array([0.0, 2.0, 5.0]),
        "x2": np.array([1.0, 0.0, 1.5]),
    },
)
```

### Fixed Cost Table

```python
import numpy as np

from propflow import FactorAgent

table = np.array([[0.0, 3.0], [2.0, 0.0]])
factor = FactorAgent.create_from_cost_table("f12", table)
```

## Troubleshooting

### BP Does Not Converge

- Try `DampingEngine` or `QRDampingEngine`.
- Increase `max_iter` or `ConvergenceConfig.min_iterations`.
- Inspect `engine.history.costs` and `SnapshotVisualizer.argmin_series()`.
- For cyclic graphs, remember BP is heuristic; convergence is not guaranteed.

### Cost Looks Inconsistent Across Engine Variants

- Use `engine.calculate_global_cost()` when comparing policies that modify
  cost tables.
- Use `engine.graph.global_cost` only when you intentionally want the modified
  factor tables.

### Random Experiments Are Not Reproducible

- Pass `seed=` to `FGBuilder.build_random_graph`.
- Seed `Simulator(seed=...)` so each worker uses deterministic NumPy/Python
  random seeds.
- Thread deterministic parameters through custom cost-table factories.

### Memory or Runtime Is Too High

- Reduce `num_vars`, `domain_size`, graph `density`, or `max_iter`.
- Avoid dense high-arity factors unless needed; factor tables scale as
  `domain_size ** num_vars`.
- Slice or sample `engine.snapshots` before serializing very long runs.

## Quick Reference

```python
from propflow import (
    BPEngine,
    DampingEngine,
    FactorAgent,
    FactorGraph,
    FGBuilder,
    MinSumComputator,
    Simulator,
    VariableAgent,
)
from propflow.configs import (
    CTFactories,
    create_poisson_table,
    create_random_int_table,
    create_uniform_float_table,
    get_ct_factory,
)
from propflow.policies import ConvergenceConfig
from propflow.snapshots import AnalysisReport, SnapshotAnalyzer, SnapshotVisualizer
```
