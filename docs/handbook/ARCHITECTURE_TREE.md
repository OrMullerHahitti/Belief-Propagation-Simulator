# PropFlow Architecture and Usage Tree

This page maps the current `propflow` implementation and the recommended user
surface.

## High-Level Flow

1. Build or load a `FactorGraph`.
2. Choose a computator and engine.
3. Run `engine.run(max_iter=N)`.
4. Inspect `engine.assignments`, `engine.calculate_global_cost()`,
   `engine.history.costs`, and `engine.snapshots`.
5. Use `Simulator` for batches and `propflow.snapshots` for analysis.

## Package Tree

```text
src/propflow
├── __init__.py                 # Public package exports
├── _version.py                 # Package version
├── cli.py                      # bp-sim --version/help
├── simulator.py                # Batch runner over engine configs and graphs
├── bp/
│   ├── computators.py          # BPComputator, Min/Max Sum/Product variants
│   ├── engine_base.py          # BPEngine synchronous loop
│   ├── engine_components.py    # Step plus history compatibility view
│   ├── engines.py              # Engine policy variants
│   └── factor_graph.py         # FactorGraph wrapper around NetworkX
├── core/
│   ├── agents.py               # VariableAgent, FactorAgent
│   ├── components.py           # Message, MailHandler, CostTable alias
│   ├── dcop_base.py            # Base Agent / Computator primitives
│   └── protocols.py            # Protocol definitions
├── configs/
│   ├── global_config_mapping.py # Defaults, CT factories, validators
│   └── loggers.py              # Logger wrapper
├── policies/
│   ├── convergance.py          # ConvergenceConfig, ConvergenceMonitor
│   ├── cost_reduction.py
│   ├── damping.py
│   ├── message_pruning.py
│   ├── normalize_cost.py
│   └── splitting.py
├── snapshots/
│   ├── analyzer.py             # SnapshotAnalyzer, AnalysisReport
│   ├── builder.py              # Builds EngineSnapshot from a Step
│   ├── manager.py              # SnapshotManager capture hook
│   ├── step_formatter.py       # Human-readable snapshot formatting
│   ├── types.py                # EngineSnapshot, Jacobians, CycleMetrics
│   ├── utils.py
│   └── visualizer.py           # SnapshotVisualizer and cost-table plots
├── utils/
│   ├── fg_utils.py             # FGBuilder and graph helpers
│   ├── create/                 # Experimental config/pickle helpers
│   ├── tools/                  # BCT, performance, drawing, convex hull tools
│   └── ...
└── engines/, computators/      # Convenience import packages
```

## Public API

Most user code should import from `propflow`:

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
```

Cost-table helpers live in `propflow.configs`:

```python
from propflow.configs import (
    CTFactories,
    create_poisson_table,
    create_random_int_table,
    create_uniform_float_table,
    get_ct_factory,
)
```

Snapshot tooling lives in `propflow.snapshots`:

```python
from propflow.snapshots import AnalysisReport, SnapshotAnalyzer, SnapshotVisualizer
```

## Factor Graphs

`FactorGraph(variable_li, factor_li, edges)` stores:

- `variables`: list of `VariableAgent`
- `factors`: list of `FactorAgent`
- `G`: NetworkX bipartite graph
- `edges`: reconstructed ordered mapping from factors to variables
- `original_factors`: deep copy used for comparable global-cost evaluation

Manual edge order matters. For `edges={f12: [x1, x2]}`, axis 0 of `f12`'s cost
table belongs to `x1`, and axis 1 belongs to `x2`.

Prefer `FGBuilder` for standard topologies:

- `build_cycle_graph`
- `build_random_graph`
- `build_lemniscate_graph`
- `build_with_unary_costs`

## Engines

`BPEngine` signature:

```python
BPEngine(
    factor_graph,
    computator=MinSumComputator(),
    init_normalization=dummy_func,
    name="BPEngine",
    convergence_config=None,
    monitor_performance=None,
    normalize_messages=None,
    anytime=None,
    snapshot_manager=None,
)
```

Step phases:

1. Variables compute and send Q messages.
2. Variable inboxes are cleared/prepared.
3. Factors compute and send R messages.
4. Factor inboxes are cleared/prepared.
5. Global cost is updated.
6. `SnapshotManager` captures an `EngineSnapshot`.
7. Convergence checks run at graph-diameter intervals.

Current top-level engine variants:

- `DampingEngine`
- `RDampingEngine`
- `QRDampingEngine`
- `DiffusionEngine`
- `SplitEngine`
- `MidRunSplitEngine`
- `CostReductionOnceEngine`
- `DampingCROnceEngine`
- `DampingSCFGEngine`
- `TRWEngine`
- `DampingTRWEngine`
- `MessagePruningEngine`

## Computators

`BPComputator` implements vectorized Q/R message updates and exposes:

- `compute_Q(messages)`
- `compute_R(cost_table, incoming_messages)`
- `compute_belief(messages, domain)`
- `get_assignment(belief)`

Provided variants:

- `MinSumComputator`: reduce by min, combine by add.
- `MaxSumComputator`: reduce by max, combine by add.
- `SumProductComputator`: reduce by sum, combine by multiply.
- `MaxProductComputator`: reduce by max, combine by multiply.

## History and Snapshots

`engine.history` is a `SnapshotHistoryView`, not an independent mutable recorder.
It derives compatibility data from `engine.snapshots`.

Useful outputs:

- `engine.history.costs`
- `engine.history.beliefs`
- `engine.history.assignments`
- `engine.history.get_bct_data()`
- `engine.snapshots`
- `engine.latest_snapshot()`
- `engine.get_snapshot(step_index)`
- `engine.snapshot_map`

`SnapshotAnalyzer` computes belief trajectories, difference coordinates,
Jacobians, block norms, cycle counts, and report exports from the snapshot list.

## Simulator

`Simulator(engine_configs, log_level=None, seed=None)` accepts configs like:

```python
engine_configs = {
    "baseline": {"class": BPEngine},
    "damped": {"class": DampingEngine, "damping_factor": 0.9},
}
```

`run_simulations(graphs, max_iter=None)` returns:

```python
dict[str, list[list[float]]]
```

Each key is an engine name and each value is one global-cost series per graph.

## Extension Points

- Add a new engine by subclassing `BPEngine` and overriding hooks such as
  `post_init`, `post_var_compute`, `pre_factor_compute`, or
  `post_factor_compute`.
- Add a new computator by implementing the `BPComputator` method surface.
- Add a new policy under `src/propflow/policies` and invoke it from an engine
  hook.
- Add a new cost factory by defining a callable and registering it in
  `CT_FACTORIES` / `CTFactories`, or pass the callable directly to `FGBuilder`.
- Add a new topology by extending `FGBuilder` with a helper that returns a fully
  initialized `FactorGraph`.

## Minimal Run

```python
import numpy as np

from propflow import BPEngine, FactorAgent, FactorGraph, VariableAgent

def table(num_vars: int, domain_size: int, **_):
    return np.array([[0.0, 1.0], [1.0, 0.0]])

x1 = VariableAgent("x1", 2)
x2 = VariableAgent("x2", 2)
f12 = FactorAgent("f12", 2, ct_creation_func=table)

fg = FactorGraph(variable_li=[x1, x2], factor_li=[f12], edges={f12: [x1, x2]})
engine = BPEngine(fg)
engine.run(max_iter=20)

print(engine.history.costs[-1], engine.assignments)
```
