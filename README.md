# Belief Propagation Simulator - PropFlow

PropFlow is a Python toolkit for building factor graphs, running belief
propagation variants, comparing engine policies, and inspecting per-iteration
snapshots. The public package is `propflow`; the core runtime lives under
`src/propflow`.

Full documentation is published at
[ormullerhahitti.github.io/Belief-Propagation-Simulator](https://ormullerhahitti.github.io/Belief-Propagation-Simulator/index.html).

## Key Features

- **Belief propagation engines**: synchronous BP with Min-Sum, Max-Sum,
  Sum-Product, and Max-Product computators.
- **Policy engines**: damping, Q/R damping, factor splitting, mid-run splitting,
  cost reduction, diffusion, message pruning, tree-reweighted BP, and combined
  damping/TRW or damping/splitting variants.
- **Graph construction helpers**: `FGBuilder` builds cycle, random, lemniscate,
  and unary-augmented factor graphs.
- **Cost-table factories**: use `create_random_int_table`,
  `create_uniform_float_table`, `create_poisson_table`, `CTFactories`, or
  `get_ct_factory`.
- **Simulation runner**: `Simulator` runs multiple engine configurations across
  multiple graphs with multiprocessing fallbacks.
- **Snapshots and analysis**: every engine step captures an `EngineSnapshot`;
  use `SnapshotAnalyzer`, `AnalysisReport`, and `SnapshotVisualizer` for cost,
  assignment, message, and Jacobian-style diagnostics.

## Installation

Install the published package:

```bash
pip install propflow
```

For development from this repository:

```bash
git clone https://github.com/OrMullerHahitti/Belief-Propagation-Simulator.git
cd Belief-Propagation-Simulator
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The CLI currently exposes a version/help entry point:

```bash
bp-sim --version
```

## Quick Start

```python
from propflow import BPEngine, FGBuilder, MinSumComputator
from propflow.configs import CTFactories

graph = FGBuilder.build_random_graph(
    num_vars=50,
    domain_size=10,
    ct_factory=CTFactories.RANDOM_INT,
    ct_params={"low": 100, "high": 200},
    density=0.25,
    seed=42,
)

engine = BPEngine(
    factor_graph=graph,
    computator=MinSumComputator(),
)
engine.run(max_iter=100)

print("Final assignments:", engine.assignments)
print("Final global cost:", engine.calculate_global_cost())
print("Iterations captured:", engine.iteration_count)
```

## Manual Factor Graphs

Use `FGBuilder` first when possible. If you construct a graph manually, every
factor must appear once in `factor_li` and once as a key in `edges`; each edge
list is ordered and defines the axes of the factor cost table.

```python
import numpy as np

from propflow import BPEngine, FactorAgent, FactorGraph, VariableAgent

def prefer_equal(num_vars: int, domain_size: int, **_):
    table = np.ones((domain_size,) * num_vars)
    np.fill_diagonal(table, 0)
    return table

x1 = VariableAgent("x1", domain=3)
x2 = VariableAgent("x2", domain=3)
f12 = FactorAgent("f12", domain=3, ct_creation_func=prefer_equal)

graph = FactorGraph(variable_li=[x1, x2], factor_li=[f12], edges={f12: [x1, x2]})
engine = BPEngine(graph)
engine.run(max_iter=20)
print(engine.assignments, engine.calculate_global_cost())
```

## Engine Variants

```python
from propflow import DampingEngine, SplitEngine, TRWEngine

damped = DampingEngine(factor_graph=graph, damping_factor=0.9)
split = SplitEngine(factor_graph=graph, split_factor=0.5)
trw = TRWEngine(factor_graph=graph, tree_sample_count=250, tree_sampler_seed=42)
```

Available top-level engine exports include `BPEngine`, `DampingEngine`,
`RDampingEngine`, `QRDampingEngine`, `DiffusionEngine`, `SplitEngine`,
`MidRunSplitEngine`, `CostReductionOnceEngine`, `DampingCROnceEngine`,
`DampingSCFGEngine`, `TRWEngine`, `DampingTRWEngine`, and
`MessagePruningEngine`.

## Running Experiments

```python
from propflow import BPEngine, DampingEngine, FGBuilder, Simulator, SplitEngine
from propflow.configs import create_random_int_table

configs = {
    "baseline": {"class": BPEngine},
    "damped": {"class": DampingEngine, "damping_factor": 0.85},
    "split": {"class": SplitEngine, "split_factor": 0.6},
}

graphs = [
    FGBuilder.build_cycle_graph(
        num_vars=12,
        domain_size=3,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 30},
    )
    for _ in range(5)
]

simulator = Simulator(configs, seed=42)
results = simulator.run_simulations(graphs, max_iter=200)
simulator.plot_results(verbose=True)
```

## Snapshot Analysis

Snapshots are captured automatically during `run()` and `step()`.

```python
from propflow.snapshots import AnalysisReport, SnapshotAnalyzer, SnapshotVisualizer

snapshots = list(engine.snapshots)
latest = engine.latest_snapshot()
print(latest.step, latest.global_cost, latest.metadata)

visualizer = SnapshotVisualizer(snapshots)
_, cost_data = visualizer.plot_global_cost(show=False, return_data=True)

analyzer = SnapshotAnalyzer(snapshots)
report = AnalysisReport(analyzer)
summary = report.to_json(step_idx=len(snapshots) - 1)
print(summary["block_norms"])
```
