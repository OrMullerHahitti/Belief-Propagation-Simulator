# Belief Propagation Simulator — Architecture & Usage Tree

This guide maps the full codebase in a tree-like view and explains how the simulator, engines, factor graphs, policies, and utilities compose together. It focuses on practical usage (inputs/outputs), recommended imports, and extension points so you can ship a clean, easy-to-use package.

## High-Level Overview

- Public API: `propflow` exposes the main classes for most users.
  - `from propflow import FactorGraph, VariableAgent, FactorAgent, BPEngine, Simulator, FGBuilder, DampingEngine, SplitEngine, MessagePruningEngine, ...`
- Core flow: create a `FactorGraph` → choose/compose an `Engine` → `engine.run(max_iter=N)` → inspect `engine.history` and (optionally) `engine.latest_snapshot()`.
- Batch flow: create multiple graphs → define multiple engines → pass both into `Simulator` → run in parallel and plot costs.

## Directory Tree (annotated)

```
src/propflow
├── __init__.py                   # Public API exports
├── _version.py                   # Package version
├── cli.py                        # Console entry-point: `bp-sim`
├── simulator.py                  # Batch runner over {engines × graphs}
├── bp/                           # BP algorithms layer
│   ├── __init__.py
│   ├── factor_graph.py           # Bipartite factor graph (variables ↔ factors)
│   ├── engine_base.py            # Base BPEngine (fixed-schedule BP loop)
│   ├── engines_realizations.py   # Engine variants (damping, splitting, pruning, ...)
│   ├── computators.py            # Min-sum / Max-sum / Sum-product / Max-product
│   └── engine_components.py      # Step, Cycle, History (BCT-ready formats)
├── core/                         # Agents and messaging primitives
│   ├── __init__.py
│   ├── agents.py                 # VariableAgent, FactorAgent
│   ├── components.py             # Message, MailHandler (inbox/outbox), CostTable alias
│   └── dcop_base.py              # Computator, Agent ABCs
├── configs/                      # Defaults, registries, logging
│   ├── __init__.py
│   ├── global_config_mapping.py  # ENGINE_DEFAULTS, SIMULATOR_DEFAULTS, CT_FACTORY registry
│   └── loggers.py                # Color/file loggers, log dirs
├── policies/                     # Optional runtime policies (BP modifiers)
│   ├── __init__.py               # Re-exports common policies
│   ├── damping.py                # `damp` (+ TD cycle-based variant)
│   ├── splitting.py              # Split factor tables (p*C,(1-p)*C)
│   ├── cost_reduction.py         # Single-pass factor cost discounting
│   ├── message_pruning.py        # Threshold-based pruning policy (accept/prune)
│   ├── normalize_cost.py         # Normalization helpers + alt pruning impl
│   └── convergance.py            # ConvergenceConfig, ConvergenceMonitor
├── snapshots/                    # Per-step capture + Jacobian/cycle analysis
│   ├── __init__.py
│   ├── types.py                  # SnapshotsConfig, SnapshotData/Jacobians
│   ├── manager.py                # SnapshotManager (A,P,B, cycles, norms, winners)
│   └── builder.py                # Extract Q/R, neighborhoods, domains from step
├── search/                       # Local search engines (DSA, MGM, K-Opt MGM)
│   ├── __init__.py
│   ├── search_engine.py          # Engines adapting the BPEngine lifecycle
│   ├── search_computator.py      # DSA, MGM, KOptMGM computators
│   └── search_agents.py          # Variable agent extensions for search
├── utils/                        # Builders, tools, IO helpers
│   ├── __init__.py
│   ├── fg_utils.py               # FGBuilder, helpers, pickle safety
│   ├── examples.py               # Easy FG creation without pickles
│   ├── create/
│   │   ├── create_factor_graph_config.py
│   │   └── create_factor_graphs_from_config.py
│   └── tools/ (perf, jacobian viz, figures, etc.)
└── ... egg-info/ (packaging)

src/analyzer                      # Optional deep analysis toolkit
├── __init__.py                   # Analysis decorator & helpers exports
├── snapshot.py, matrices.py, ... # Cycle analysis, invariants, margins, winners
```

## Data Flow (one run)

1) Build or load a `FactorGraph` with `VariableAgent` and `FactorAgent` nodes and cost tables.
2) Create a BP `Engine` (e.g., `DampingEngine`) with a `Computator` (default: min-sum).
3) Call `engine.run(max_iter=N)`; the engine executes synchronized phases per step:
   - Variables compute Q messages → send → clear → prepare
   - Factors compute R messages → send → clear → prepare
   - Compute global cost, track history; optional normalization & convergence checks
   - Optional: `SnapshotManager` captures Q/R, Jacobians (A,P,B), cycles, winners
4) Inspect outputs: `engine.history` (costs, beliefs/assignments per cycle), `engine.get_beliefs()`, `engine.assignments`, `engine.latest_snapshot()`.

## Public API (recommended imports)

- Graph & Agents:
  - `FactorGraph`, `VariableAgent`, `FactorAgent`
- Engines (BP):
  - `BPEngine` (base), `DampingEngine`, `SplitEngine`, `CostReductionOnceEngine`, `DampingSCFGEngine`, `DampingCROnceEngine`, `DiscountEngine`, `MessagePruningEngine`
- Batch runner:
  - `Simulator`
- Builders:
  - `FGBuilder` (`build_random_graph`, `build_cycle_graph`), `CTFactory` registry
- Search (optional):
  - `search` engines (`DSAEngine`, `MGMEngine`, `KOptMGMEngine`) and computators

## Factor Graphs

- Structure: `FactorGraph(variables, factors, edges)` where `edges: {FactorAgent: [VariableAgent, ...]}`.
- Nodes:
  - `VariableAgent(name, domain)`: holds domain size, computes Q, tracks belief and current assignment.
  - `FactorAgent(name, domain, ct_creation_func, param|cost_table)`: holds an n-D cost table; computes R.
- Internals:
  - `G`: NetworkX bipartite graph (variables bipartite=0, factors bipartite=1)
  - `connection_number`: per-factor map of variable name → dimension index (for cost-table broadcasting/order)
  - `global_cost`: sum of costs using current variable assignments across original factors
  - `set_computator(...)`: assigns computator to all nodes
  - Pickle-safe (`__getstate__/__setstate__`) for multiprocessing & persistence

## Engines (BP)

- Base: `BPEngine(factor_graph, computator=MinSumComputator(), init_normalization=dummy, name="BPEngine", convergence_config=None, monitor_performance=None, normalize_messages=None, anytime=None, use_bct_history=None, snapshots_config=None)`
  - Inputs:
    - Required: `factor_graph`
    - Optional: `computator` (min-sum/max-sum/sum-product/max-product), convergence & performance configs, normalization flags, snapshots
  - Step Phases (synchronized): variable Q compute → send; factor R compute → send; cost update → history; optional normalization and convergence; optional snapshot
  - Outputs: `history` (Step/Cycle, costs, beliefs, assignments), `assignments`, `get_beliefs()`, `latest_snapshot()`
  - Hooks for variants: `post_init`, `pre/post_factor_compute`, `post_var_compute`, `post_two_cycles`, `post_factor_cycle`

- Engine Variants (what they add)
  - `Engine`: alias to base engine
  - `DampingEngine(damping_factor=0.9)`
    - Applies `damp(var, x)` after variable compute; stores last iteration per variable
  - `SplitEngine(split_factor=0.6)`
    - `post_init` splits every factor into two clones with tables `p*C` and `(1-p)*C`; updates graph and factor list
  - `CostReductionOnceEngine(reduction_factor=0.5)`
    - `post_init` single-pass cost reduction across all factors; `post_factor_compute` multiplies factor outbox attentively
  - `DampingSCFGEngine(damping_factor=0.9, split_factor=0.6)`
    - Composition of damping + splitting
  - `DampingCROnceEngine(damping_factor=0.9, reduction_factor=0.5)`
    - Composition of damping + single-pass cost reduction
  - `DiscountEngine()`
    - `post_factor_cycle` applies discounted updates to factor side
  - `MessagePruningEngine(prune_threshold=1e-4, min_iterations=5, adaptive_threshold=True)`
    - Initializes a pruning policy to reduce near-duplicate messages (accept/prune by L2-diff); integrate by setting policy on agent mailers if needed

## Computators

- `BPComputator` (base): vectorized message operations (dispatch tables for `reduce`/`combine`).
  - `compute_Q(messages)` → list[Message] from variable to factor
  - `compute_R(cost_table, incoming_messages)` → list[Message] from factor to variable
  - `compute_belief(messages, domain)` → np.ndarray
  - `get_assignment(belief)` → arg-reduce index (e.g., argmin for min-sum)
- Variants:
  - `MinSumComputator` (default): reduce=min, combine=add
  - `MaxSumComputator`: reduce=max, combine=add
  - `MaxProductComputator`: reduce=max, combine=mul
  - `SumProductComputator`: reduce=sum, combine=mul

## Policies (optional modifiers)

- `damping.damp(variable, x=0.9)` and cycle-based `TD(vars, x, diameter)`
- `splitting.split_all_factors(fg, p=0.5)`
- `cost_reduction.cost_reduction_all_factors_once(fg, k)` / `discount_attentive`
- `message_pruning.MessagePruningPolicy(prune_threshold, min_iterations, adaptive)` with `MailHandler.set_pruning_policy(policy)`
- `normalize_cost.normalize_inbox(variables)` after cycles and other normalization helpers
- Convergence: `ConvergenceMonitor(ConvergenceConfig)` used by `BPEngine` during cycles

## History, Step/Cycle, and Snapshots

- `engine_components.History` tracks:
  - `costs`: per-step scalar cost; `beliefs`/`assignments`: per-graph-diameter cycles
  - Optional BCT mode: step-level `step_beliefs`, `step_messages`, `step_costs` for detailed analysis
  - `save_results(filename)` / `to_json(path)`
- `Step`: per-step captured messages
  - `q_messages`: variable→factor; `r_messages`: factor→variable (used by snapshots)
- `snapshots.SnapshotManager(SnapshotsConfig)`
  - Builds `SnapshotData` with Q/R, neighborhoods, domains; computes Jacobians `(A,P,B)` and optional cycle metrics and block norms; supports `save_step(dir, save=True)` for explicit persistence

## Simulator (batch)

- `Simulator(engine_configs: dict, log_level=None)`
  - `engine_configs` shape:
    ```python
    engine_configs = {
      "DampingEngine": {"class": DampingEngine, "damping_factor": 0.9},
      "Split": {"class": SplitEngine, "split_factor": 0.6},
      # ...any engine class with its kwargs
    }
    ```
  - `run_simulations(graphs: list[FactorGraph], max_iter=SIMULATOR_DEFAULTS['default_max_iter'])` → dict[str, list[list[float]]]
    - Returns: `engine_name -> [cost_series_per_graph]`
    - Uses multiprocessing with fallbacks; logs per-process; timeouts configurable
  - `plot_results(max_iter=None, verbose=False)` → inline matplotlib plot (avg ± std if verbose)
  - `set_log_level(level_str)` to switch logging during a run

## Builders, Configs, and Registries

- `FGBuilder`
  - `build_random_graph(num_vars, domain_size, ct_factory, ct_params, density)` → `FactorGraph`
  - `build_cycle_graph(num_vars, domain_size, ct_factory, ct_params, density=1.0)` → `FactorGraph`
- Config infrastructure (`configs/global_config_mapping.py`)
  - Defaults: `ENGINE_DEFAULTS`, `POLICY_DEFAULTS`, `CONVERGENCE_DEFAULTS`, `SIMULATOR_DEFAULTS`
  - CT factories: register via `@ct_factory("name")` and then use `CTFactory.name` or `get_ct_factory("name")`
  - Graph types registry: `GRAPH_TYPES` → dotted path of builder functions
  - Logging: `LOGGING_CONFIG`, `Logger`, level names: `HIGH`, `INFORMATIVE`, `VERBOSE`, `MILD`, `MINIMAL`
- Declarative graph config (optional): `utils/create/*`
  - `ConfigCreator.create_graph_config(...)` → pickled `GraphConfig`
  - `FactorGraphBuilder.build_and_return(cfg_path)` or `build_and_save(cfg_path)`

## Search Engines (optional)

- Engines adapt BP lifecycle to local search: `SearchEngine`, `DSAEngine`, `MGMEngine`, `KOptMGMEngine`
- Computators: `DSAComputator`, `MGMComputator`, `KOptMGMComputator`
- Agents: `SearchVariableAgent` extends `VariableAgent` with neighbor-value access, pending decisions, and direct assignment property
- I/O:
  - Inputs: same `FactorGraph` + a search computator
  - Per-step output: best assignment/cost tracking via `history.costs` and `best_assignment` fields (see `SearchEngine`)

## CLI & Examples

- CLI: `bp-sim --version` prints version (more commands can be added later)
- Examples:
  - Quick start (`examples/quick_start.py`): minimal 2-variable graph and `DampingEngine`
  - Simulator run (`examples/run_simulator.py`): builds many random graphs and compares engines with plots

## Typical Usage Patterns

1) Minimal run
```python
from propflow import FactorGraph, VariableAgent, FactorAgent, DampingEngine
import numpy as np

v1, v2 = VariableAgent("v1", 2), VariableAgent("v2", 2)
def table(num_vars=None, domain_size=None, **_):
    return np.array([[0, 1], [1, 0]])
f = FactorAgent("f", 2, ct_creation_func=table)
fg = FactorGraph([v1, v2], [f], edges={f: [v1, v2]})
engine = DampingEngine(factor_graph=fg, damping_factor=0.9)
engine.run(max_iter=20)
print(engine.history.costs[-1], engine.assignments)
```

2) Batch compare engines on random graphs
```python
from propflow import Simulator, FGBuilder, DampingEngine, SplitEngine
from propflow.configs import CTFactory

graphs = [
  FGBuilder.build_random_graph(
    num_vars=50, domain_size=10,
    ct_factory=CTFactory.random_int.fn,
    ct_params={"low": 100, "high": 200}, density=0.25,
  ) for _ in range(10)
]

engines = {
  "Damping": {"class": DampingEngine, "damping_factor": 0.9},
  "Split": {"class": SplitEngine, "split_factor": 0.6},
}

sim = Simulator(engines, log_level="INFORMATIVE")
results = sim.run_simulations(graphs, max_iter=5000)
sim.plot_results(verbose=True)
```

3) Enable snapshots (Jacobian/cycle analysis)
```python
from propflow.snapshots import SnapshotsConfig
from propflow import DampingEngine

snap_cfg = SnapshotsConfig(
  compute_jacobians=True, compute_cycles=True, compute_block_norms=True,
  retain_last=25, save_each_step=False,
)
engine = DampingEngine(factor_graph=fg, damping_factor=0.9, snapshots_config=snap_cfg)
engine.run(max_iter=50)
snap = engine.latest_snapshot()
print(snap.jacobians.block_norms if snap and snap.jacobians else None)
```

## Inputs/Outputs Summary (engines)

- Common engine inputs
  - `factor_graph`: `FactorGraph`
  - `computator`: one of `MinSumComputator` (default), `MaxSumComputator`, `SumProductComputator`, `MaxProductComputator`
  - Optional: `ConvergenceConfig`, performance monitoring, normalization flags, `SnapshotsConfig`
- Common outputs
  - `history.costs`: list[float]
  - `history.beliefs[cycle]`: dict[var -> np.ndarray]
  - `history.assignments[cycle]`: dict[var -> int]
  - `get_beliefs()`: dict[var -> np.ndarray]
  - `assignments`: dict[var -> int]
  - `latest_snapshot()`: `SnapshotRecord | None`

## Extension Points

- Add a new engine
  - Subclass `BPEngine`; override hooks (`post_init`, `post_var_compute`, `post_factor_compute`, `post_factor_cycle`) as needed
  - Add to your app via `engine_configs = {"MyEngine": {"class": MyEngine, ...}}`
- Add a new computator
  - Subclass `BPComputator` and define reduce/combine ops; plug into an engine via `computator=YourComputator()`
- Add a new policy
  - Implement a callable or `Policy` subclass; apply from engine hooks or set on `MailHandler` (e.g., `set_pruning_policy`)
- Add a new cost-table factory
  - Decorate a function with `@ct_factory("my_factory")` in `global_config_mapping.py`
- Add a new graph topology builder
  - Implement a builder returning `(variables, factors, edges)`; register in `GRAPH_TYPES`

## Packaging & Imports (clean user surface)

- Preferred imports for end users (stable):
  - `from propflow import FactorGraph, VariableAgent, FactorAgent, Simulator, FGBuilder`
  - `from propflow import DampingEngine, SplitEngine, MessagePruningEngine` (engine variants)
  - `from propflow.configs import CTFactory` (cost-table factories)
- CLI: `bp-sim --version`

## Notes & Conventions

- Python 3.10+, type hints, Black formatting, `flake8` max line length 120.
- Avoid circular imports; use `propflow.__init__` re-exports for end-user code.
- Deterministic seeds for experiments; keep logs/artifacts under `configs/logs` and results in dedicated folders.

---

If you want this file split into shorter topic pages (Engines, Factor Graphs, Simulator, Snapshots/Analyzer) or need a deeper table of each class’ arguments/returns, tell me and I’ll generate those as well.
