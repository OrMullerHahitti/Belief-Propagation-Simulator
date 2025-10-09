# Conceptual Overview

## 1. What PropFlow Solves
PropFlow is a belief propagation (BP) experimentation platform. It builds factor graphs, runs synchronous BP with configurable policies, and records the evolution of messages, assignments, and costs. The toolkit targets researchers and engineers who need a reproducible environment for comparing BP variants such as pure Min-Sum, damped updates, splitting, or custom convergence strategies.

## 2. Core Runtime Concepts

| Concept | Description |
| --- | --- |
| **Factor Graph** | Bipartite graph of `VariableAgent` and `FactorAgent` nodes. Variables carry domains; factors encode local cost tables. Defined in `src/propflow/bp/factor_graph.py` and constructed via helpers in `src/propflow/utils/fg_utils.py`. |
| **Message** | Directed data structure (`src/propflow/core/components.py`) representing variable→factor (Q) or factor→variable (R) updates. Messages contain NumPy vectors aligned to variable domains. |
| **Engine** | Implementation of the BP update loop (`src/propflow/bp/engine_base.py` and concrete classes in `src/propflow/bp/engines.py`). Engines orchestrate the six-phase synchronous schedule: variable compute, send, factor compute, send, pruning, and convergence evaluation. |
| **Policy** | Optional behaviours applied during message computation: damping, splitting, cost reduction, pruning. Policies live under `src/propflow/policies`. |
| **Simulator** | High-level runner (`src/propflow/simulator.py`) that executes batches of engine configurations over a set of factor graphs, collects cost trajectories, and handles multiprocessing. |
| **History & Snapshots** | Built-in engine history stores assignments, costs, and optional BCT artefacts. External snapshot tooling (`src/analyzer/snapshot_recorder.py`) captures per-iteration state for analysis or visualisation. |
| **Analyzer Utilities** | Lightweight visual tools such as the `SnapshotVisualizer` (`src/analyzer/snapshot_visualizer.py`) for interpreting minimisers over time. |

## 3. Creation Pipeline (Top-Down)

PropFlow’s runtime mirrors the documentation flow:

1. **Agents** — `VariableAgent` and `FactorAgent` live in `src/propflow/core/agents.py`.
   They encapsulate message mailboxes and call into computators.
2. **Factor graphs** — `FactorGraph` (`src/propflow/bp/factor_graph.py`) stitches agents
   together, assigns dimension indices, and triggers cost-table creation. Use
   `FGBuilder` helpers (`src/propflow/utils/fg_utils.py`) wherever possible.
3. **Engines** — `BPEngine` and variants (`src/propflow/bp/engine_base.py`,
   `src/propflow/bp/engines.py`) orchestrate the synchronous message schedule,
   apply policies, and record history.
4. **Simulator** — `Simulator` (`src/propflow/simulator.py`) executes batches of
   engine configurations over one or many graphs, typically for benchmarks or
   parameter sweeps.
5. **Analyzer** — Modules under `src/analyzer/` capture snapshots and visualise
   results, letting you inspect or report on specific runs.

The quickest path for an end user is thus:
``FGBuilder → BPEngine (or variant) → Simulator (optional) → Analyzer tooling``.
Drop down to manual factor graph construction only when you need a topology that
`FGBuilder` does not support, taking care to supply ordered factor–variable edge
lists so cost-table dimensions remain aligned.

## 4. Directory Layout

```
src/
  propflow/
    bp/            # Engines, engine components, factor graph primitives
    core/          # Shared agent/message abstractions
    policies/      # Damping, splitting, convergence controls
    configs/       # Global defaults, logging, registries
    utils/         # Graph builders, tooling (FGBuilder, etc.)
    simulator.py   # High-level simulation orchestrator
    cli.py         # Placeholder CLI entrypoint
  analyzer/
    snapshot_recorder.py     # External recorder for per-step data capture
    snapshot_visualizer.py   # Visualiser for argmin trajectories
    README.md                # Snapshot tooling guide

examples/          # Demonstrations (e.g. min-sum walkthrough)
tests/             # Pytest suite covering engines, policies, utilities
docs/handbook/     # Deployment & operations handbook (this document)
configs/           # Logs and generated artefacts (excluded from version control)
```

## 5. Data Flow Through an Engine Step

1. **Variable Phase**: Variable agents read inbox messages (`Mailer` component), compute outgoing Q-messages via the assigned computator (default Min-Sum), and stage them.
2. **Variable Send & Reset**: Messages are dispatched; inboxes are cleared in preparation for the next iteration.
3. **Factor Phase**: Factor agents assemble incoming Q-messages and cost tables to compute R-messages back to neighbours.
4. **Factor Send & Reset**: R-messages are sent, and mailboxes are cleared.
5. **Bookkeeping**: Engine updates the global cost, logs history, and (optionally) emits snapshot records via `SnapshotManager` or external recorders.
6. **Convergence Checks**: Using `ConvergenceMonitor` and the configured thresholds, engines decide whether to halt.

This loop repeats for each iteration until convergence or the configured maximum is reached.

## 6. Extensibility Points
- **Custom Engines**: Subclass `BPEngine` and override hooks (`post_var_compute`, `pre_factor_compute`, etc.). Register the class in simulation configs for reuse.
- **Policies**: Implement new policies under `src/propflow/policies` and wire them into engines or simulator configs.
- **Cost Table Factories**: Extend `CTFactory` registries in `src/propflow/configs/global_config_mapping.py` to generate domain-specific cost tables.
- **Analysis**: Use `EngineSnapshotRecorder` to capture structured outputs suitable for dashboards, machine learning pipelines, or audits.

## 7. Operational Roles
- **Simulation Engineer**: Configures factor graphs, chooses engine variants, evaluates convergence metrics.
- **Algorithm Researcher**: Modifies computators, policies, or engines to test new BP strategies.
- **Data Analyst**: Uses snapshot recordings and visualisers to interpret message dynamics and assignment stability.
- **DevOps / Platform Team**: Deploys the simulator into production or research clusters, automates pipelines, and monitors resource consumption.

Understanding these components provides the foundation for the remaining sections, which focus on getting the project running, deploying it reliably, and maintaining it over time.
