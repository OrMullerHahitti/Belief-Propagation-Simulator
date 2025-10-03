# Reference Tables

## 1. Modules & Key Entrypoints

| Module | Purpose | Notes |
| --- | --- | --- |
| `propflow.simulator.Simulator` | Batch orchestration of engines over factor graphs. | Handles multiprocessing, logging, plotting. |
| `propflow.bp.engine_base.BPEngine` | Base synchronous BP engine. | Provides step routine, history tracking, convergence checks. |
| `propflow.bp.engines` | Concrete engine variants (`BPEngine`, `DampingEngine`, `SplitEngine`, etc.). | Mix of damping, cost reduction, splitting strategies. |
| `propflow.utils.fg_utils.FGBuilder` | Helper for constructing factor graphs. | Supports random graph generation and deterministic setups. |
| `propflow.configs.global_config_mapping` | Centralised defaults and registries. | Contains `ENGINE_DEFAULTS`, `POLICY_DEFAULTS`, `SIMULATOR_DEFAULTS`, CT factories. |
| `analyzer.snapshot_recorder.EngineSnapshotRecorder` | External per-step recorder. | No engine modifications; outputs JSON-friendly data. |
| `analyzer.snapshot_visualizer.SnapshotVisualizer` | Plots argmin trajectories. | Accepts JSON or in-memory snapshot lists. |

## 2. Configuration Defaults (`global_config_mapping.py`)

| Section | Key | Default | Meaning |
| --- | --- | --- | --- |
| Engine | `max_iterations` | 2000 | Per-run iteration cap. |
| Engine | `normalize_messages` | `True` | Whether to min-normalise messages post-cycle. |
| Engine | `monitor_performance` | `False` | Enables `PerformanceMonitor`. |
| Engine | `anytime` | `False` | Enables anytime inference updates. |
| Engine | `use_bct_history` | `False` | Collects detailed BCT data. |
| Simulator | `default_max_iter` | 5000 | Default iteration cap for simulator runs. |
| Simulator | `default_log_level` | `"INFORMATIVE"` | Logger default verbosity. |
| Simulator | `timeout` | 3600 | Multiprocessing get timeout (seconds). |
| Policy | `damping_factor` | 0.9 | Default damping parameter. |
| Policy | `split_factor` | 0.5 | Weight for split engines. |

*Override defaults by passing overrides when constructing engines or updating the defaults dictionary.*

## 3. CLI & Scripts

| Command | Description |
| --- | --- |
| `uv run bp-sim --version` | Prints CLI version. |
| `uv run python main.py` | Runs bundled random-graph simulation demo. |
| `uv run python examples/minsum_basic.py` | Demonstrates Min-Sum on a small graph. |
| `uv run python src/analyzer/snapshot_visualizer.py <json>` | Plots argmin trajectories from snapshot JSON. |

## 4. Environment Variables
- `PYTHONPATH` should include `src/` if you are not using editable installs.
- `PROPLOW_LOG_LEVEL` (custom) can be set before running scripts to override simulator logging (extend `Simulator` to read it if desired).

## 5. File & Directory Conventions

| Path | Usage |
| --- | --- |
| `configs/logs/` | Default destination for simulator and engine logs. |
| `results/` | Suggested output folder for experiment artefacts (plots, JSON). |
| `notebooks/` | Exploratory analysis; keep results light-weight. |
| `docs/handbook/` | Deployment & operations documentation. |

## 6. Glossary
- **Belief**: Aggregated message vector representing variable costs/probabilities.
- **Argmin**: Index of minimum value within a belief vector; indicates preferred assignment under Min-Sum semantics.
- **Neutral Message**: Message whose minimum value is not unique; indicates ties or flat preferences.
- **Snapshot**: Structured record of a BP iteration capturing messages, assignments, and cost metrics.
- **CT Factory**: Function that generates cost tables (e.g., random integers, Poisson) for factor nodes.

Use this reference as a quick lookup when working with the codebase or extending the simulator.
