# Running Simulations & Tools

This chapter assumes you already know how to assemble graphs and engines (see
the :doc:`../user_guide`). Here we focus on operationalising the flow:
``FGBuilder → engine → Simulator → analyzer``.

## 1. Quick Smoke Test (`main.py`)
The repository root includes `main.py`, which generates random factor graphs and runs multiple engine variants.

```bash
uv run python main.py
```

What it does:
- Builds 10 random factor graphs (`FGBuilder.build_random_graph`) with 50 variables, domain size 10, density 0.25.
- Configures `BPEngine`, `DampingSCFGEngine`, and `SplitEngine` variants using defaults from `ENGINE_DEFAULTS` / `POLICY_DEFAULTS`.
- Runs the `Simulator` across each engine/graph combination, timing total runtimes.
- Plots aggregated cost trajectories if results are available.

> **Troubleshooting**: If you see `ModuleNotFoundError: No module named 'propflow'`, ensure the `src/` directory is on `PYTHONPATH` or run via `uv run`/`pip install -e .`.

## 2. Using the `Simulator` API
`Simulator` (`src/propflow/simulator.py`) accepts a dict of engine configurations and a list of factor graphs.

```python
from propflow.simulator import Simulator
from propflow.utils import FGBuilder
from propflow.bp.engines import BPEngine
from propflow.configs import CTFactory

engines = {"baseline": {"class": BPEngine}}
fg = FGBuilder.build_random_graph(
    num_vars=20,
    domain_size=5,
    ct_factory=CTFactory.random_int.fn,
    ct_params={"low": 0, "high": 25},
    density=0.3,
)

sim = Simulator(engines)
results = sim.run_simulations([fg], max_iter=1000)
sim.plot_results()
```

Key kwargs:
- `max_iter`: Defaults to `SIMULATOR_DEFAULTS["default_max_iter"]` (5000).
- `log_level`: Accepts symbolic levels (`"INFORMATIVE"`, `"HIGH"`, etc.).
- Internally, `Simulator` parallelises runs via `multiprocessing.Pool` with graceful fallbacks to sequential execution when necessary.

## 3. Command-Line Interface (`bp-sim`)
The CLI is defined in `src/propflow/cli.py`. Currently it exposes a version check and placeholder messaging. Extend this module if you wish to support command-line configuration of simulations.

```bash
uv run bp-sim --version
```

## 4. Snapshot Capture & Analysis

### 4.1 EngineSnapshotRecorder
`src/analyzer/snapshot_recorder.py` provides an external recorder that keeps engine internals untouched. Snapshots are already captured in-memory; call `recorder.save(engine)` to persist them when needed.

```python
from analyzer.snapshot_recorder import EngineSnapshotRecorder

recorder = EngineSnapshotRecorder(engine)
recorder.record_run(max_steps=100, break_on_convergence=True)
recorder.to_json("results/run_001_snapshots.json")
```

Captured fields per iteration include:
- Message flows (sender, recipient, values, argmin index, neutrality flag)
- Variable assignments and global cost
- Counts of neutral messages / step neutrality

### 4.2 SnapshotVisualizer
`src/analyzer/snapshot_visualizer.py` helps interpret argmin trajectories per variable.

```python
from analyzer.snapshot_visualizer import SnapshotVisualizer

viz = SnapshotVisualizer.from_json("results/run_001_snapshots.json")
series = viz.argmin_series(vars_filter=["X1", "X5"])
print(series)
# {'X1': [0, 0, 1, ...], 'X5': [2, 1, 1, ...]}

viz.plot_argmin_per_variable(vars_filter=["X1", "X5"], savepath="plots/X15.png")
```

CLI usage:
```bash
uv run python src/analyzer/snapshot_visualizer.py results/run_001_snapshots.json --vars X1 X5 --save plots/X15.png
```

## 5. Examples Directory
- `examples/minsum_basic.py`: step-by-step min-sum demonstration.
- `examples/analyzer_ring_demo.py`: runs a 4-variable ring, records snapshots, and generates per-variable and combined argmin plots via the visualiser.
- Use these scripts to validate installation or to template new experiments.

Run any example with:
```bash
uv run python examples/minsum_basic.py
```

## 6. Testing & Validation

### Unit Tests
```bash
uv run python -m pytest -q
```
Focus areas: BP engine behaviour, policies, utilities, and search components.

### Coverage
```bash
uv run python -m pytest --cov=src --cov-report=term-missing
```

### Static Analysis
```bash
uv run flake8
uv run mypy src
uv run black --check .
```

## 7. Logging & Artefacts
- Logs default to `configs/logs/`. Ensure this directory is writable when deploying.
- Engine history and simulator outputs can be persisted via `Simulator` or custom scripts.
- Use `results/` or dedicated experiment directories to store plots, JSON traces, or CSV exports.

With these tools you can execute and monitor simulations. Next, read [Deployment Playbooks](04_deployment_playbooks.md) to see how to package and run PropFlow in different environments.
