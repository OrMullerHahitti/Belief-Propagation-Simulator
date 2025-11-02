# Running Simulations & Tools

This chapter assumes you already know how to assemble graphs and engines (see
the :doc:`../user_guide`). Here we focus on operationalising the flow:
``FGBuilder → engine → Simulator → snapshots``.

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

Snapshots are always available via :attr:`engine.snapshots`. Persist them by
serialising to JSON and leverage the visualiser/analyzer utilities inside
``propflow.snapshots``.

```python
import json
from pathlib import Path

from propflow.snapshots import SnapshotAnalyzer, AnalysisReport
from propflow.snapshots import SnapshotVisualizer

engine = BPEngine(factor_graph=fg, use_bct_history=True)
engine.run(max_iter=100)

snapshots = list(engine.snapshots)

Path("results").mkdir(exist_ok=True)
with open("results/run_001_snapshots.json", "w", encoding="utf-8") as handle:
    json.dump([
        {
            "step": snap.step,
            "assignments": snap.assignments,
            "global_cost": snap.global_cost,
        }
        for snap in snapshots
    ], handle, indent=2)

viz = SnapshotVisualizer(snapshots)
viz.plot_argmin_per_variable(show=True)

analyzer = SnapshotAnalyzer(snapshots)
report = AnalysisReport(analyzer)
summary = report.to_json(step_idx=len(snapshots) - 1)
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
