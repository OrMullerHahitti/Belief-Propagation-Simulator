# System Setup & Dependencies

## 1. Supported Platforms
- **Operating systems**: macOS, Linux, Windows 10/11 (WSL recommended on Windows).
- **Python**: 3.10 – 3.13 (project currently tested with 3.13.7 via Homebrew and `uv`).
- **Hardware**: CPU-only; BP workloads scale with core count. Memory requirements depend on factor graph size (expect several hundred MB for dense graphs with 50+ variables).

## 2. Core Dependencies
Defined in `pyproject.toml` under `[project.dependencies]`:

| Package | Purpose |
| --- | --- |
| `networkx` | Factor graph manipulation and diameter calculations. |
| `numpy`, `scipy` | Numerical operations for message updates and cost computation. |
| `matplotlib` | Plotting cost trajectories and visual analytics. |
| `dash`, `dash_cytoscape`, `seaborn`, `pandas` | Optional dashboards and analysis tooling. |
| `psutil`, `colorlog` | Monitoring and structured logging. |

Development extras (`.[dev]`) include `pytest`, `pytest-cov`, `black`, `mypy`, `pre-commit`, `jupyter`, etc.

## 3. Environment Bootstrapping

### Option A – Recommended: `uv`
```bash
# From repository root
uv sync
uv run python -m pip check  # optional verification
```
`uv` creates a `.venv` folder by default and resolves dependency locks via `uv.lock`.

### Option B – Standard virtualenv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e .[dev]
pre-commit install
```

### Option C – Conda
```bash
conda create -n propflow python=3.11
conda activate propflow
pip install -e .[dev]
```

## 4. Verifying Installation
1. Activate the environment.
2. Run `python -c "import propflow; print(propflow.__version__)"` to confirm package import.
3. Execute `python examples/minsum_basic.py` to ensure engines and graph builders are functional.
4. Optionally run `pytest -q` to validate the suite (expect some longer runtimes for heavy graph tests).

## 5. External Services & Data
No external services are required. Factor graphs are generated on the fly or loaded from pickled artefacts. For reproducibility, ensure deterministic seeds are configured in configs or snapshot recorders.

## 6. File System Expectations
- Write access to `configs/logs/` (default log directory) and optional `results/` directories for export.
- `docs/handbook/` houses this deployment guide; no runtime writes expected.
- Ensure large outputs (plots, CSVs) are excluded from version control or placed under a dedicated `results/` directory.

With an environment ready, proceed to [Running Simulations & Tools](03_running_simulations.md) to launch jobs and capture artefacts.
