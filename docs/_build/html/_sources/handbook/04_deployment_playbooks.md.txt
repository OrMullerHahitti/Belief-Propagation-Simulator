# Deployment Playbooks

This section outlines practical strategies for running PropFlow beyond local experiments. Choose the pathway that matches your infrastructure.

## 1. Local Workstation or Lab Server

### Steps
1. **Clone & Install** using `uv sync` or `pip install -e .[dev]`.
2. **Prepare Experiment Scripts** (e.g., adapt `main.py` or write your own driver under `scripts/`).
3. **Run Simulations** using `uv run python your_script.py` or `python -m ...` once the environment is activated.
4. **Persist Artefacts** to a dedicated directory (e.g., `results/<experiment_id>/`).
5. **Automate** with cron or task schedulers if you need recurring jobs.

### Tips
- Use `EngineSnapshotRecorder` for reproducible records.
- Capture environment metadata (`python --version`, `pip freeze`) alongside results.

## 2. Containerised Deployment (Docker)

### Example `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system packages (optional, e.g., build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY examples ./examples

RUN pip install --upgrade pip && pip install uv
RUN uv sync --frozen

ENV PYTHONPATH="/app/src"

CMD ["uv", "run", "python", "main.py"]
```

### Build & Run
```bash
docker build -t propflow:latest .
docker run --rm -v $(pwd)/results:/app/results propflow:latest
```

### Notes
- Mount a host volume (`results/`) to retain logs and plots.
- Use environment variables or config files for hyperparameters (e.g., number of graphs, damping factors).

## 3. HPC / Batch Scheduling

1. **Create a workload script** that activates the environment and runs your simulation script.
2. **Set deterministic seeds** for comparability.
3. **Leverage multiple nodes** by distributing factor graphs and aggregating results post-run.
4. **Monitor resource usage** (CPU, RAM) via scheduler dashboards or `psutil` within the script.

Example Slurm script:
```bash
#!/bin/bash
#SBATCH --job-name=propflow
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

module load python/3.11
source /path/to/venv/bin/activate

cd /path/to/Belief-Propagation-Simulator
python scripts/run_batch.py --max-iter 5000 --graphs 50
```

## 4. Web Service / API Integration

While PropFlow is primarily batch-oriented, you can wrap engines inside a web service for interactive workloads:

1. **Expose a REST API** (e.g., FastAPI) that receives factor graph definitions or seeds.
2. **Instantiate engines on demand**, run limited iterations, return convergence metrics or snapshots.
3. **Persist heavy artefacts** asynchronously to avoid blocking HTTP responses.
4. **Containerise** the service and deploy via Kubernetes / ECS with resource quotas.

## 5. Packaging & Distribution

- **Wheel build**: `uv build` or `python -m build` generates `.whl` and `.tar.gz` under `dist/`.
- **Private PyPI**: Upload with `twine upload --repository-url ... dist/*`.
- **Versioning**: Update `src/propflow/_version.py` and `pyproject.toml` before releasing.

## 6. Configuration Management

- Central defaults live in `src/propflow/configs/global_config_mapping.py`.
- To override parameters (e.g., `max_iterations`, `damping_factor`), supply dict overrides when constructing engines or `Simulator` configs.
- Keep custom configurations in `configs/` or `examples/` with descriptive filenames; avoid modifying defaults unless you intend to change project-wide behaviour.

## 7. Logging & Monitoring

- `Simulator` uses the `Logger` wrapper (`src/propflow/configs/loggers.py`) that supports colourised console output and file logging.
- When deploying, redirect logs to standard output (for container logs) or a centralised logging solution.
- Consider integrating with monitoring tools by extending the simulator to emit metrics (e.g., Prometheus format) per iteration.

With deployment blueprints in place, read [Development Workflow](05_development_workflow.md) for contributing guidelines and quality gates.
