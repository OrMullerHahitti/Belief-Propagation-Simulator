# Troubleshooting & FAQ

## 1. Installation Issues

### `ModuleNotFoundError: No module named 'propflow'`
- Ensure you installed the package in editable mode (`pip install -e .`) or run scripts with `uv run` which adjusts `PYTHONPATH` automatically.
- Confirm you are running commands from the repository root.

### `python: command not found`
- Use `python3` or the interpreter inside your virtual environment (`.venv/bin/python`).
- On systems where `python` is not symlinked, update scripts to call `python3` explicitly.

### Dependency compilation failures
- Install system build tools (`build-essential` on Debian/Ubuntu, `xcode-select --install` on macOS) if SciPy or other compiled packages require them.

## 2. Runtime Errors

### `RuntimeError: Invalid default configuration`
- Ensure that `EngineDefaults()` contains all keys expected by `validate_engine_config`. Reinstall or merge the latest package if defaults and validators drift.

### `ModuleNotFoundError: No module named 'networkx'`
- Development extras may not be installed. Run `pip install -e .[dev]` or `uv sync` to pull runtime dependencies.

### Multiprocessing permissions (macOS sandbox errors)
- When running inside restricted sandboxes (CI, macOS seatbelt), multiprocessing may be denied. Set `Simulator` to use sequential fallback or run scripts outside the sandboxed environment.

## 3. Performance & Resource Usage
- **Slow runs**: Reduce graph size (`num_vars`, `density`) or iteration limits (`max_iter`).
- **High memory**: Large cost tables can be heavy; consider sparse representations or smaller domains.
- **CPU saturation**: Reduce the graph batch size or pass smaller workloads to `Simulator.run_simulations()`. `SimulatorDefaults().cpu_count_multiplier` documents the intended throttle, but the current runner uses `multiprocessing.cpu_count()` directly.

## 4. Visualisation Problems
- **Matplotlib not installed**: Ensure `matplotlib` is part of the environment (it is in core dependencies).
- **Plot window not showing**: Pass `show=False` and `savepath="..."` to `SnapshotVisualizer` plotting methods when running headless.
- **Mixed domain lengths**: `SnapshotVisualizer` raises `ValueError` if a variable receives messages of differing lengths; inspect your factor graph construction.

## 5. Snapshot Recorder Questions
- **Why are some argmins `None`?** There were no R-messages and no assignment fallback for that variable. Ensure the engine produced assignments or adjust your recorder to infer defaults.
- **Large snapshot size**: For long runs, consider slicing the snapshot list or sampling every `k` iterations.

## 6. Deployment Questions
- **How do I keep logs outside the container?** Mount a host volume when running Docker (`-v $(pwd)/logs:/app/configs/logs`).
- **Can I run on GPU?** Current implementation is CPU-based; GPU acceleration is not supported.
- **How do I share results?** Serialize selected fields from `engine.snapshots`, write analyzer reports with `AnalysisReport.to_csv()`, and store plots under `results/`.

## 7. Getting Support
- Check this handbook first.
- Review code docstrings and inline comments.
- Examine the tests in `tests/` for usage examples.
- Raise issues or start discussions in your internal tracking system / GitHub repository.

Keeping this FAQ handy should help diagnose most issues encountered during deployment and operation.
