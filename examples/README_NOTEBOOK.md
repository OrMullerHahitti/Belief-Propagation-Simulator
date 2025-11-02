# Snapshot Analysis Tutorial Notebook

This Jupyter notebook provides a comprehensive, hands-on guide to using PropFlow's snapshot analysis module.

## Quick Start

1. **Ensure Jupyter is installed:**
   ```bash
   uv pip install jupyter ipykernel notebook
   ```

2. **Install the kernel (already done):**
   ```bash
   uv run python -m ipykernel install --user --name=propflow --display-name="Python 3 (PropFlow)"
   ```

3. **Open the notebook:**
   
   **Option A - VS Code:**
   - Open `analyzer_complete_tutorial.ipynb` in VS Code
   - Select kernel: "Python 3 (PropFlow)" from the kernel picker (top-right)
   
   **Option B - Jupyter Notebook:**
   ```bash
   uv run jupyter notebook analyzer_complete_tutorial.ipynb
   ```
   
   **Option C - JupyterLab:**
   ```bash
   uv run jupyter lab analyzer_complete_tutorial.ipynb
   ```

## What's Inside

The notebook covers:

1. **Basic Setup** - Imports and environment configuration
2. **Simple Examples** - 2-variable constraint problems
3. **Factor Graphs** - Building cycle graphs with FGBuilder
4. **Snapshot Capture** - Using the built-in `SnapshotManager`
5. **Data Analysis** - Cost trajectories, message statistics
6. **Visualization** - Argmin plots with `SnapshotVisualizer`
7. **Message Flow** - Deep dive into message passing
8. **Convergence** - Detecting when BP stabilizes
9. **Persistence** - Saving/loading snapshots as JSON
10. **Comparison** - Side-by-side engine analysis

## Troubleshooting

### Kernel not showing in VS Code

1. Reload VS Code window: `Cmd+Shift+P` → "Reload Window"
2. Select interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → `./.venv/bin/python`
3. Select kernel: Click kernel button → "Python 3 (PropFlow)"

### Import errors

Make sure you're using the uv environment:
```bash
uv pip list | grep jupyter
uv pip list | grep propflow
```

### Missing packages

Install dev dependencies:
```bash
uv pip install -e ".[dev]"
```

## Running All Cells

To execute the entire notebook from the command line:
```bash
uv run jupyter nbconvert --to notebook --execute analyzer_complete_tutorial.ipynb \
    --output analyzer_complete_tutorial_executed.ipynb
```

## Notes

- The notebook uses `np.random.seed(42)` for reproducibility
- All plots can be customized via matplotlib parameters
- Results are saved to `notebook_results/` directory
- The notebook takes ~30 seconds to run completely
