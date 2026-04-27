# Experiments

Source code and outputs for the experiments in the paper. All experiments use [PropFlow](https://github.com/or-muller/propflow) (`propflow` package) for belief propagation on factor graphs.

## Layout

```
experiments/
├── aij/        # paper bundle: signal propagation, structured-vs-random, fig 5a/5b/8
│   ├── code/   # all scripts (flat, files renamed for collision-free coexistence)
│   ├── plots/  # final PDFs / PNGs + the JSONs that produced fig5/8 plots
│   └── data/   # examples, traces, graph metadata, raw run results
├── other/      # one-off / supplementary experiments, each self-contained
│   ├── non_convergence_chain/{code,config,data,plots}
│   └── seed0_identity_timing_diagnostic/{code,config,data,plots}
└── archive/    # superseded plots kept for comparison/rollback
```

## Setup

```bash
uv pip install -e ".[dev]"
```

## aij/ — paper experiments

See `aij/README.md` for details. Quick reference:

```bash
# fig 5a / 5b / 8 — replay BP at the configured iteration count using saved cost tables
uv run python experiments/aij/code/reproduce_figure.py --batch-bw --replay
# (writes PDFs into experiments/aij/plots/)

# regenerate the fig5/8 example datasets (slow — rejection sampling)
uv run python experiments/aij/code/generate_fig58_csv.py

# signal propagation
uv run python experiments/aij/code/run_signal_propagation.py
uv run python experiments/aij/code/plot_signal_propagation.py
uv run python experiments/aij/code/plot_signal_range_analysis.py

# structured vs random
uv run python experiments/aij/code/generate_structured_vs_random_graphs.py
uv run python experiments/aij/code/run_structured_vs_random.py
uv run python experiments/aij/code/plot_structured_vs_random.py
```

## other/ — supplementary experiments

Each has its own `README.md` and self-contained `code/ config/ data/ plots/` subdirs.

- `other/non_convergence_chain/` — chain-graph non-convergence study (oscillation detection, route analysis, midrun-split, long-damping)
- `other/seed0_identity_timing_diagnostic/` — diagnostic for seed-0 identity / timing

## archive/

Old fig5a/5b PDFs at 100 iterations and the original combined `cost_curves_bw.png` (one figure with both engines as subplots). Superseded by the latest split versions in `aij/plots/`.
