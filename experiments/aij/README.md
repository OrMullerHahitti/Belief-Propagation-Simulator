# AIJ paper experiments

Three experiments bundled for the AIJ submission. All scripts live flat under `code/`, all outputs flat under `plots/`, all inputs flat under `data/`.

## Three experiments

### 1. Figures 5a, 5b, 8 — belief traces on cycle graphs

Damped Min-Sum BP (damping=0.9) on binary-domain N-cycle graphs, plotting per-variable belief trajectories.

- **Figure 5a** — consistent assignment, no tail
- **Figure 5b** — consistent assignment, with tail
- **Figure 8** — inconsistent assignment, no tail (Theorem 1 demonstration)

Cycle sizes: 3, 6, 9, 12. Iteration counts: 200 for fig5a/5b, 100 for fig8.

```bash
# replay all 12 plots from saved data — fast
uv run python experiments/aij/code/reproduce_figure.py --batch-bw --replay

# regenerate the underlying CSV datasets (slow)
uv run python experiments/aij/code/generate_fig58_csv.py
```

Outputs in `plots/`:
- `figXa_cycleN.pdf` + `figXa_cycleN_cost_tables.json` (24 files for fig5a/5b)
- `fig8_cycleN.pdf` + `fig8_cycleN_cost_tables.json` (8 files)

### 2. Signal propagation

Measures how a single activated factor's signal reaches a distant observer variable through BP message passing. Compares undamped BP vs damped BP (damping=0.9).

```bash
uv run python experiments/aij/code/run_signal_propagation.py
uv run python experiments/aij/code/plot_signal_propagation.py
uv run python experiments/aij/code/plot_signal_range_analysis.py
```

Outputs in `plots/`:
- `factor_graph_comparison.png`, `signal_comparison_bars.png`
- `scatter_damped.png`, `scatter_undamped.png`, `scatter_side_by_side.png`

### 3. Structured vs random cost tables

Evaluates how replacing structured cost tables with random ones affects BP convergence. 11 graph variants progressively replace 0%–100% of the structured factors with random ones.

```bash
uv run python experiments/aij/code/generate_structured_vs_random_graphs.py
uv run python experiments/aij/code/run_structured_vs_random.py
uv run python experiments/aij/code/plot_structured_vs_random.py
```

Outputs in `plots/`:
- `cost_curves_bw_BPEngine.png`, `cost_curves_bw_DampingEngine.png` (one figure per engine, side legend)
- `final_cost_bw.png`

## Layout

```
aij/
├── code/                                 # all scripts + utils/, no propflow
├── plots/                                # final PDFs / PNGs + JSONs that fed them
├── data/                                 # examples, traces, results, graph pickles
├── README.md
└── FIG58_DATASET_AUDIT.md                # data-provenance notes for fig5/8
```

See `code/utils/fig58_repro.py` for the BP runner / classifier; `code/utils/plot_helpers.py` for shared plotting helpers.
