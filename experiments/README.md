# Experiments

Source code for the experiments in the paper. All experiments use [PropFlow](https://github.com/or-muller/propflow) (`propflow` package) for belief propagation on factor graphs.

## Setup

```bash
pip install propflow matplotlib pandas numpy
```

## Experiments

### 1. Figures 5a, 5b, 8 — Belief Traces on Cycle Graphs

Reproduces belief-trace figures from Zivan et al. Runs damped Min-Sum BP (damping=0.9) on binary-domain cycle graphs and plots per-variable belief trajectories over iterations.

- **Figure 5a** — consistent assignment, no tail (period-1 route from iteration 0)
- **Figure 5b** — consistent assignment, with tail (transient before locking)
- **Figure 8** — inconsistent assignment, no tail (variables cycle through all domain values)

**Generate all 12 plots (3 figures x 4 cycle sizes):**

```bash
python reproduce_figure.py --batch-bw --n-examples 10
```

**Single figure:**

```bash
python reproduce_figure.py --figure 5a --cycle-size 3
```

**Generate CSV datasets (50 examples per configuration):**

```bash
python generate_fig58_csv.py
```

Output: `generated_csv/{figure_5a,figure_5b,figure_8}/cycle_{3,6,9,12}/` with `examples.csv` (graph metadata) and `traces.csv` (per-iteration beliefs).

### 2. Signal Propagation

Measures how a single activated factor's signal reaches a distant observer variable through BP message passing. For each factor in a random graph (20 vars, density 0.25), sets its cost table to ones (all others zero) and records the final Q-message at a fixed observer after 50 iterations. Compares undamped BP vs damped BP (damping=0.9).

```bash
cd signal_propagation
python run_experiment.py
python plot_results.py
```

Output: `results/signal_results_combined.csv`, scatter plots in `scatter_distributions/`.

### 3. Structured vs Random Cost Tables

Evaluates how replacing structured cost tables with random ones affects BP convergence. Structured tables have a single zero-cost cell at `[0,0]` (one clear preferred assignment) with all other entries in [100, 200). Random tables have all entries in [100, 200) with no preferred assignment. Generates 11 graph variants from a fixed topology (100 vars, density 0.7), progressively replacing 0%–100% of the structured factors with random ones in 10% steps, then runs both undamped and damped BP for 200 iterations each.

```bash
cd structured_vs_random
python generate_graphs.py
python run_experiment.py
python plot_results.py
```

Output: per-graph CSVs in `results/`, cost-curve and final-cost plots in `plots/`.
