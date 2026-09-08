# DABP Edge-Weight Analysis

Component analysis of the per-edge weights DABP learns while solving, on small
random problems. DABP (Deng et al., NeurIPS 2022) picks, for every directed
variable-to-factor edge and every iteration, a softmax pair of weights — one on
the attention-aggregated new message and one on the previous message (the
learned per-edge damping) — plus attention weights over the incoming messages
feeding each edge. The engine integration computes these tensors every
iteration; this experiment records and analyzes them.

## Problem spec

- 10 agents, domain 10, integer costs U[100, 200), Erdos-Renyi density 0.3
  (force-connected, so the graph is cyclic with ~13 binary factors)
- tiny random unary tie-break preferences (uniform [0, 1e-2)), as in the AAAI
  benchmarks
- seeds 0..49, one problem per seed, fully deterministic
  (`problems.build_random_10`)

## Engines

- `DABPEngineSymSplit` (`--engine symsplit`, default): DABP with its built-in
  SCFG split at a symmetric ratio of 0.5/0.5, so every original factor becomes
  two identical halves before iteration 0.
- `DABPEngine` (`--engine asym`): same network and driver with DABP's native
  asymmetric split ratio of 0.95/0.05, so the halves scale the original table
  by 0.95 and 0.05. Outputs go to `data_asym/`.

Upstream defaults otherwise (4 heads, `update_interval=20`,
`eff_iterations=2`, AdamW lr 1e-4). Runs on **CPU float64** — auto-select would
pick MPS/float32 on a Mac, and the split-pair symmetry measurement needs full
precision.

## Convergence rule

One BP iteration per engine step. A run stops when the assignment vector is
unchanged for 25 consecutive iterations (`--stable-iters`; larger than the
20-iteration training window, so a stable stretch spans at least one optimizer
step) or at the 1000-iteration cap (`--max-iter`; below `restart_period=2000`,
so DABP's message-state restart never fires inside a run).

## Recorded quantities

Per iteration, from `engine.weights_log` (see
`src/propflow/integrations/dabp/model.py`):

- `damped_weights [T, 2, H]` — per target edge and head, softmax pair of
  (new-message weight, previous-message weight). `damped[:, 1, :]` is the
  learned per-edge damping; its head mean is the effective damping lambda
  actually applied. T = 2 x (number of function nodes) under the symmetric
  split.
- `attention_weight [S, H]` — softmax share of each *other* incident function
  node inside a target edge's aggregation group. The twin half of the target's
  own factor appears in this group (only the target itself is excluded); its
  share is reported as `twin_share`.

Index provenance (`engine.weight_metadata()`) maps every row back to
(variable, original factor, split half).

## Split-pair analysis

For each variable and original factor, the symmetric split creates two edges —
one per half. Pairing them and comparing weights measures whether DABP breaks
the 0.5/0.5 symmetry. Caveat: the two halves are automorphic nodes of the GNN
with bitwise-identical inputs, so in exact arithmetic their trajectories are
identical; any asymmetry is seeded by float rounding and then amplified (or
not) by training. A ratio pinned at exactly 1 is therefore a meaningful result,
not a bug. Under the asymmetric 0.95/0.05 split (`--engine asym`) the two
halves have different inputs, so the automorphism is broken by construction
and any learned asymmetry shows up directly in the same ratio.

## Files

- `code/run_weights.py` — sequential runner; writes `data/raw/seed{NNN}.npz`
  (weights, costs, assignments, provenance) + `data/metadata.json`
- `code/analyze_weights.py` — writes the summary CSVs into `data/`
  (pair_asymmetry, pair_asymmetry_dynamics, edge_weights, attention_final,
  structure_correlation, correlation_stats, run_summary)
- figures: `experiments/dabp_plots/code/plot_small.py` reads both data dirs
  and writes `experiments/dabp_plots/small_10agents_50seeds/` (see the README
  there)

With `--engine asym`, `data/` becomes `data_asym/` throughout. Neither data
dir is committed (see the `.gitignore` in each); everything is regenerable
from the seeds.

## Run

```bash
bash experiments/dabp_weights/code/run_full.sh                # 50-seed suite, symsplit
bash experiments/dabp_weights/code/run_full.sh --engine asym  # 0.95/0.05 contrast
uv run python experiments/dabp_weights/code/run_weights.py --n-problems 1   # smoke test
```
