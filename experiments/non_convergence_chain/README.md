# Non-Convergence Chain Diagnostics

This package contains two first-class experiment families:

- the specific Min-Sum BP non-convergence example on the symmetrically split chain;
- seeded random-graph split-percentage studies.

The non-random chain structure is already symmetrically split at initialization:

```text
        F12_1             F23_1
       /     \           /     \
X1 ---       --- X2 ---       --- X3
       \     /           \     /
        F12_2             F23_2
```

It is designed to test diagnostic hypotheses, not to prove a theorem. Reports use
terms such as candidate condition, diagnostic signature, and observed behavior.

## Provide The Example

Paste the meeting cost tables into:

```bash
experiments/non_convergence_chain/configs/nonconverging_chain_template.yaml
```

Paste one table for `F12` over `["X1", "X2"]` and one table for `F23` over
`["X2", "X3"]`. The graph builder creates two parallel copies of each table:
`F12_1`, `F12_2`, `F23_1`, and `F23_2`. By default each copy keeps the full
pasted table (`symmetric_chain_cost_scale: 1.0`), matching the diagrammed
symmetric structure. The CLI fails with a clear error while either table is
`null`; no arbitrary cost tables are shipped as the meeting example.

## Run The Study

The chain study is enabled with `run_chain: true` and requires the pasted
`F12`/`F23` tables. It runs on the four-factor symmetric structure.

```bash
.venv/bin/python -m experiments.non_convergence_chain.run_non_convergence_study \
  --config experiments/non_convergence_chain/configs/nonconverging_chain_template.yaml \
  --out results/non_convergence_chain \
  --max-iter 200 \
  --damping 0.5 0.9 \
  --long-damping-max-iter 10000
```

The chain study runs standard Min-Sum, damping, long damping, trace recording,
diagonal diagnostics, and cost-table-only local diagnostics on the symmetric
structure. It does not run mid-run split experiments; those belong to the
random-graph study family.

## Random Graph Split-Percentage Studies

Random graph studies use `FGBuilder.build_random_graph` and do not require the
meeting `F12`/`F23` tables. Use the random graph template or set
`run_chain: false` and `random_graph.enabled: true` in another config:

```yaml
run_chain: false

random_graph:
  enabled: true
  num_vars: 50
  domain_size: 2
  density: 0.25
  split_at_iters: [20, 40, 60, 80]
  split_percentages: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  percentage_split_at_iter: 20
  run_split_at_sweep: true
  run_percentage_sweep: true
  run_combined_sweep: true
  split_transfer_modes: ["transfer"]
```

```bash
.venv/bin/python -m experiments.non_convergence_chain.run_non_convergence_study \
  --config experiments/non_convergence_chain/configs/random_graph_template.yaml \
  --out results/non_convergence_random_graph
```

The CLI also supports an explicit family override:

```bash
--mode chain
--mode random-graph
--mode both
```

Use `--mode both` only with valid chain cost tables, because chain diagnostics
still need the pasted meeting example.

Random graph split runs use the core `MidRunSplitEngine`. The primary mode is
`transfer`, which redistributes each prior `R[F->X]` into `p * R` from `F'`
and `(1 - p) * R` from `F''`. This preserves the aggregate factor-to-variable
contribution at the variable inbox (so the variable's belief and its Q
messages to unaffected neighbors are preserved on the split iteration). It
does **not** preserve the Q messages going into the clones themselves: each
clone's inbox picks up the sibling clone's transferred share, so each clone's
first outgoing R is not a canonical continuation of the un-split dynamics.
Treat the split iteration as an empirical transfer heuristic, not as a
mathematically canonical mid-run split.

The random graph runner produces three separate sweeps:

- split-at sweep: vary `split_at_iters` while splitting all factors;
- percentage sweep: vary `split_percentages` at `percentage_split_at_iter`;
- combined grid: run every `split_at_iters` x `split_percentages` pair.

`split_percentages` may be written as fractions (`0.1`) or whole percentages
(`10`); both mean 10%.

All random-graph runs in one CLI invocation start from the same seeded base
graph. The runner builds that graph once, deep-copies it for every experiment,
and records a `random_graph_fingerprint` in each run summary. Random run
directories are grouped by graph size/density and experiment setting, while
the seed is kept in the filenames inside the directory, for example
`random_graph_vars_20_density_0_25_split_at_20_all_transfer/seed_0_trace.jsonl`.

Add `"reset"` to `split_transfer_modes` only when you deliberately want a
cold-message comparison.

## Outputs

The output root contains aggregate files:

- `summary.json`
- `summary.csv`
- `condition_report.md`
- `plots/run_cost_comparison.png`

Each experiment run is saved in its own directory:

- `standard/`
- `damping_lambda_<lambda>/`
- `long_damping_lambda_<lambda>/`
- `random_graph_vars_<V>_density_<D>_standard/`
- `random_graph_vars_<V>_density_<D>_split_at_<N>_all_transfer/`
- `random_graph_vars_<V>_density_<D>_split_pct_<P>_at_<N>_transfer/`
- `random_graph_vars_<V>_density_<D>_split_grid_at_<N>_pct_<P>_transfer/`
- matching `_reset/` directories only when reset is explicitly enabled

Each chain run directory contains:

- `trace.jsonl`
- `snapshots.jsonl`
- `summary.json`
- `plots/assignments.png`
- `plots/global_cost.png`
- `plots/beliefs.png`
- `plots/belief_deltas.png`
- `plots/q_deltas.png`
- `plots/r_deltas.png`
- `plots/parity.png`
- `plots/diagonal_orientation.png`

Each random run directory contains seed-specific files:

- `seed_<S>_trace.jsonl`
- `seed_<S>_snapshots.jsonl`
- `seed_<S>_summary.json`
- `plots/seed_<S>/...`

Each trace row records assignments, global cost, beliefs, Q/R messages, binary
deltas, parity, reconstructed selected minimizers, and split event metadata.

## Interpreting Diagnostics

- Tail/t0 is the estimated first iteration of the final periodic regime.
- Immediate oscillation means the detected periodic regime starts at iteration
  0 or 1.
- Tail-induced behavior means the periodic regime appears only after a transient
  prefix.
- Binary diagonal orientation is `main`, `anti`, `mixed`, or `unknown` for
  selected minimizer entries. For domains larger than 2, the report uses a
  selected-entry signature instead of diagonal language.
- Damping stability is empirical for the tested configuration. The report checks
  whether oscillation returns and records measured mechanisms such as shrinking
  deltas or stabilized selected minimizers.
- Exact route analysis is implemented only for pairwise single-cycle graphs. For
  the primary chain, the route analyzer skips exact route classification and
  emits local cost-table/minimizer diagnostics.
