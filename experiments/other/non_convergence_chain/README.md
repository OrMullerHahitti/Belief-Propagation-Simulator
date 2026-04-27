# Non-Convergence Chain: Minimal Reproducibility Slice

This directory contains the first reviewable slice for the non-converging
chain study. It runs one experiment only:

```text
        F12_1             F23_1
       /     \           /     \
X1 ---       --- X2 ---       --- X3
       \     /           \     /
        F12_2             F23_2
```

The runner builds the symmetric chain from pasted `F12` and `F23` tables, runs
standard Min-Sum with `BPEngine` and `MinSumComputator`, records a trace, and
classifies the final behavior as converged, period-2 oscillation,
transient-then-oscillation, or unclear.

This code does not claim a theorem. It produces reproducible diagnostics for
later analysis.

## Provide The Cost Tables

Paste the exact meeting tables into:

```bash
experiments/non_convergence_chain/configs/nonconverging_chain_template.yaml
```

The template intentionally ships with `table: null`. The CLI fails clearly until
both `F12` and `F23` are provided.

Each pasted table is copied into two parallel factors. By default the copies use
the full pasted table (`symmetric_chain_cost_scale: 1.0`), matching the
symmetric-chain diagram.

## Run

```bash
.venv/bin/python -m experiments.non_convergence_chain.run_non_convergence_study \
  --config experiments/non_convergence_chain/configs/nonconverging_chain_template.yaml \
  --out results/non_convergence_chain \
  --max-iter 200
```

By default the runner records the full `max_iter` trace and keeps PropFlow
message normalization enabled. This avoids convergence-monitor aliasing hiding
period-2 behavior. Use `--stop-on-convergence` only when you explicitly want
the engine monitor to stop early.

Snapshot output is controlled by `save_snapshots` in the YAML config. Set
`save_snapshots: false`, or pass `--no-snapshots`, when full traces are enough.

## Outputs

The output paths are deterministic:

- `summary.json`
- `summary.csv`
- `condition_report.md`
- `standard/trace.jsonl`
- `standard/snapshots.jsonl` when `save_snapshots: true`
- `standard/summary.json`

Each trace row includes iteration, parity, assignments, global cost, beliefs,
Q/R messages, binary deltas, and reconstructed selected minimizers.

## Scope

Damping, splitting, random-graph sweeps, route analysis, and plotting are
intentionally outside this minimal slice. The CLI reports those diagnostics as
skipped so the base standard-run path stays small and reviewable.
