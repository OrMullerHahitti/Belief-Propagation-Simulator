# DABP weight figures

All figures about the per-edge weights DABP learns, in one place. Every panel
shows a weight value itself (on its 0-1 scale) or a cost — no ratios, logs or
normalized differences. Three data sources feed it:

| Folder | Experiment | Data |
|---|---|---|
| `small_10agents_50seeds/` | 10 agents, domain 10, density 0.3, seeds 0-49, run until the assignment is stable for 25 iterations (cap 1000) | `experiments/dabp_weights/data/` (0.5/0.5) and `data_asym/` (0.95/0.05) |
| `bigger_seed0/` | one graph (edge probability 0.5, domain 10, graph seed 0) at 20 and at 50 nodes, every iteration kept, stopped when assignments and weights are stable for 100 iterations | `experiments/dabp_node_dynamics/outputs/{20,50}nodes_seed0/` |
| `damping_audit/` | learned damping/attention replaced by fixed values: 81 runs on the saved seed-0 20-node graph, then 540 runs on 5 graphs x 2 objectives | the Codex worktree's `results/dabp_damping_audit/` (branch `orx/dabp-damping-audit`) |

## Layout

```
small_10agents_50seeds/
  split_0.5_0.5/      damping_weights, split_halves, attention_weights,
  split_0.95_0.05/    cost_and_convergence, structure
  compare_splits.pdf
bigger_seed0/
  20nodes/  split_0.5_0.5/, split_0.95_0.05/ (same four, no structure), compare_splits.pdf
  50nodes/  same
damping_audit/
  audit_summary.pdf
```

## What each figure shows

- `damping_weights.pdf` — the weight DABP puts on the previous message (its
  learned damping): histogram of every edge's final value, the same per
  attention head, every edge over the run for one seed, and the lowest /
  median / highest edge over the run pooled over seeds. The dashed line is
  the start value 0.5.
- `split_halves.pdf` — each original factor is split into half A (the larger
  cost share) and half B; a variable therefore has two edges per factor.
  Panels: half-A weight against half-B weight (one point per pair), both as
  histograms, one pair over the run per head, and the median / range of each
  half over the run.
- `attention_weights.pdf` — the share each incoming neighbor gets when the
  edge aggregates its messages: learned share against the uniform share
  1/(number of neighbors), the share of the twin half against the other
  factors, all incoming shares of one edge over the run, and the twin-half
  share over the run.
- `cost_and_convergence.pdf` — cost of the current assignment over the run;
  for the 50-seed set also how long each seed ran, all cost curves, and best
  cost against final cost; for the single-graph runs, how many variables still
  change value each iteration.
- `structure.pdf` (small only) — the pair's damping weight against the
  variable's number of neighbors, bridge vs cycle factors, and the factor's
  cost spread / range, with the rank correlation.
- `compare_splits.pdf` — the 0.5/0.5 and 0.95/0.05 splits side by side on the
  same problems: final cost, run length (or cost and changes over the run for
  a single graph), the damping-weight histograms and the half-A/half-B scatter.
- `audit_summary.pdf` — best and final cost with learned damping against
  damping fixed at 0.5 / 0.9 for each representation; best cost with learned
  against uniform attention for every paired run; and where the learned damping
  weight went compared with the range the network could reach.

## Regenerate

```bash
bash experiments/dabp_plots/code/make_all.sh
uv run python experiments/dabp_plots/code/plot_small.py    # 50-seed set only
uv run python experiments/dabp_plots/code/plot_bigger.py   # seed-0 graphs only (loads ~1 GB for 50 nodes)
uv run python experiments/dabp_plots/code/plot_audit.py --results <audit results dir>
```

`plot_small.py` needs the raw npz files of `experiments/dabp_weights`
(`bash experiments/dabp_weights/code/run_full.sh` and `... --engine asym`).
`plot_bigger.py` needs the node-dynamics output directories. `plot_audit.py`
defaults to the worktree path used in September 2026.
