# DABP weight figures

All figures about the weights DABP learns, in one place. DABP learns two kinds:
**edge weights** (the attention share each incoming neighbor's message gets when
a variable builds an outgoing message; uniform shares reproduce plain min-sum)
and **damping weights** (the weight on the previous message of the same edge
when the new message is mixed with it). Every panel shows a weight value itself
(on its 0-1 scale) or a cost — no ratios, logs or normalized differences.
Three data sources feed it:

| Folder | Experiment | Data |
|---|---|---|
| `small_10agents_50seeds/` | 10 agents, domain 10, density 0.3, seeds 0-49, run until the assignment is stable for 25 iterations (cap 1000) | `experiments/dabp_weights/data/` (0.5/0.5) and `data_asym/` (0.95/0.05) |
| `bigger_seed0/` | one graph (edge probability 0.5, domain 10, graph seed 0) at 20 and at 50 nodes, every iteration kept, stopped when assignments and weights are stable for 100 iterations | `experiments/dabp_node_dynamics/outputs/{20,50}nodes_seed0/` |
| `damping_audit/` | learned damping/attention replaced by fixed values: 81 runs on the saved seed-0 20-node graph, then 540 runs on 5 graphs x 2 objectives | the Codex worktree's `results/dabp_damping_audit/` (branch `orx/dabp-damping-audit`) |

## Layout

```
small_10agents_50seeds/
  split_0.5_0.5/      damping_weights, split_halves_damping, split_halves_edge_weights,
  split_0.95_0.05/    attention_weights, cost_and_convergence, structure
  compare_splits.pdf
bigger_seed0/
  20nodes/  split_0.5_0.5/, split_0.95_0.05/ (same five, no structure), compare_splits.pdf
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
- `split_halves_damping.pdf` — each original factor is split into half A (the
  larger cost share) and half B; a variable therefore sends one message to
  each half. Panels: the damping weight of the message to half A against the
  damping weight of the message to half B (one point per pair), both as
  histograms, the pair that moved most over the run per head, and the
  median / range of each half over the run.
- `split_halves_edge_weights.pdf` — the other kind of weight for the same two
  halves: when the variable builds its message to a third factor, both
  halves' incoming messages are among the sources, and each gets an attention
  share. Panels: share of half A against share of half B (one point per pair
  and outgoing message), both as histograms, the pair whose shares drifted
  apart most over the run, and the median / range of each half over the run
  for messages with the most common neighbor count (dashed line = uniform).
- `edge_pair_ratios_all.pdf`, `edge_pair_ratios_grid/degree_{d}.pdf` (0.95/0.05
  small set, seed 0 only; `code/plot_pair_ratios.py`) — the ratio share(half A)
  / share(half B) of every edge pair over the run, all pairs in one plot and
  one panel per pair, one file per degree of the variable the message is sent
  to (the variable on the other side of the third factor). Pairs that share
  the variable and the factor overlap almost exactly, whichever third factor
  the message goes to.
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
