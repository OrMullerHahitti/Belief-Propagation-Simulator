# Findings

A running log of what we have actually established, and what is still open.
Newest first. Each entry says whether it is verified against a real run or
still a guess.

---

## 2026-09-13 — Why splitting makes min-sum converge, and where it ends up

**Status: verified against real runs. Everything is in
`experiments/splitting_explanation/` — `EXPLANATION_formal.md` (theorems,
proofs, measurements), `EXPLANATION_hebrew.md` (plain-language version),
scripts `exp0`–`exp6`, `results/`, `plots/`.**

### The mechanism

- The symmetric 0.5/0.5 split is exactly plain min-sum on the original graph
  with one change in the variable rule: `Q = 2·cavity + (the R this factor sent
  me)`. Proved (Theorem 1) and checked numerically to 2e-8. Two ingredients:
  **doubling** (the factor sees outside evidence at twice its scale, so it needs
  half the margin to commit to one row) and **echo** (each edge becomes a
  two-step feedback loop).
- The echo is what locks. On the unsplit graph, `Q = belief` (echo only) freezes
  100% of DMS runs (median 208 iterations on random dense); `Q = 2·cavity`
  (doubling only) freezes fewer runs than plain DMS; both together (= the split)
  freeze 100% in a median of 62. Plain DMS: 80% in a median of 630; MS never.
  (A run counts as frozen only if at least 100 unchanged iterations follow its
  last assignment change; without that rule runs still moving at iteration
  1996 counted as frozen.)
- After the decisions lock, every committed factor is locally constant, so the
  damped messages settle geometrically. Measured decay per iteration after the
  freeze: 0.92–0.95 at λ=0.9, 0.58–0.61 at λ=0.5 (above λ because 5–23% of the
  arcs stay uncommitted).
- Large scale: with all arcs committed the decoded dynamics is synchronous best
  response, which is alternating minimisation of `cost_2(x, y)` on the bipartite
  double cover. Period 1 or 2. Measured: in every period-2 run each layer is
  exactly the best response to the other.

### The two bad solutions

- Undamped split ends in period 2 in 159/160 random runs at all densities
  0.05–1.0 and in 158/160 bipartite runs. The layers are unoptimised exactly on
  edges whose two endpoints both alternate: those edges cost 151–155 (a random
  table entry; the table mean is 149.5) against 107–132 for edges with fixed
  endpoints. Each layer has 22–41 improving single moves out of 50 variables.
- Density raises the share of such edges (38% at density 0.05, 66–70% at 0.8+)
  but does not create the alternation. What makes them *bad* solutions is
  non-bipartiteness: on bipartite graphs re-phasing the two layers gives two
  assignments with zero improving single moves, and the better one beats the
  damped split in 111/158 runs. The smallest example is a triangle of "be
  different" constraints: the undamped split alternates between 000 and 111
  (cost 30 each, the worst), because cost_2(000, 111) = 0.
- How much damping kills the alternation depends on the tables, not the density:
  random U[100,200) tables need λ ≥ 0.4 at every density; "be different" tables
  (graph coloring) still alternate in 16/20 runs at λ=0.2 and 1/20 at 0.4.

### Why DMS+split is slightly worse than DMS on random dense (99758 vs 99575)

- Both end points are locally optimal up to three-variable moves (zero
  improving 1-, 2- and sampled 3-path moves in every frozen run). Paired on the
  seeds where DMS froze (40/50), the split's point is worse by 538 ± 49; where
  DMS did not freeze it is better by 1236. DMS wins the dense mean because it
  freezes in 80% of dense runs and only 38% of sparse ones.
- The split locks the nearest committed point to the current message state.
  Splitting DMS at iteration K gives 99543 (K=50) → 99296 (K=1500), always better
  than greedy 1-opt from DMS's decoded state at K, and the later the better.
- The Weiss–Freeman guarantee on the split graph covers only 1- and 2-variable
  moves (every edge is a 4-cycle); on the unsplit graph it covers every
  tree-shaped move. The measured gap sits in moves of 4+ variables (not
  enumerated).

### Practical

- With the split, λ = 0.2–0.5 freezes fastest (median 17–23) and costs the same
  as λ = 0.9 on random dense/sparse, but 3/50 dense runs never settle at λ = 0.5
  (long-period wandering, not period 2) and on meeting scheduling 14/50 stay in
  period 2, so 0.9 is the safe default.
- An asymmetric split p = 0.9–0.95 freezes 95–100% of the runs in 90–180
  iterations at a better final cost than both the symmetric split and unsplit
  DMS (dense 99798 vs 100300 / 100004; sparse 14388 vs 14518 / 14889).

### Implementation facts that matter for the AAAI numbers

- `compute_R` casts Q messages to the cost table's dtype
  (`src/propflow/bp/computators.py:200`). random_dense and random_sparse tables
  are int64, so every unsplit factor in the recorded runs saw trunc(Q); split
  copies are float and did not truncate. Effect on DMS over 50 seeds: +346 ± 281
  (dense), −134 ± 119 (sparse) — not measurable.
- With truncation and PropFlow's normalise-every-diameter schedule, the
  vectorised engine reproduces the recorded cost curves iteration for iteration
  (MS, MS+split, DMS+split, DMS split-at-K) on random dense/sparse and coloring.
  Damped-and-truncated runs (DMS before any split) are not bit-reproducible: the
  truncated value of 0.9·old + 0.1·new depends on the last float bit.
- Undamped, un-normalised message levels grow like (degree−1)^t; PropFlow
  reaches 1e17 in 23 iterations and decodes float noise. Normalising per step
  (a per-message constant) fixes it and is what every cross-check does.
- `data_cuda/meeting_scheduling_raw_costs.csv`: the DMS and split rows carry
  integer costs (no tie-break unaries); only the MS rows match the current
  `problems.py` builder. Still to be resolved.

---

## 2026-09-08 — compute_R reads the cost-table axes in the wrong order

**Status: verified, and fixed in `9c2f4c8`. The reruns are still in
progress, so the results committed here are still the pre-fix ones.**

This is the one that matters. Read it before you trust any min-sum number in
`experiments/aaai/`.

### What goes wrong

A factor's cost table has one axis per variable it connects to. Axis 0 belongs
to one variable, axis 1 to another, and the factor records that mapping in
`connection_number`.

When the factor builds its outgoing messages it ignores that mapping. It walks
the messages in its inbox and assumes the first message it happens to hold
belongs to axis 0, the second to axis 1, and so on.

The engine fills that inbox in a different order. It sorts variables by name as
plain strings, so `x13` sends before `x2` — "1" sorts before "2". A factor
joining `x2` and `x13` therefore receives `x13` first and treats it as axis 0,
when the cost table says `x13` is axis 1.

The result is that the factor optimizes the transposed table. Costs are still
scored against the real table, so nothing crashes and no test fails — the run
just quietly solves a slightly different problem.

### Why no test caught it

String order and numeric order only disagree once you reach two-digit names.
Any graph with 9 or fewer variables is unaffected, and every test graph is
small.

### How we confirmed it

A scratch patch that sorts the incoming messages by `connection_number` makes
propflow's DMS (damping 0.5, split 0.5) reproduce the frozen DABP trajectory
exactly on unary-free graphs. On AAAI `random_dense` seed 0 it gives 98681,
which is the learned `Attentive_SymSplit` result for that seed. Unpatched
propflow gives 100485.

### How far it reaches

Seed 0, counting factors that are both wrongly ordered and asymmetric:

| benchmark | affected / total |
|---|---|
| random_sparse | 30 / 145 |
| random_dense | 161 / 726 |
| random_ternary | 32 / 84 |
| scale_free | 36 / 138 |
| random_sparse_ternary | 32 / 84 |
| random_dense_ternary | 169 / 491 |
| scale_free_ternary | 30 / 48 |
| meeting_scheduling_ternary | 62 / 87 |

Not affected: `graph_coloring`, binary `meeting_scheduling`, and
`graph_coloring_ternary` — their tables are symmetric, so transposing changes
nothing. DABP is not affected either; its own builder already uses
`connection_number`.

Every MS / DMS / split / merge line in `experiments/aaai` has been running this
way since 2025-07-14, as has any other propflow run with more than 9 variables
and asymmetric tables.

### The fix

`compute_R` now sorts the incoming messages by the factor's
`connection_number` before it assigns axes, so the loop index is the
variable's real dimension in the cost table. It ships with a regression test
that builds a factor over `x2` and `x13` and checks each variable gets the
message for its own axis, which is the case the old code got backwards.

Evidence: `src/propflow/bp/computators.py` (the sort, and the `enumerate` that
follows it) and `src/propflow/bp/engine_base.py:81` (the string sort that fills
the inbox in the first place).

### What still needs doing

Every AAAI min-sum number produced before this fix came from a partly
transposed problem, so all of them need regenerating.

That rerun covers `random_dense`, `random_sparse` and `scale_free`, and it was
still running when this note was written — see
`experiments/aaai/logs/rerun_axis_fix.log`. The **ternary benchmarks and the
AIJ figure 5/8 examples were not part of it and are still pre-fix**.

Until a benchmark has been regenerated, treat its cost CSVs and the plots built
from them as pre-fix results, and never put a pre-fix line and a post-fix line
on the same axes.

---

## 2026-09-08 — DABP's learned weights never leave their starting point

**Status: verified across three separate experiments.**

DABP is supposed to learn two things per edge. The **edge weight** is how much
attention a variable pays to each incoming neighbour's message when it builds
an outgoing one; uniform edge weights are just plain min-sum. The **damping
weight** is how much of the previous message on the same edge is carried
forward.

Neither moves.

With a 0.5/0.5 symmetric split the two halves of a split factor stay bitwise
identical, because those nodes are automorphic in the network. A 0.95/0.05
split breaks that symmetry, and the halves still end up within 7.5e-5 of each
other on 10 agents, and within 6e-6 on 20-50 nodes. Every attention share stays
within 1.4e-4 of uniform. Damping weights sit within about 2e-4 of 0.5 on 10
agents and about 1e-4 on 20-50 nodes.

We then switched the learning off directly, and nothing changed. Fixed damping
of 0.5 or 0.9 reaches the same best cost as learned damping in every
representation — only the unsplit case fails to hold its best. Uniform
attention matches learned attention on violated-constraint count in all 270
pairs tested. Learned damping only ever ranged over 0.49962 to 0.50043, while
the reachable range is 0.269 to 0.731.

What does change results is the split ratio, not the learning. All 50 symmetric
seeds settle by iteration 30-55. Only 38 of 50 asymmetric seeds settle at all;
the other 12 oscillate until the iteration cap.

Experiments: `experiments/dabp_weights/`, `experiments/dabp_node_dynamics/`,
and the damping audit in the `orx/dabp-damping-audit` worktree. Figures:
`experiments/dabp_plots/`, raw weight values only.

---

## 2026-09-08 — Splitting alone is what creates the cycle

**Status: demonstrated in a notebook.**

Added a second worked example to `notebooks/split_tail_example.ipynb` using
split halves `[[0,20],[30,2]]` and no tail factor.

Without the tail there is a single factor over two variables, so the unsplit
graph is a tree and BP converges immediately. Splitting that factor produces
the loop X1 - F12' - X2 - F12'' - X1. So the example isolates what the split
itself does: beliefs inflate around the new loop while the argmin stays put at
the optimum.

---

## engine.history is deprecated — read results from snapshots

**Status: settled, applies to all new code.**

`engine.history` and `SnapshotHistoryView` are no longer the way to read a run.
Use the snapshots API instead.

Per-iteration cost comes from a snapshot's `global_cost`. Reach for a specific
iteration with `engine.get_snapshot(i)`, and for the whole run in order with
`engine.snapshots`. Those are not the same lookup: `snapshots` is a list built
from the recorded steps in sorted order, so its index only equals the iteration
number when every step was recorded. `engine.snapshot_map` gives the raw
step-to-snapshot dict, and `engine.latest_snapshot()` gives the last one.
`global_cost` is `None` when the engine did not compute it.

Assignments and beliefs come from the live `engine.assignments` and
`engine.get_beliefs()` during a manual `step(i)` loop, or from the snapshot
objects when a fuller `SnapshotManager` is configured. For lightweight batch
runs there is a cost-only manager in
`experiments/other/non_convergence_chain/code/run_random_graph_simulator_average.py`.

---

## Plot conventions

**Status: applied.**

Legends were covering the curves at the right edge, so they now sit outside the
axes on the right and the figures widened from 9 to 11 to make room. Labels are
short: `DMS d=.5 s=.5`, `DABP sym-split`, `MGM@200`.

DABP weight figures show raw weight values only — no log ratios, no normalized
differences — and the small and larger experiments stay in separate folders.
