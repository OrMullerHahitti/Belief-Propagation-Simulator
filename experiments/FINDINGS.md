# Findings

A running log of what we have actually established, and what is still open.
Newest first. Each entry says whether it is verified against a real run or
still a guess.

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
transposed problem, so all of them need regenerating. That rerun was running
when this note was written and is not finished. Until it is, treat the cost
CSVs and the plots built from them as pre-fix results, and do not compare a
pre-fix line against a post-fix one on the same axes.

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
