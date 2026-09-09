# DABP node dynamics

Inspect one saved graph at edge probability 0.5 under both symmetric
0.5/0.5 and asymmetric 0.95/0.05 splitting. The same saved cost tables and
initial network parameters are used in both runs. Every iteration is retained.
The current run has **50 variables**, with the earlier 20-variable run retained
in its own output directory.

## Run and replay

From the repository root, using the existing development environment with the
`dabp` extra installed:

```bash
uv run --no-sync python -m experiments.dabp_node_dynamics.run \
  --nodes 50 --out-dir experiments/dabp_node_dynamics/outputs/50nodes_seed0
```

An existing output directory is refused. To regenerate the reader without
rerunning either solver:

```bash
uv run --no-sync python -m experiments.dabp_node_dynamics.run \
  --out-dir experiments/dabp_node_dynamics/outputs/50nodes_seed0 --report-only
```

To replay the exact graph into a new directory, pass `--graph-file` pointing
to its `graph.json`, together with the original settings from `run.json`.
All settings are CLI flags (`--help` lists them). CPU float64 and one Torch
thread provide a reproducible execution path; exact replay is expected within
the recorded library/platform environment, not across arbitrary versions.

For a short smoke run:

```bash
uv run --no-sync python -m experiments.dabp_node_dynamics.run \
  --nodes 4 --domain 3 --max-iterations 6 --stable-window 4 --update-interval 2 \
  --out-dir experiments/dabp_node_dynamics/outputs/smoke
```

## The five output files

- `report.html`: self-contained, offline, selectable graph and trajectories.
  Open in a current Chrome, Edge, Firefox, or Safari. Native browser deflate
  decompression reads losslessly compressed float64 trajectories.
- `graph.json`: exact node order, factor names, ordered tensor axes, original
  cost tables, graph layout, and exact structural measurements.
- `run.json`: all settings, dependency versions, input/source fingerprints,
  initial network fingerprint, and stopping outcomes for both runs.
- `symmetric.npz` and `asymmetric.npz`: all raw iterations, without pickle.

Outputs are ignored by Git. No existing experiment or core solver is modified.
The reader uses the already installed Plotly library, embedded in the file;
there are no CDN scripts, remote data sources, or hosted components.

### Open the UI

Open `outputs/50nodes_seed0/report.html` directly, or use `reader.html` as the
entry page. `reader.template.html` is a build input; it contains placeholders
and is not an executable report.

If your embedded browser does not allow local-file URLs, serve only this run's
output directory on loopback:

```bash
uv run --no-sync python -m http.server 8769 --bind 127.0.0.1 \
  --directory experiments/dabp_node_dynamics/outputs/50nodes_seed0
```

Then open <http://127.0.0.1:8769/report.html>. The 50-variable viewer runs in the
`viewer50` window of the `propflow-dabp-node-dynamics` tmux session; the earlier
20-variable viewer remains at port 8768 in `viewer`. Attach with
`tmux attach -t propflow-dabp-node-dynamics`; stop the server with Ctrl-C in
that window.

## Defaults and stopping

The CLI default remains 20 variables; the current run explicitly passes
`--nodes 50`. Domain size 10, edge probability 0.5, graph seed 0 and model seed
0 are shared. Existing `FGBuilder` generates the graph and connects components if needed.
Both requested and realized density are recorded; 0.5 is a probability, not a
promise of exactly 95 edges. Binary costs are integer U[100,200), with unary
U[0,0.01) tie breakers, matching the previous experiment's cost family.

DABP uses its existing four heads, AdamW learning rate 1e-4, weight decay 5e-5,
cost scaling, and two selected loss iterations per 20-iteration training
window. The experiment does not change DABP's training objective.

Stop when **both** conditions hold over the last 100 recorded iterations:

1. the entire assignment vector is identical throughout the window;
2. every applied old-message damping and incoming message coefficient has
   `maximum - minimum < 0.001` throughout the window.

The cap is 2,000 iterations. An iteration-limit outcome is never relabeled as
convergence. Stability here is an operational observation, not a proof of
future convergence. Each variant stops independently. The original default
2,000-step message restart is not reached inside the run (steps 0–1999).
The window includes optimizer updates, but future parameter updates could still
change a currently stable trajectory. Stop diagnostics record assignment and
coefficient stability separately, including their first observed windows.

## Definitions

Structural metrics use the original undirected, unweighted variable graph:
degree, normalized betweenness, closeness, and clustering. All are computed
exactly using NetworkX. Splitting does not duplicate structural nodes.

For one outgoing target message `k`, let `lambda[k]` be the mean old-message
weight across heads and `new[k]` the mean new-message weight. For incoming
source row `s`, `alpha[s]` is its mean attention share and `k = source_target[s]`.
The actual incoming coefficient is:

```text
c[s] = incoming_count[k] * alpha[s] * new[k]
```

This is **the product of head means**, because the implementation averages
attention first, then applies the damping heads. Averaging head-wise products
would reconstruct a different algorithm. A focused integration test rebuilds
actual normalized messages from these coefficients for both variants, including
steps that cross optimizer updates.

Node summaries for a selected family of coefficients use:

- initial/final mean: mean over the node's coefficients at iteration 1/last;
- total movement: mean over coefficients of `sum(abs(w[t] - w[t-1]))`;
- largest jump: `max(abs(w[t] - w[t-1]))` over the node's coefficients and time.

Absolute differences are taken before averaging. Different run lengths affect
total movement; the reader always displays the duration of each run. One graph
supports descriptive within-graph comparisons, not independent-sample causal
claims. Raw attention shares are shown alongside actual coefficients so
degree-dependent normalization is visible.

For a split pair feeding the **same outgoing destination**, show the original
split `p`, both applied coefficients `a,b`, and:

```text
first-half share = p*a / (p*a + (1-p)*b)
```

This is a split-adjusted coefficient diagnostic. It does not change cost tables
and is not a measurement of full nonlinear message contributions. Excluded
halves are never filled with zero or compared against unrelated destinations.
Degree-one variables have no eligible common destination and show an explicit
unavailable state for this comparison.

DABP passes each attention score through a sigmoid before softmax. Consequently
the ratio of two source attention weights toward the same destination lies
between approximately 1/2.718 and 2.718. Their common degree and new-message
multiplier cancel in the ratio. Complete 95/5 compensation would require a
ratio of 19, so it is unreachable for this coefficient diagnostic. Symmetric
halves are automorphic in the model and can stay equal by construction. Neither
fact prevents recording weight evolution and its relationship to structure.

No logarithmic axes, log ratios, or logarithmic analysis transformations are
used. The solver's existing loss and training semantics remain unchanged.

## Raw schema

Open with `np.load(path, allow_pickle=False)`. `metadata_json` is a scalar JSON
string containing names, indices, settings, checksums, and the stopping outcome.

| Array | Shape | Meaning |
|---|---|---|
| `iteration` | `[I]` | one-based recorded iteration |
| `damped` | `[I,T,2,H]` | new/old message weights, per target and head |
| `attention` | `[I,S,H]` | source attention share, per target group and head |
| `assignments` | `[I,N]` | assignments in `ordered_names` order |
| `costs` | `[I]` | cost on the exact original tables |

`T` indexes outgoing variable-to-split-factor messages; `S` indexes each
eligible incoming-source/outgoing-target combination. Metadata connects every
row to the variable, original factor, split half, and destination. `fn_half=0`
uses `p`; `fn_half=1` uses `1-p`.

## Reader and chart contract

The agreed artifact is an experiment inspector with linked node, original
factor, and destination selection. Its primary workspace is the original graph;
the adjacent inspector supplies context. A restrained white surface, one blue
accent, and solid/dashed half identities keep overlapping curves readable.
Hover details, click-to-select, and linked selections provide the interaction.

- Graph: original topology and four properties; node selection highlights only
  that node's neighborhood. The stored layout is shared across both variants.
- Evolution: every-iteration lines, two split halves, paired variants with shared
  axis ranges. Damping, attention, and actual coefficients remain distinct.
- Changes: sortable table with one row per variable and clickable nodes.
- Structure: scatter plots at one-variable grain; property and behavior selectors;
  one observation per variable and variant, with shared scales and clickable points.
- Balance: actual coefficient lines plus the split-adjusted percentage diagnostic;
  50% reference line, per pair and eligible common destination.

The balance view also answers whether small split changes hold for every
variable. Its per-variable table checks every eligible factor/destination pair and
selects, separately for each variable and run, the pair with the largest
absolute departure from its own first recorded split at any iteration. Initial
and final splits refer to that same pair, never an average across connections.
The departure is measured in percentage points. A variable with no eligible
pair is explicitly unavailable. Clicking a row selects that pair in the linked
detail view for both runs.

The detail view can show either the calculated first-half percentage or its
signed change from iteration 1. Change curves retain every recorded iteration,
use a common linear scale centered on zero, and expose both split percentages
on hover. Their focused scale is labeled in percentage points. The existing
blue/gray styling, solid/dashed identities, and embedded Plotly runtime are
retained. Browser QA covers all rows, both runs, sorting, and the exact
factor/destination selected by each row.

Variable selectors show an explicit rank and retain original node identities.
Graph overview ranks by normalized betweenness. Weight evolution ranks by the
largest damping or applied-coefficient jump across either run, selected by its
order control. Node changes defaults to largest jump in the selected run and
quantity; its selector follows the table's current sort. Structure ranks by
the selected behavior's maximum across either run and defaults to largest jump.
Split balance defaults to largest departure in the selected run. Ties use
original variable order; unavailable pairs sort last. Opening a view through
the sidebar selects its first-ranked variable, while linked node clicks retain
the requested variable. Graph coordinates and scatter axes keep their meanings.

Weight evolution includes each displayed item's starting value from iteration
1: old damping, new-message weight, attention share, and applied edge coefficient
for both halves and both runs. Individual starting values use lossless decimal
representations of the displayed float64 values. Initial/final node averages
also retain their displayed precision instead of rounding to six decimals.

The sixth view, Architecture, is an illustrative two-variable/two-factor graph.
Its split and outgoing-message controls show the active incoming path and the
excluded return path. It explains cost scaling, splitting, the GRU/GAT/attention
weight generator, and the damped variable update. In this topology each target
has one eligible incoming message, so attention and incoming count are both 1.
It does not provide numerical weights from a separate two-node training run.
The common-destination split diagnostic requires another incident factor and
is explicitly distinguished from this minimal message-flow example.

All axes are linear. Observed-range mode is labeled and uses common limits for
paired plots. It has a minimum visible span to avoid magnifying floating-point
noise. Include-zero mode is available. Each curve ends at its actual stop;
there is no resampling, extension, smoothing, or averaging across runs.

## Validation

```bash
uv run --no-sync python -m pytest tests/test_dabp_node_dynamics.py -q
uv run --no-sync python -m black --check experiments/dabp_node_dynamics tests/test_dabp_node_dynamics.py
uv run --no-sync python -m flake8 experiments/dabp_node_dynamics tests/test_dabp_node_dynamics.py
node --max-old-space-size=6144 tests/test_dabp_node_reader.cjs experiments/dabp_node_dynamics/outputs/50nodes_seed0/report.html
```

Tests cover exact graph/table replay, non-cancelling movement, whole-window
stability, missing split halves, actual message reconstruction, and an end-to-end
paired archive/report smoke. The report builder rejects mixed graph, settings,
initialization, provenance, or outcome data.

The Node reader check uses DOM/Plotly doubles to exercise all variable
selections, both split settings, plot data, sort controls, axis selectors, and
all six views, including ranking, starting weights, and architecture controls.
A separate browser pass through the loopback viewer verifies
the rendered graph, node clicks, linked factor/destination selection, table
sorting, structural selectors, and all six views. The entry-page regression
test ensures opening `reader.html` no longer exposes the unbuilt template.
