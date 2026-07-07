# AAAI paper experiments

Six DCOP benchmarks x algorithm families, 50 problem instances each,
reporting mean cost and paired statistical significance. Layout follows
`experiments/aij/`: scripts in `code/`, CSVs in `data/`, PDFs in `plots/`.

## Benchmarks (50 instances each, seeds 0-49)

| Name | Spec |
|---|---|
| `random_sparse` | 50 agents, domain 10, p1 = 0.1, integer costs U[100, 200) |
| `random_dense` | 50 agents, domain 10, p1 = 0.6, integer costs U[100, 200) |
| `random_ternary` | 50 agents, domain 10, true arity-3 factors with p3 = 2 * 0.1 / (50 - 2), integer costs U[100, 200); only `DMS_split_0.5` is run |
| `graph_coloring` | 50 agents, 3 colors, p1 = 0.1, not-equal constraints: equal = 10, else 0 |
| `scale_free` | Barabasi-Albert per Cohen, Galiki & Zivan (AIJ 2020) §6: 7 initial agents randomly connected, each new agent attaches preferentially to 3 existing agents, n = 50, domain 10, integer costs U[100, 200) (the §6.2 cost range used for the splitting experiments; §6.1 used U[0, 100)) |
| `meeting_scheduling` | Cohen et al. (AIJ 2020) §6: 90 agents schedule 20 meetings into 20 time slots; each agent participates in two random meetings; per constrained meeting pair travel time ~ U{6..10}; cost = number of overbooked (shared) agents when the slot difference is below the travel time |

All problems get tiny random unary value preferences (U[0, 1e-2)) for tie
breaking, as in all Max-sum versions of the AIJ paper (Farinelli et al. 2008
style). Their total mass (<= 0.5) stays below the smallest structural cost gap
(1 for random/meeting, 10 for coloring), so it cannot change which assignment
is optimal; without them min-sum is degenerate on the symmetric problems (graph
coloring especially).

Random/coloring topologies use `FGBuilder.build_random_graph` (Erdos-Renyi,
components force-connected — engines require a connected graph). The ternary
benchmark builds true 3-variable factors directly because `FGBuilder`'s random
helper is binary-only.

## Algorithms

| Label | Item | Description |
|---|---|---|
| `MS` | baseline | `BPEngine`, normal undamped min-sum on the original factor graph |
| `DMS` | 2a | `DampingEngine`, lambda = 0.9 |
| `DMS_split_0.5` | 2b | `DampingSCFGEngine`, constant symmetric split (0.5/0.5) |
| `DMS_split_0.4_0.6` | 2c | DMS on a *random* SCFG: each cost-table entry c is split into u·c / (1-u)·c with u ~ U[0.4, 0.6) — the "0.4-0.6" version of the AIJ paper (§6.2), which it found best |
| `DMS_split_at_{50,100,300,500,1000}` | 2d | `MidRunSplitEngine` + damping; all factors split at iteration K, `transfer` mode (prior R messages redistributed p/(1-p) across clones, as in the late-split experiments in `experiments/other/non_convergence_chain`) |
| `DMS_split_at_1500` | 2d+ | same, **`random_dense` only** — an extra late split point; opt-in (not part of `--algorithms all`), added via `run_full.sh`. 1500 leaves 500 post-split iterations of the 2000 horizon |
| `Attentive` | 2e | DABP (Deep Attentive Belief Propagation): a per-instance graph-attention network driven one BP iteration per step (`AttentiveEngine` -> `propflow.integrations.dabp.DABPEngine`; needs the optional `[dabp]` extra). Uses DABP's built-in SCFG step, which splits every factor **asymmetrically (0.95/0.05)** at graph-build time (i.e. from iteration 0). Far more expensive per iteration than min-sum, so its curve is **stretched** on the cost plots (see Notes) |
| `Attentive_NoSplit` | 2e′ | same DABP network and driver as `Attentive` but with DABP's SCFG split **disabled** — one DABP factor per original binary factor (`AttentiveNoSplitEngine` -> `DABPEngineNoSplit`) |
| `Attentive_SymSplit` | 2e″ | same DABP network and driver as `Attentive` but the SCFG split is **symmetric (0.5/0.5) from the start** instead of the 0.95 default (`AttentiveSymSplitEngine` -> `DABPEngineSymSplit`) — the DABP counterpart of the `*_split_0.5` min-sum families |
| `Optimal` | 2f | depth-first branch and bound on the original tables, 60 s/instance limit; reported only for instances where the search completed (expect graph_coloring and meeting_scheduling; the domain-10 benchmarks generally won't finish) |
| `MS_split_0.5` | 2g | undamped min-sum on a 0.5/0.5 SCFG |
| `MS_split_MGM_200` | 2h | the two assignments at iterations 198/199 of the `MS_split_0.5` run (its period-2 oscillation branches = "the two options after 200 iterations") merged with MGM-1 restricted to the per-variable binary menu {b1[v], b2[v]}; run from both seeds, best kept |
| `MS_split_opt_200` | 2i | same two options merged *optimally*: branch and bound over the binary menus, on tables conditioned on the menus (300 s cap). Exact on every benchmark except `random_dense`, where the conditioned subproblem has induced width ~25-36 (30-46 disagreeing variables on a dense graph) and proving optimality is infeasible — there the reported value is the best menu merge found within the cap, warm-started from the MGM result (so always <= `MS_split_MGM_200`); the run log counts these instances |

`MS_split_0.5`, `MS_split_MGM_200` and `MS_split_opt_200` share one engine run
per instance, so all three see exactly the same oscillation branches.
`random_ternary` is intentionally narrower and runs only `DMS_split_0.5`; DABP
is excluded because the integration supports only unary/binary factors.

## Running

```bash
# full run (50 problems x all algorithms x 2000 iterations) — hours
uv run python experiments/aaai/code/run_experiments.py --benchmarks all

# one benchmark / subset of algorithms / smoke test
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks graph_coloring --n-problems 5 --max-iter 300 --merge-at 200

# means + paired t-tests + Wilcoxon signed-rank (per benchmark)
uv run python experiments/aaai/code/analyze_results.py

# DABP/DMS per-iteration time ratio (one instance each, DABP-supported benchmarks only) -> data/dabp_timing.csv
uv run python experiments/aaai/code/time_dabp.py

# mean per-iteration cost curves, DABP stretched onto the time axis (PDF)
# also writes zoom PDFs for crowded lower-cost curve clusters
uv run python experiments/aaai/code/plot_results.py
```

CSV-writing scripts back up existing `data/*.csv` files to
`backups/data_before_*_<timestamp>/` before modifying outputs. Pass
`--skip-backup` only when running against disposable data.

## Outputs

- `data/{benchmark}_final_costs.csv` — algorithm, seed, final_cost, anytime_cost
- `data/{benchmark}_raw_costs.csv` — algorithm, seed, iteration, cost (cost is always evaluated on the **original** cost tables, also after splitting)
- `data/{benchmark}_metadata.json` — run parameters
- `data/{benchmark}_summary.csv` — per-algorithm mean/std (final + anytime)
- `data/{benchmark}_significance.csv` — pairwise paired t-test + Wilcoxon p-values on both metrics, computed over the common solved seeds (AIJ 2020 reported paired t-tests on final and anytime results)
- `data/dabp_timing.csv` — per-benchmark DABP/DMS per-iteration time ratios (from `time_dabp.py`, excluding `random_ternary`), used to stretch each DABP curve: `ratio` (`Attentive`, 0.95 split), `nosplit_ratio` (`Attentive_NoSplit`), `symsplit_ratio` (`Attentive_SymSplit`, 0.5 split)
- `plots/{benchmark}_cost.pdf` — colored mean per-iteration cost (the only plot; anytime and B&W variants are no longer generated)
- `plots/{benchmark}_cost_zoom.pdf` — tail-window close-up of the crowded lower-cost curve cluster, generated from the same CSV data when a separated cluster is detected

## Notes / interpretation decisions

- **Attentive (2e) = DABP**, and its per-iteration cost is not comparable to
  min-sum (a full graph-attention forward pass, plus a periodic backward +
  optimizer step, per BP iteration). `code/time_dabp.py` times one instance
  (seed 0) of each benchmark for DABP vs DMS and writes the per-iteration ratio
  to `data/dabp_timing.csv`. `plot_results.py` reads that ratio and **stretches
  the DABP curve onto a wall-clock-equivalent x-axis**: DABP iteration `k` is
  drawn at `x = ratio * k` (e.g. a measured 19:1 ratio makes each DABP step jump
  19 iterations), so DABP is read as "where it reaches within the wall-clock
  budget of the plotted horizon". A missing ratio falls back to 1.0 (no stretch).
- **"Two options after 200 iterations"** (2h/2i): the assignments of the last
  two iterations before the merge point (198, 199). Under split-only min-sum
  these are the two period-2 oscillation branches (see
  `notebooks/06_split_oscillation_mgm_merge.ipynb`); if a run converged, both
  options coincide and the merge is a no-op.
- Engines run the full horizon (convergence does not stop them) so every curve
  has all 2000 iterations; anytime = running best of the per-iteration costs.
- Iteration counts follow the AIJ 2020 splitting experiments (50 instances,
  2000 iterations); K = 1000 still leaves 1000 post-split iterations.

## Ternary suite (arity-3)

A parallel suite of **true arity-3** benchmarks (`code/problems_ternary.py`),
run through the same machinery and written to `ternary_data/` + `ternary_plots/`
so the binary suite under `data/`/`plots/` is untouched. The question is *not*
binary-vs-ternary; it is whether the binary phenomena (damping helps, the 0.5
split helps, mid-run split timing, the MGM/optimal split merges) **reappear when
every constraint is genuinely ternary**. DABP (`Attentive`) is excluded — its
integration supports only unary/binary factors — so the suite runs the full
family minus the two DABP variants. Everything else (MS, DMS, all DMS/MS splits,
both merge variants, Optimal) is arity-generic and runs unchanged.

### Benchmarks (50 instances each, seeds 0-49)

| Name | Binary analog | Construction | Fit |
|---|---|---|---|
| `random_sparse_ternary` | `random_sparse` | arity-3 random factors, p3 = 2·0.1/(n−2) (== `build_random_ternary`) | **faithful** (exact expected-degree match) |
| `random_dense_ternary` | `random_dense` | same, dense degree p3 = 2·0.6/(n−2) | **faithful**, but **heaviest** (~490 arity-3 factors) |
| `meeting_scheduling_ternary` | `meeting_scheduling` | each agent is in **three** meetings → one ternary "no two of my meetings too close" constraint | **faithful** (the natural arity-3 generalization) |
| `graph_coloring_ternary` | `graph_coloring` | each factor = the **triangle of pairwise not-equals**, cost = 10·(#equal pairs), domain 3 | **constructed** — no canonical ternary coloring; this is the natural graded generalization |
| `scale_free_ternary` | `scale_free` | **preferential-attachment hypergraph**: each new agent forms one ternary hyperedge with two existing agents chosen ∝ hyperdegree (arity-3 Barabási–Albert) | **constructed** — no canonical scale-free hypergraph |

Densities reuse the expected-degree match already used by the binary→ternary
sparse benchmark: a variable's expected incident-triple count p3·C(n−1,2) is set
equal to the binary expected degree p1·(n−1), giving p3 = 2·p1/(n−2). The
tie-break unary preferences (U[0,1e-2)) carry over unchanged; their total mass
(≤0.5) still sits below the smallest structural cost gap (1 for random/meeting,
10 for coloring).

### What doesn't fit

- **DABP / `Attentive`** — unary/binary only; omitted from the whole suite.
- **Exact `Optimal` (branch and bound)** on the **domain-10** ternary benchmarks
  (`random_sparse_ternary`, `random_dense_ternary`, `scale_free_ternary`) —
  10^50 states, same as their binary analogs; omitted there.
- **Exact `Optimal` is also harder than in binary** on `graph_coloring_ternary`
  and `meeting_scheduling_ternary`: each ternary factor is a *triangle* of primal
  edges, so the primal graph is denser (higher induced width) than the binary
  version. B&B is attempted with the 300 s/instance cap but may solve **fewer
  instances** than binary (it solved none of a 2-seed coloring smoke at a 20 s
  cap); the Optimal reference line is drawn only over instances that completed,
  so a sparse or absent line there is expected, not a bug.
- `MS_split_opt_200` on `random_dense_ternary` is *not* exact (as in binary
  dense): the menu-conditioned subproblem is too wide, so the 300 s cap returns
  the best menu merge found (warm-started from MGM, so ≤ `MS_split_MGM_200`).

The two **constructed** families (`graph_coloring_ternary`, `scale_free_ternary`)
fit computationally but have no canonical ternary form, so `run_full_ternary.sh`
runs only the **three faithful analogs** (sparse, dense, meeting) by default. The
constructed builders stay available by name or via `--benchmarks all_ternary` if
you want them.

### Running

```bash
# the three faithful analogs (sparse, dense, meeting) x full family minus DABP
# x 50 problems x 2000 iters — hours; random_dense_ternary dominates the wall
# time (~490 arity-3 factors). The constructed coloring/scale-free benchmarks
# are excluded here (add them via the all_ternary form below).
bash experiments/aaai/code/run_full_ternary.sh

# smoke test (fast): one cheap benchmark, a few iterations
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks graph_coloring_ternary --algorithms all \
    --n-problems 5 --max-iter 200 --merge-at 100 --opt-time-limit 60 \
    --out-dir experiments/aaai/ternary_data

# any ternary benchmark by name, or the whole set via the all_ternary sentinel
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks all_ternary --algorithms MS DMS DMS_split_0.5 \
    --out-dir experiments/aaai/ternary_data

# analysis + plots point at the ternary dirs (no time_dabp.py: no DABP curve)
uv run python experiments/aaai/code/analyze_results.py --data-dir experiments/aaai/ternary_data
uv run python experiments/aaai/code/plot_results.py \
    --data-dir experiments/aaai/ternary_data --plots-dir experiments/aaai/ternary_plots
```

`--benchmarks all` is unchanged (the binary six); `all_ternary` expands to the
five ternary benchmarks. For a ternary benchmark, `--algorithms all` resolves to
the full family minus DABP, and naming a DABP label is reported as skipped.
Outputs mirror the binary suite (`{benchmark}_final_costs.csv`,
`_raw_costs.csv`, `_summary.csv`, `_significance.csv`, `_metadata.json`, and the
`{benchmark}_cost.pdf` / `_cost_zoom.pdf` plots), just under `ternary_data/` and
`ternary_plots/`.
