# AAAI paper experiments

Five DCOP benchmarks x nine algorithm families, 50 problem instances each,
reporting mean cost and paired statistical significance. Layout follows
`experiments/aij/`: scripts in `code/`, CSVs in `data/`, PDFs in `plots/`.

## Benchmarks (50 instances each, seeds 0-49)

| Name | Spec |
|---|---|
| `random_sparse` | 50 agents, domain 10, p1 = 0.1, integer costs U[100, 200) |
| `random_dense` | 50 agents, domain 10, p1 = 0.6, integer costs U[100, 200) |
| `graph_coloring` | 50 agents, 3 colors, p1 = 0.05, not-equal constraints: equal = 10, else 0 |
| `scale_free` | Barabasi-Albert per Cohen, Galiki & Zivan (AIJ 2020) §6: 7 initial agents randomly connected, each new agent attaches preferentially to 3 existing agents, n = 50, domain 10, integer costs U[100, 200) (the §6.2 cost range used for the splitting experiments; §6.1 used U[0, 100)) |
| `meeting_scheduling` | Cohen et al. (AIJ 2020) §6: 90 agents schedule 20 meetings into 20 time slots; each agent participates in two random meetings; per constrained meeting pair travel time ~ U{6..10}; cost = number of overbooked (shared) agents when the slot difference is below the travel time |

All problems get tiny random unary value preferences (U[0, 1e-6)) for tie
breaking, as in all Max-sum versions of the AIJ paper (Farinelli et al. 2008
style). Too small to change which assignment is optimal; without them min-sum
is degenerate on the symmetric problems (graph coloring especially).

Random/coloring topologies use `FGBuilder.build_random_graph` (Erdos-Renyi,
components force-connected — engines require a connected graph).

## Algorithms

| Label | Item | Description |
|---|---|---|
| `DMS` | 2a | `DampingEngine`, lambda = 0.9 |
| `DMS_split_0.5` | 2b | `DampingSCFGEngine`, constant symmetric split (0.5/0.5) |
| `DMS_split_0.4_0.6` | 2c | DMS on a *random* SCFG: each cost-table entry c is split into u·c / (1-u)·c with u ~ U[0.4, 0.6) — the "0.4-0.6" version of the AIJ paper (§6.2), which it found best |
| `DMS_split_at_{50,100,300,500,1000}` | 2d | `MidRunSplitEngine` + damping; all factors split at iteration K, `transfer` mode (prior R messages redistributed p/(1-p) across clones, as in the late-split experiments in `experiments/other/non_convergence_chain`) |
| `Attentive` | 2e | min-sum where each variable discounts incoming messages by 1/degree every iteration (the repo's `discount_attentive` policy). **Interpretation flag** — see notes |
| `Optimal` | 2f | depth-first branch and bound on the original tables, 60 s/instance limit; reported only for instances where the search completed (expect graph_coloring and meeting_scheduling; the domain-10 benchmarks generally won't finish) |
| `MS_split_0.5` | 2g | undamped min-sum on a 0.5/0.5 SCFG |
| `MS_split_MGM_200` | 2h | the two assignments at iterations 198/199 of the `MS_split_0.5` run (its period-2 oscillation branches = "the two options after 200 iterations") merged with MGM-1 restricted to the per-variable binary menu {b1[v], b2[v]}; run from both seeds, best kept |
| `MS_split_opt_200` | 2i | same two options merged *optimally*: branch and bound over the binary menus, on tables conditioned on the menus (300 s cap). Exact on every benchmark except `random_dense`, where the conditioned subproblem has induced width ~25-36 (30-46 disagreeing variables on a dense graph) and proving optimality is infeasible — there the reported value is the best menu merge found within the cap, warm-started from the MGM result (so always <= `MS_split_MGM_200`); the run log counts these instances |

`MS_split_0.5`, `MS_split_MGM_200` and `MS_split_opt_200` share one engine run
per instance, so all three see exactly the same oscillation branches.

## Running

```bash
# full run (50 problems x all algorithms x 2000 iterations) — hours
uv run python experiments/aaai/code/run_experiments.py --benchmarks all

# one benchmark / subset of algorithms / smoke test
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks graph_coloring --n-problems 5 --max-iter 300 --merge-at 200

# means + paired t-tests + Wilcoxon signed-rank (per benchmark)
uv run python experiments/aaai/code/analyze_results.py

# mean per-iteration and anytime cost curves (PDF)
uv run python experiments/aaai/code/plot_results.py
```

## Outputs

- `data/{benchmark}_final_costs.csv` — algorithm, seed, final_cost, anytime_cost
- `data/{benchmark}_raw_costs.csv` — algorithm, seed, iteration, cost (cost is always evaluated on the **original** cost tables, also after splitting)
- `data/{benchmark}_metadata.json` — run parameters
- `data/{benchmark}_summary.csv` — per-algorithm mean/std (final + anytime)
- `data/{benchmark}_significance.csv` — pairwise paired t-test + Wilcoxon p-values on both metrics, computed over the common solved seeds (AIJ 2020 reported paired t-tests on final and anytime results)
- `plots/{benchmark}_cost.pdf`, `plots/{benchmark}_anytime.pdf`

## Notes / interpretation decisions

- **Attentive**: implemented with the existing `discount_attentive` policy
  (degree-inverse message discounting), the only "attentive" notion in this
  repo. If the intended algorithm is something else (e.g. Deep Attentive BP,
  NeurIPS 2022), swap the engine in `code/run_experiments.py::make_engine`.
- **"Two options after 200 iterations"** (2h/2i): the assignments of the last
  two iterations before the merge point (198, 199). Under split-only min-sum
  these are the two period-2 oscillation branches (see
  `notebooks/06_split_oscillation_mgm_merge.ipynb`); if a run converged, both
  options coincide and the merge is a no-op.
- Engines run the full horizon (convergence does not stop them) so every curve
  has all 2000 iterations; anytime = running best of the per-iteration costs.
- Iteration counts follow the AIJ 2020 splitting experiments (50 instances,
  2000 iterations); K = 1000 still leaves 1000 post-split iterations.
