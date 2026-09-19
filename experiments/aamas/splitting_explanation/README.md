# Why factor splitting makes min-sum converge fast (and where it ends up)

Two documents and the experiments behind them.

* `EXPLANATION_formal.md` — the exact reduction theorem (symmetric split = unsplit min-sum with
  `Q = 2 cavity + own R`), the commitment/echo mechanism, the two-phase convergence argument, the
  large-scale picture (synchronous best response = alternating minimisation on the bipartite double
  cover), and why dense graphs end in two bad solutions. Every claim is tagged proved / quoted from the
  paper / measured.
* `EXPLANATION_hebrew.md` — the same story in plain Hebrew, walked through on two tiny examples.

## Experiments

All scripts use `lab.py` (a vectorised synchronous min-sum engine with split / damping / exact-PropFlow
modes, instance builders for the five AAAI benchmarks, and the analysis helpers) and `plotting.py`
(labels, styles, PDF output into `plots/`). Run from the repository root with `uv run python`.

| script | what it measures | outputs |
|---|---|---|
| `exp0_checks.py` | (a) the reduction identity, (b) agreement with PropFlow's engines, (c) iteration-for-iteration reproduction of the recorded AAAI cost curves, (d) the integer-truncation artifact | `results/exp0_checks.txt`, `results/exp0_truncation_*.csv` |
| `exp1_speed.py` | freeze time, commitment, cost and message change for MS / DMS / MS+split / DMS+split / DMS(0.5)+split on the five benchmarks, 50 seeds | `results/exp1_*.npz`, `results/exp1_summary.md`, `plots/exp1_<bench>_{example,aggregate,message_change}.pdf` |
| `exp2_ingredients.py` | doubling only vs echo only vs both, on the unsplit graph | `results/exp2_*.npz`, `results/exp2_summary.md`, `plots/exp2_<bench>.pdf` |
| `exp3_two_solutions.py` | the period-2 end state vs density: edge classes, best-response steps, cost_2 monotonicity, bipartite re-phasing | `results/exp3_{random,bipartite}.csv`, `results/exp3_summary.md`, `plots/exp3_density.pdf` |
| `exp4_quality.py` | what the split freezes on: 1-opt / edge / path neighbourhoods, greedy repair, split at iteration K | `results/exp4.csv`, `results/exp4_summary.md`, `plots/exp4_split_at_k.pdf` |
| `exp5_damping.py` | damping sweep with and without the split | `results/exp5.csv`, `results/exp5_summary.md`, `plots/exp5_damping.pdf` |
| `exp6_split_ratio.py` | asymmetric split p / (1-p) | `results/exp6.csv`, `results/exp6_summary.md`, `plots/exp6_split_ratio.pdf` |
| `examples_tiny.py` | the two worked examples (one edge; a triangle) | `results/examples_tiny.txt` |

Each `expN` script runs its experiment with `multiprocessing.Pool`, writes the summary, and plots;
`--plot-only` re-plots from the saved results. exp1 and exp4 take the longest (50 seeds × 2000
iterations × several engines); the whole set is a few hours on a laptop. The repository ignores
`results/` by default; the summaries, CSV and text results are committed anyway (they are small), the
per-iteration `.npz` arrays of exp1 and exp2 (10 MB) are not, so `--plot-only` for those two needs a
rerun first.

Frozen: a run's freeze time is the iteration after its last assignment change, and a run counts as
frozen only if at least 100 unchanged iterations follow (`lab.freeze_time`); a run that was still
changing a few iterations before the end is reported as not frozen.

Semantics: the mechanism experiments run in float arithmetic. The recorded AAAI runs truncated Q
messages to integers on the integer-table benchmarks (PropFlow's `compute_R` casts to the table dtype);
`exp0_checks.py` (c)/(d) reproduces that mode and measures that it does not bias the results.
