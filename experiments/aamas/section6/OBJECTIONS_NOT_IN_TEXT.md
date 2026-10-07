# Objections we can answer but did not put in the paper

Evidence we have for objections a reviewer might raise, left out of the main text for space or
readability. Each entry says the objection, the answer with its numbers, where the data is, and the
sentence that was removed from (or never entered) the text, so it can go into the supplementary material
or a rebuttal without new runs.

## 1. "The jump to the bounds after a delayed split comes from resetting the damping, not from the split"

**Why someone would say it.** Damping needs each variable's previous message. When DMS-$k$DS splits
mid-run, those stored messages are discarded, so the first messages after the split are sent undamped.
An undamped burst could be what pushes the messages to a bound, with the split itself doing nothing.
(Raised by the outside AI review of 2026-09-29; no human reviewer has raised it.)

**Answer.** A control run of plain DMS, no split, that discards the stored messages at iteration 1000
(paper units) behaves like DMS: 100 iterations later 0.08, 0.61, 0.18, 0.01 and 0.05 of its messages are
at a bound on random sparse, random dense, scale free, graph coloring and meeting scheduling (DMS itself
at that iteration: 0.09, 0.61, 0.18, 0.01, 0.05; the split at the same $k$ in the same lab run: 0.75, 0.94,
0.76, 0.68, 0.56, where the paper's own run gives 0.66 on graph coloring), its runs freeze on 16, 38, 22, 9
and 4 of the 50 instances against 49, 50, 50, 34 and 45 with the split (DMS: 19, 40, 22, 8, 3), and its
final cost is higher than that of DMS on all five benchmarks (lab-engine costs, not the paper's). So the
split, not the reset, causes the change.

**Data.** `experiments/aamas/splitting_explanation/exp1_reset_control.py`; results
`experiments/aamas/splitting_explanation/results/exp1_reset_<benchmark>.npz`; the table above is
`experiments/aamas/splitting_explanation/results/exp1_reset_summary.md` (run 2026-09-30 on the Mac,
50 instances per benchmark, resets at library iterations 50 and 500 = paper 100 and 1000; the lab
FastEngine, not the paper's engine, so its costs differ slightly from the paper data).

**Sentence removed from Section 6** (stop 6 of the 2026-10-06 rewrite, in `_v6_ors_sec6`):

> At the split the damping history is cleared, so the first messages after it are not damped. Clearing the
> history without a split does not cause this change: in a control run that clears it at $k = 1000$, 100
> iterations later 0.08, 0.61, 0.18, 0.01 and 0.05 of the messages are at a bound, as in DMS, and the final
> cost is not lower than that of DMS.

Plainer wording proposed the same day, if it is ever put back:

> When the split is performed, the messages stored for damping are discarded, so the first messages after
> the split are not damped. To check that discarding them is not what brings the messages to a bound, we ran
> DMS without a split and discarded the stored messages at iteration 1000: 100 iterations later 0.08, 0.61,
> 0.18, 0.01 and 0.05 of its messages are at a bound, as in DMS, and its final cost is not lower than that
> of DMS.

## 2. "Why not count a message as at a bound when only its two smallest entries come from one row?"

**Where it came from.** Roie's v7 (2026-10-07) reworded the definition of a message at a bound as "the
difference between its two minimal beliefs is equal to the difference between the costs of the corresponding
entries in the row of the constraint table". For binary variables that is the paper's definition; for larger
domains it only asks that the two smallest entries of the message be served by the same sender value, while
the paper's definition (and Figure 5) asks that one sender value serve every receiver value, i.e. that the
message be one row of the table up to a constant.

**What it gives.** Measured on the same runs (`FastEngine.commit2_mask`, saved as `sats2` next to `sats`),
mean over 50 instances at the end of the run, old measure -> new:

| benchmark | MS | DMS | MS-SCFG | DMS-SCFG | DMS-kDS (k = 1000) |
|---|---|---|---|---|---|
| random sparse | 0.07 -> 0.73 | 0.14 -> 0.76 | 0.77 -> 0.97 | 0.77 -> 0.97 | 0.78 -> 0.98 |
| random dense | 0.47 -> 0.91 | 0.81 -> 0.97 | 0.96 -> 1.00 | 0.95 -> 0.99 | 0.96 -> 1.00 |
| scale free | 0.11 -> 0.76 | 0.23 -> 0.81 | 0.76 -> 0.97 | 0.76 -> 0.97 | 0.78 -> 0.98 |
| graph coloring | 0.00 -> 1.00 | 0.04 -> 1.00 | 0.19 -> 1.00 | 0.83 -> 1.00 | 0.82 -> 1.00 |
| meeting scheduling | 0.01 -> 1.00 | 0.06 -> 0.94 | 0.07 -> 0.99 | 0.56 -> 0.98 | 0.58 -> 0.98 |

On graph coloring and meeting scheduling the looser measure is about 1 for every line from the first
iterations, split or not: a not-equal row has only two distinct values, so the two smallest entries of a
message always share a row; only the receiver value that clashes with the sender's best value is served by
another row, and whether the best row serves that value too is exactly the bound. On the random benchmarks
the contrast between lines shrinks from 0.07-0.77 to 0.73-0.97. So the looser measure does not separate
splitting from no splitting; the paper keeps the stricter one, whose numbers are the ones in the text.

**Data.** `experiments/aamas/splitting_explanation/results/exp1_<benchmark>.npz` and
`exp1_delayed_<benchmark>.npz` (rerun 2026-10-07 on the Mac with both measures; the old `sats` and costs
reproduced exactly, backup of the previous files in `results/backup_before_sats2_20261007/`); figure
`section6/out/mechanism_fraction_at_bound_sats2.pdf` (`mechanism_figure.py --measure sats2`).
