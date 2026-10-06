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
