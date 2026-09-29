# Section 6 handoff (2026-09-28)

Written on Or's Mac for the session on `rtx` (`~/projects/belief-propagation-simulator`). Everything
below was checked by running the scripts named here; nothing is a guess.

## What the job is

Rewrite Section 6 (Experimental Evaluation) of the AAMAS-2027 paper "Split and You Shall Converge".
The file to edit is `publication/min-sum_split_AAMAS-2027_ors_revision_3_section6.tex` (Overleaf
git clone; Section 6 starts at `\section{Experimental Evaluation}`, line 636, Conclusions at 797,
experimental appendix at 1086). It is a byte copy of revision 3, which merged Shir's Section 4 with
revision 2's Section 5 (Roie's Sept 27 rewrite). Do not touch revision 2, revision 3 or Shir's file.

Overleaf: the clone's `origin` is `https://git.overleaf.com/69caa4706dec02eb84b1e287`. The token
lives in the Mac keychain, not in the clone, so a push from rtx asks for credentials. Either Or
enters the token there once, or the edited file goes back to the Mac to be pushed.

## Decisions Or made (2026-09-28)

1. Section 6 tells the story in this order:
   1. the mechanism of Section 4 in large graphs: the fraction of function-to-variable messages
      "at a bound". Measured as exp1's committed arc: one sender value minimizes for every
      receiver value. In the binary case this is exactly a message difference at a bound of
      Section 4; say so in one sentence.
   2. with damping, splitting makes the run converge in hundreds of iterations instead of
      thousands. This is a SPEED claim, not a quality claim (see the numbers).
   3. the split as a mechanism applied mid-run: DMS for k iterations, then a 0.5 split, damping
      kept. Present the WHOLE k sweep on all 50 instances (no selected k, no held-out selection).
      "Later is better because the start is better" is true on the random families only.
   4. the MS-SCFG-MGM / MS-SCFG-opt merge lines stay as a fourth item (they are the only evidence
      that Section 5's two-runs observation holds on large graphs).
   DABP: keep the curves as reference lines, one paragraph, no claims about why it works.
2. No anytime / best-seen mechanism anywhere. The best-checkpoint split of `experiments/aamas/late_split`
   is not used.
3. Statistics: paired Wilcoxon only.
4. Iteration units: the paper counts two iterations per library iteration (4000 paper = 2000
   library; k = 2 × the CSV label).
5. Mechanism figure: one row of five panels, linear x axis, color. k sweep: both a table and a
   plot were drafted; Or chooses later.
6. Section 5 will be fixed AFTER Section 6: Roie's "Inter-DMS" paragraph must shrink to the two
   orders that were run (DMS then DMS-split; MS-split then MGM/SyncBB merge). Nothing else in
   Sections 4–5 constrains Section 6.
7. Commits under Or's name only, no co-author line, Conventional Commits.

## Data map

- `experiments/aaai/data_paper_20260928/` — the folder Section 6 quotes. Built by
  `experiments/aaai/code/build_paper_data.py --out-dir ... --write` from `data/` plus the float-table
  reruns; then `settling.py`, `key_comparisons.py`, `analyze_results.py` ran on it (logs in
  `experiments/aaai/logs/paper_data_*_20260928.log`). Files per benchmark: `_final_costs.csv`
  (`final_cost`, `anytime_cost` — do not use anytime), `_raw_costs.csv`, `_summary.csv`,
  `_significance.csv` (t-test and Wilcoxon), `_settling.csv` (settled = cost constant over the last
  100 library iterations), `_heldout_k.csv`, `_key_comparisons.csv`, `_metadata.json`.
- `experiments/aaai/data_float_tables_20260923/` — random sparse and dense rerun with float cost
  tables: MS, DMS and DMS_split_at_{50,100,300,500,1000} (dense also 1500). Integer tables
  truncate messages on unsplit graphs, which is why these lines were rerun. Scale-free, graph
  coloring and meeting scheduling use float tables by construction.
- `experiments/aaai/data_paper_20260923/` — the previous paper folder (k ∈ {300, 1000} only on
  sparse/dense). Kept for reference; the Section 6 prompt-builder numbers came from it.
- `experiments/aaai/data/` — the raw post-fix runs, integer tables on sparse/dense.
- `experiments/aamas/splitting_explanation/` — exp1 (`exp1_speed.py`, results
  `results/exp1_<bench>.npz`, `results/exp1_summary.md`): MS, DMS, MS_split, DMS_split,
  DMS05_split; per iteration `costs`, `sats` (fraction at a bound), `changes`, `dq`; per run
  `freeze`, `period`, `final`, `best`, `t95`. New: `exp1_delayed.py` → `results/exp1_delayed_<bench>.npz`
  with the same records for `DMS_split_at_{50,100,300,500,1000,1500}`, and
  `results/exp1_delayed_summary.md`. exp1 runs the lab's FastEngine (float, checked against
  PropFlow in `exp0_checks.py`). Its costs differ from the paper data by up to 0.1%, so quote costs
  from the paper data only and use exp1 only for the fraction at a bound and the freeze times.
- `experiments/aamas/section6/` — this folder: `mechanism_figure.py` (→ `out/mechanism_fraction_at_bound.pdf`,
  `--k` picks the DMS-kDS line, default 500 library = 1000 paper), `delayed_split_sweep.py`
  (→ `out/delayed_split_sweep.{csv,md,tex,pdf}`).
- `experiments/aaai/code/plot_final.py` — draws the paper's cost figures from a data folder
  (last used on `data_paper_20260923`, output `experiments/aaai/final_plots_paper_20260923/`);
  rerun it on `data_paper_20260928` once Or picks the lines.
- `experiments/aamas/runs/` (2.8 GB) was NOT copied to rtx; nothing in Section 6 needs it.

## Verified numbers (all 50 instances unless said)

Fraction of messages at a bound at the end (exp1): MS .07/.47/.11/.00/.01, DMS .14/.81/.23/.04/.06,
MS-SCFG .77/.96/.76/.19/.07, DMS-SCFG .77/.95/.76/.83/.56 (sparse/dense/scale-free/coloring/meeting).
The delayed split reaches the DMS-SCFG level from any k (e.g. k=500: .78/.96/.78/.82/.58) and the
runs freeze 16–45 library iterations after the split on the random families, 55–145 on the
structured ones.

Settled runs of 50 (cost constant over the last 100 library iterations) and median settling
iteration, library units: DMS 17/33/22/8/2 (median ~500–1400); DMS-SCFG 49/50/49/40/44 (70/60/43/129/154).

Quality, DMS-SCFG vs DMS (final cost): sparse p 6.5e-5 (split better); dense p 0.054 with the split's
MEAN lower but the split worse on 37 of 50 instances; scale-free p 0.25. Hence item 2 is a speed claim.

Delayed split vs DMS-SCFG, mean final cost, k in paper iterations, * = Wilcoxon p < 0.01:
sparse −0.32*/−0.36*/−0.32*/−0.42*/−0.41* % at k = 100/200/600/1000/2000;
dense −0.22*/−0.31*/−0.32*/−0.33*/−0.36*/−0.40* % at 100/200/600/1000/2000/3000 (monotone);
scale-free −0.39*/−0.27/−0.59*/−0.65*/−0.75* % (monotone from k = 200);
coloring +3.5/−10/−11/−4/−8 %, p ≥ 0.14; meeting −1.5/−3.5/−0.5/−1.2/−0.7 %, p ≥ 0.03.
Held-out selection (k chosen on 25 instances, tested on the other 25): dense k=3000 p 3e-6,
scale-free k=1000 p 0.002, sparse k=1000 p 0.28, structured not significant.

Starting-point check (scratch script on the Mac; the delayed-split run equals the DMS run before
the split, max difference 0): on the random families the final cost after the split tracks the DMS
cost at the split moment (per-instance Spearman 0.85–0.92 sparse/dense, 0.50–0.77 scale-free); on
coloring and meeting it does not (Spearman ≤ 0.38; coloring finals 30.6–35.6 in no k order,
meeting 7.8–8.1). In the 200 library iterations after the split the cost drops beyond DMS's own
drop by 540–840 (sparse, scale-free), 970–2050 (dense), 80–110 (coloring), 13–15 (meeting).
Per instance the split does not always end below its own pre-split cost (dense k=2000: 23 of 50).

Merge lines (MS-SCFG-MGM/opt, from the 09-23 analysis, unchanged data): the selection closes
82.5–95.4% of the gap between MS-SCFG and DMS-SCFG; DMS-SCFG still beats MS-SCFG-opt on every
benchmark (max p 1.6e-5). B&B cap 300 s hit on 48/50 dense, 2/50 scale-free instances.

DABP: the recorded runs on dense, scale-free and meeting restart messages at library iteration
1000 while the NoSplit runs do not — do not claim the two differ only by the split.

## Old Section 6 text to remove or replace

SyncBnB → SyncBB; `MGM@200` / `Opt merge@200` labels (paper units 400); "DMS-$K$DS" and
"K ∈ {600, 2000} to avoid graph density"; the best-K sentence; "paired t-tests"; "simulated
runtime" for DABP; "do not converge but rather perform oscillation"; "best among distributed";
"MS-split" (write MS-SCFG). The current three "novel insights" of the Discussion are replaced by
the four items above.

## Next steps

1. Or picks: sweep as table or plot; which lines the cost figures show (suggested: MS, MS-SCFG,
   MS-SCFG-opt, DMS, DMS-SCFG, DMS-kDS at one k, DABP as reference); one common k for the
   mechanism figure (drafted with k = 1000 paper).
2. Regenerate the cost figures from `data_paper_20260928` with `plot_final.py`; put PDFs where
   the tex expects them (`sparse_2.pdf` etc. in the Overleaf clone root, or update the paths).
3. Write the setup paragraph with ONE definition of "converged" and ONE of "at a bound", then
   the four items, then the Discussion. Show the text to Or before it goes into Overleaf.
4. Fix Section 5's Inter-DMS paragraph and the Conclusions to match.
5. Commit the new scripts and data under Or's name (no co-author), small commits.

## Update, 2026-09-28 evening (written on the Mac)

- **Meeting scheduling rerun.** All its lines except DABP and Optimal were rerun with the current code
  (`experiments/aaai/data_meeting_rerun_20260928`, 4.2 min on rtx) because the June 28 lines could not
  be reproduced (no tie-break fractions in their costs, trajectories differ). `data_paper_20260928` was
  rebuilt with them (old build kept as `data_paper_20260928_pre_meeting_rerun`); settling, held-out k,
  key comparisons and analyze_results were rerun on it. DABP on random dense, scale free and meeting
  scheduling are still June runs (see FOLLOWUPS.md).
- **Split at the best point** (`run_split_at_best.py`, results in `split_at_best/`): a demonstration line,
  split at DMS's best iteration + 1, damping kept, 1000 library iterations after the split. Random
  families: never higher than the DMS best, kept on 34/41/39 of 50. Coloring 33.2 -> 27.2 (23 lower,
  15 higher), meeting 8.9 -> 8.0 (33 lower, 6 higher).
- **Section 6 draft written:** `SECTION6_draft.tex` (replaces the section and the second Conclusions
  paragraph), compiled inside the working copy with 0 errors and 0 undefined references
  (`out/SECTION6_draft_preview.pdf`, `out/SECTION6_draft_full_paper.tex`); Conclusions now start on
  page 9 (one page later than revision 3). Figures under the Overleaf names in `overleaf_upload/`.
  Every number is in `out/numbers.md`. Follow-ups for Sections 5, Conclusions and the appendix in
  `FOLLOWUPS.md`. The split at the best point is the line DMS-BDS in the cost figures. Or's final choice (2026-09-29): split
  each instance at its OWN best DMS iteration (t* + 1), keep damping, run to library iteration 2000 like
  every other line, and average (`run_split_at_best.py --horizon 2000` -> `split_at_best_fixed_horizon/`,
  drawn by `plot_final.py --section6 --extra-raw experiments/aamas/section6/split_at_best_fixed_horizon`).
  The averaged line leaves DMS gradually. `split_at_best/` (t*+1, 1000 after) and `split_at_best_window/`
  (best of the first 1000, restored and split at 1000; `best_point_curve.py`) are the rejected variants,
  kept for reference. With 7 non-DABP lines the zoom figures are regenerated too.
- Nothing in `publication/` edited, nothing pushed, nothing committed.
