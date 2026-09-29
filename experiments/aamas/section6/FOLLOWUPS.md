# Section 6 follow-ups (2026-09-28)

Things the Section 6 draft (`SECTION6_draft.tex`) needs elsewhere in the paper, and data facts Or must decide on.

## Data facts to decide on

1. **DABP lines on random dense, scale free and meeting scheduling are old runs.** Their costs have no
   tie-break fractions, so they come from the June generator (the same problem that forced the meeting
   rerun). The other DABP lines (random sparse, DABP-NoSplit everywhere) are current. Either rerun DABP
   on those three benchmarks (needs the `dabp` extra: torch + torch-geometric, GPU on rtx) or drop the
   two DABP sentences marked `% CHECK` in the draft. The earlier note that these three runs restart
   their messages at iteration 2000 (library 1000) is the same old-run artifact.
2. **Meeting scheduling was rerun** (all lines except DABP and Optimal) into
   `experiments/aaai/data_meeting_rerun_20260928` and merged into `data_paper_20260928`. The old
   folder is kept as `data_paper_20260928_pre_meeting_rerun`. The meeting numbers changed slightly
   (DMS 20.7 -> 21.2, DMS-SCFG 8.1 -> 8.1, delayed split now +0.1..+1.6% vs DMS-SCFG, none significant).
3. **Section 5's damping paragraph** quotes "232 of the 250 runs" (DMS-SCFG cost constant over the last
   200 iterations, against 7 of 250 for MS-SCFG). With the current data the counts are 49+50+49+40+43 =
   231 of 250 for DMS-SCFG and 0+0+0+1+0 = 1 of 250 for MS-SCFG. Roie commented that sentence out; if it
   comes back, use 231 and 1.

## Text elsewhere in the paper

4. **Section 5, Roie's Inter-DMS paragraph.** Only two orders were run: DMS then DMS-split (DMS-kDS),
   and MS-split then MGM/SyncBB (the merge lines). The paragraph must not promise other interleavings,
   the anytime mechanism sentence must go, and "DMS-KDS" -> "DMS-$k$DS".
5. **Section 5 line "The results presented in Section 6 demonstrate which of the versions was the most
   successful"** (Roie's wording): the draft supports "delaying the split lowers the mean cost of
   DMS-SCFG on the three random benchmarks, significantly, and does not change it on the structured
   ones". Or's earlier sentence ("on all five benchmarks, significantly on the three random ones") is no
   longer true for meeting scheduling after the rerun.
6. **Conclusions, second paragraph:** replacement text is at the end of `SECTION6_draft.tex`.
7. **Appendix, Additional Experimental Results:** the zoom figures `*_zoom_2.pdf` were regenerated with the
   Section 6 lines (in `overleaf_upload/`, same names); their captions must now say "DMS-SCFG, DMS-kDS,
   DMS-BDS and MS-SCFG-opt". The ternary figures are untouched (ternary data was not
   rerun; it is from the post-fix rerun of 2026-09-15).
8. **Names:** the draft writes MS-SCFG, MS-SCFG-MGM, MS-SCFG-opt, DMS-SCFG, DMS-$k$DS, DMS-BDS (split at the best point), DABP, DABP-NoSplit.
   Section 6 of the old text used "MS-split", "SyncBnB", "DMS-$K$DS", `MGM@200`, `Opt merge@200`; the
   legends of the new figures use MS-SCFG, MS-SCFG-opt, DMS-SCFG, "DMS-kDS, k=1000", DABP.
9. **Page budget:** the draft is about the same length as the old Section 6 but adds one full-width figure
   (five small panels) and one small table. If it overflows, the first cut is the DMS-SCFG-rand sentence,
   the second is the "split at the best point" paragraph.

## Not done

- Nothing in `publication/` was edited; nothing was pushed to Overleaf.
- The new scripts and data are uncommitted on both machines (`experiments/aamas/section6/`,
  `experiments/aaai/data_meeting_rerun_20260928`, `data_paper_20260928`, `final_plots_paper_20260928`,
  the appended k grid in `data_float_tables_20260923`, `experiments/aamas/splitting_explanation/exp1_delayed.py`
  and its results, `build_paper_data.py`, `plot_final.py --section6`).
