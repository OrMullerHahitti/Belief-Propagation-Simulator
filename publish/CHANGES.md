# Revision Changelog — main_revised.tex + sec*.tex

The paper is split one file per section; `main_revised.tex` inputs them in order.
Compile requirements unchanged (aaai2027 kit, `examp.pdf`, `two_cycle_examp.pdf`,
`plots/*.pdf`, the three `.bib` files, `ReproducibilityChecklist.tex`).

## Data pulled from `experiments/aaai` (all n=50, verified against CSVs)

- **DABP-SymSplit column added to Table 1** (from `data_cuda/*_final_costs.csv`;
  the Attentive/NoSplit means there match the existing table run-for-run):
  graph coloring 35.02, meeting scheduling 30.15, random dense 99665.79,
  random sparse 14631.11, scale-free 16771.71. Stds in
  `dabp_symsplit_summary_rows.csv` (this folder) — merge into your
  `data/*_summary.csv` or rerun `analyze_results.py` including the CUDA runs.
- **λ = 0.9** stated in §6.1 (from `run_experiments.py: DAMPING = 0.9`).
- **Benchmark generation specs** written into §6.1 from `experiments/aaai/README.md`
  (50 agents/domain 10/p=0.1/0.6/U[100,200), 3-color GC cost 10, BA scale-free,
  90-agent meeting scheduling, tie-breaking unaries U[0,0.01)).
- **Significance paragraph** in §6.3: Wilcoxon/paired-t p-values quoted from
  `data/*_significance.csv` (final_cost metric). DABP-vs-SymSplit p-values
  computed from `data_cuda` final costs (paired, scipy): unstructured families
  p<1e-8; GC p=0.44; MS p=0.23.
- **Merge procedure corrected** (§5, §6.2): branches taken at iterations 198/199
  (`run_experiments.py: branch_iters = (merge_at-2, merge_at-1)`), MGM run from
  both branches best-kept, opt merge exact except random dense (300 s cap,
  warm-started). Optimal B&B: 60 s cap, completed on 21/50 GC instances
  (legend `Opt n=21` now defined).
- **Legend variants defined** in §6.2: `DMS s=.5`, `DMS s=.4-.6` (random split
  U[0.4,0.6)), `MS s=.5`, `MGM inv@200`, `Opt n=k`.

## Review-driven edits (see the referee report / roadmap)

- Abstract + intro rewritten in present tense; scoped claims (structured-benchmark
  advantage explicit; DABP best on unstructured families acknowledged in §6.3
  and Limitations). Fixed double "Nevertheless", "guaranty"→"guarantee",
  "is dramatically"→"is dramatic", the "attentive version of the attentive
  algorithm" duplication, and the DABP run-on in §2.
- §4: added "Scope of the analysis" paragraph; harmonized the running assumption
  to `M_a < M_b < B_b < B_a` (≪ kept as intuition only) in the three places it
  appeared; footnote period; grammar fix after Lemma 4.2; labels added to
  Observation 4.1 / Lemma 4.2.
- **NEW §4.5 "Toward Damped and Asymmetric Splits"** — two propositions:
  - Prop. (Geometric settling under damping): conditional; two-line proof via
    exponential smoothing of the locked regime constant.
  - Prop. (Asymmetric round-trip constants): forward table C′, backward C″;
    reduces to Thm 4.6 at C″=C′; absorption condition c_U(C′,C″) ≥ τ_U(C′).
  - Remark (Split-ratio design) ties this to DABP's 0.95/0.05.
  **VERIFY both and add to the Lean development before submission** (they are
  direct substitutions into Thms 4.5/4.6, but they are new claims).
- §5: oscillation finding stated as a numbered Observation; extraction of the
  two branches now matches the code; honesty clause vs DMS-SCFG added.
- §6: removed the in-text TODO sentence (SymSplit now in Table 1); per-row best
  bolded; added Statistical significance paragraph; SymSplit subsection rewritten
  around the measured result (0.95/0.05 significantly better only on unstructured
  families; statistically indistinguishable on structured ones — note GC mean
  actually favors SymSplit); added **NEW §"Measuring the Mechanism"** (instrumentation
  experiment design — RUN IT and replace the \todofill); added Limitations paragraph;
  delayed-splitting claims calibrated to significance.
- Background: footnote moved out of the subsection title; `\label{BoothB19}` →
  `\label{sec:bct}`; "Min sum"→"Min-sum".

## Still on the authors before submission (cannot be done from here)

1. Fill `ReproducibilityChecklist.tex` (every answer is still a placeholder).
2. Populate the anonymized repo (code + Lean) and replace the placeholder URL.
3. Run the §"Measuring the Mechanism" instrumentation and replace the \todofill.
4. Verify Props 4.x (damped + asymmetric) and the §2 claim that DABP's
   centralization stems from network training/application — marked with
   `% AUTHORS:` comments.
5. Regenerate plots if you want SymSplit stds in the summary CSVs
   (`analyze_results.py` over the CUDA final-costs), and consider pruning
   overloaded legends (16 entries on GC) — move minor variants to the appendix.
6. Strip every `\todofill` (grep) and clear PDF metadata (exiftool) last.

## Note on the earlier referee report

One minor issue in the review — "infinitely many iterations" in §4.3 — was an
artifact of PDF text extraction; the source correctly reads "in finitely many
iterations". No change was needed.
