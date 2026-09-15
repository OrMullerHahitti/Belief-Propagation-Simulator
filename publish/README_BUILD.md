# Build guide — publish/

## Authoritative build (for submission)

`main_revised.tex` is the build entry point. It `\input`s, in order:

```
sec0_abstract            abstract + AAAI links block
sec1_introduction
sec2_related_work        (+ new closing paragraph: parallel-symmetric period-2 lineage)
sec3_background          (two-phase schedule now authoritative: R^i uses Q^i)
sec4_effect_of_splitting (+ indexing note; math untouched)
sec5_splitting_no_damping (BCT speculation replaced by bridge to the theorems)
sec5b_two_solutions      NEW: two-solution theory + damping analysis (main text)
sec6_experiments         (Measuring-the-Mechanism filled with real measurements;
                          Limitations updated to conditional scope)
sec7_appendix            \appendix + benchmark plot figures
secA_oscillation_appendix NEW: deferred proofs, protocol, census, damping audit
```

Bibliography: `\bibliography{refs,DisCSP_refs,DampIns_AIJ,oscillation_refs}` —
`oscillation_refs.bib` (in this folder) holds the three new entries
(GolesOlivos80, PoljakSura83, AshkenaziGolan25).

## What you must drop in before the real compile

1. **AAAI-27 author kit**: `aaai2027.sty`, `aaai2027.bst`,
   `ReproducibilityChecklist.tex` (all answers still need filling).
2. **Bib files**: `refs.bib`, `DisCSP_refs.bib`, `DampIns_AIJ.bib`.
3. **Figures**: all 18 referenced graphics currently exist as *visibly marked
   placeholders* (dashed frame, "PLACEHOLDER — replace with <name>"), so the
   build compiles today; overwrite each with the real figure, same filename:
   `examp.pdf`, `two_cycle_examp.pdf`, and `plots/{graph_coloring,
   meeting_scheduling, random_sparse, random_dense, scale_free}_cost[_zoom].pdf`
   plus the three `*_ternary_cost[_zoom].pdf` pairs.

## Pre-submission checklist

- [ ] Replace all placeholder figures (search the PDF for "PLACEHOLDER").
- [ ] Remove the `\todofill` macro definition from `main_revised.tex`
      (no usages remain in the text).
- [ ] Replace the anonymized links placeholder URL in `sec0_abstract.tex`.
- [ ] Fill `ReproducibilityChecklist.tex`.
- [ ] Check page budget; if over, first cut the census table in
      `secA_oscillation_appendix.tex` (keep pooled numbers in prose), then fold
      Lemma `lem:decode` into Lemma `lem:sibling` in `sec5b_two_solutions.tex`.
- [ ] Clear PDF metadata (`exiftool -all:all= -overwrite_original main.pdf`).

## Other files in this folder

- `full_paper_merged.tex` — single-file snapshot of the whole paper (real AAAI
  preamble). Convenience copy for Overleaf upload; regenerate after editing
  the sec files, or ignore and use the split build.
- `full_paper_preview.pdf` — readable preview of the split build compiled with
  a substitute article style (AAAI kit not in repo), placeholder figures, and
  only the new references resolved. Not the submission layout; content only.
