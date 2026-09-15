# Where to Apply Each Change in Overleaf

Anchors refer to your current single-file main.tex. Apply Part 1 yourself now;
Part 2 is other people's text — use the sec*.tex files as proposed diffs for review.

## Part 1 — Yours, paste now (§6 + §7 + one preamble line)

1. **Preamble** — right after `\usepackage{xcolor}` add:
   ```latex
   \newcommand{\todofill}[1]{\textcolor{red}{\textbf{[TO FILL: #1]}}}
   ```
   (needed by the new §6; strip before submission).

2. **Experiments** — select from the line `\section{Experimental Evaluation}`
   down to the line just before `\appendix`, and replace with the full contents
   of `sec6_experiments.tex`.

3. **Appendix** — select from `\appendix` down to the line just before
   `\bibliography{refs,DisCSP_refs,DampIns_AIJ}`, and replace with
   `sec7_appendix.tex`.

**Compile fix until §4.5 is approved** (new §6 cites two propositions that live
in the proposed §4.5; until co-authors merge it, make these 3 micro-edits in
your new §6, then revert when §4.5 lands):
- In subsection "The Symmetric-Split DABP Ablation", delete the sentence
  `Proposition~\ref{prop:asym} gives the corresponding formal handle: ... absorption condition.`
- In the Limitations paragraph, change
  `with two-value domains; undamped updates, except for the conditional Proposition~\ref{prop:damped}; and influence`
  to `with two-value domains, undamped updates, and influence`.
- In the Limitations paragraph, delete the sentence
  `Proposition~\ref{prop:asym} extends the absorption analysis ... only for the symmetric case.`

## Part 2 — Co-authors' sections (propose these as Overleaf comments/diffs)

**Abstract** (`sec0_abstract.tex`)
- Replace everything between `\begin{abstract}` and `\end{abstract}`.
- What changed: double "Nevertheless" fixed; "we will present/prove" → present
  tense; last sentence scoped (structured-benchmark claim made explicit).

**Introduction** (`sec1_introduction.tex`)
- One-word fix, anchor `there is no such guaranty` → `guarantee`.
- Replace the last three paragraphs, from the anchor
  `In this study, we investigate the success of function-node splitting`
  to the end of the section (present tense, cleaned grammar, damping-role
  sentence fixed).

**Related Work** (`sec2_related_work.tex`)
- Anchor `converges fast to a solution with a much higher quality solution` →
  `converges quickly to a solution of much higher quality`; next sentence
  `unexplainable` → `unexplained`.
- Replace the whole paragraph starting `\citet{DengKL022} identified` to the end
  of the section: splits the 90-word run-on, removes "attentive version of the
  attentive algorithm", and adds a *reason* for the distributed-execution claim
  (marked `% AUTHORS: verify`).

**Background** (`sec3_background.tex`) — three cosmetic fixes
- Move the `\footnote{...}` out of the DCOP `\subsection{...}` title (attach it
  to "A DCOP" in the first sentence).
- Anchor `\label{BoothB19}` (on the BCT subsection) → `\label{sec:bct}`.
- Anchor `the standard Min sum` → `standard Min-sum`.

**§4 Effect of Splitting** (`sec4_effect_of_splitting.tex`) — four edits
1. Insert the `\paragraph{Scope of the analysis.}` block right after the
   sentence `...and the reason it triggers convergence.` — insert TOGETHER
   with edit 4 (it references `sec:extensions`).
2. Assumption harmonized: anchor `M_a < M_b \ll B_b < B_a`, replace the
   sentence with the `<` version (≪ kept as intuition only); also change the
   two later occurrences of `agreement structure $M'_a < M'_b \ll B'_b < B'_a$`
   to `<`. (The Lemma already uses `<`; the ≪ did no formal work.)
3. Micro: footnote after `the content of the $R$ messages` gets a period;
   anchor `need to be much larger` → `is much larger`.
4. **Insert new §4.5** — everything from `\subsection{Toward Damped and
   Asymmetric Splits}` to the end of `sec4_effect_of_splitting.tex`, placed
   after the `\end{remark}` of "Tightness of the perturbation bound",
   immediately before `\section{Splitting with No Damping}`.
   NEW MATH — co-authors must verify (and ideally Lean) both propositions.

**§5 Splitting with No Damping** (`sec5_splitting_no_damping.tex`)
- Replace the whole section (anchor `\section{Splitting with No Damping}` to
  just before the Experiments section).
- What changed: lemniscate finding stated as a numbered Observation; the merge
  extraction now matches the code (iterations 198/199, both branches, best
  kept — was "that iteration and following two iterations"); `is dramatically`
  → `is dramatic` + honesty clause vs. DMS-SCFG; Fig. 2 caption notes the
  copies hold halved entries.

## Safe merge order

1. Part 1 (with the 3 micro-edits) — compiles standalone against the current doc.
2. §5 replacement — no dependencies.
3. Abstract / Intro / Related Work / Background — independent, any order.
4. §4 edits 1–4 together — then revert the 3 micro-edits in §6.
