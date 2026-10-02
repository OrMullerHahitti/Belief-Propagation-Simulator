# Peer Review: Split and You Shall Converge: How does Function Splitting Trigger the Convergence of Belief Propagation

| | |
|---|---|
| **Decision** | Reject |
| **Recommendation** | Weak Reject |
| **Overall score** | 4/10 — weak reject: notable flaws outweigh the merits as the file stands |
| **Reviewer confidence** | 4/5 — confident: the math was re-checked by exact simulation and the numbers against the data files; the benchmark runs were not repeated |

Reviewed file: `min-sum_split_AAMAS-2027_v4_section6.tex` at Overleaf commit `018c212` (the newer commit `95be9de` changes no file). All line numbers below refer to that file. Review date: 2026-09-30.

Numbering note: this file has one more definition than the version the 5.5/10 reviewer read (Roie added "Assignment Convergence"). So the reviewer's Lemma 4.5 is Lemma 4.6 here, the reviewer's Theorem 4.8 is Theorem 4.9 here, and the reviewer's Theorem 4.9 is Theorem 4.10 here.

## Summary

The paper asks why splitting every function-node of a factor graph into two half-cost copies makes damped Min-sum (DMS) converge fast. For one split binary constraint without damping, it derives a clipped recurrence for the message differences around the 4-node cycle. From it, the paper gets the limits under a constant outside message (Lemma 4.6), bounds on the flipping threshold (Lemma 4.8), a count of passes until a bound is reached (Theorem 4.9), and an exact threshold for staying at the bound (Theorem 4.10). A lemniscate example shows that undamped Min-sum on a split graph can run as two offset runs that settle on different solutions (Section 5). On five DCOP benchmarks the paper measures how many messages sit at a bound, how fast DMS-SCFG converges, what happens when the split is delayed, and how well a search over the two alternating assignments of MS-SCFG does (Section 6).

## Strengths

- The local analysis is exact and checkable. The one-pass recurrence (Equation 1, appendix L776–780) gives closed-form limits and thresholds. The worked example (L556–570) is correct: in my exact simulation a single outside message of −24 received at iteration 21 gives a tie, and −25 flips X_j, as the text says.
- The lemniscate explanation (L599–603) is concrete. It explains why undamped Min-sum shows costs 300 and 1200 while its two offset runs sit on 130 and 132.
- Section 6 measures the mechanism, not only the cost: the fraction of messages at a bound (Figure 5, L671), the final message change, assignment stability, and the Hamming distance between the two snapshots handed to the selection (L673).
- The statistics are reported with care: paired Wilcoxon tests with win counts, per-instance Spearman correlations with the undefined cases named (L696), branch-and-bound cap hits disclosed (L649), and a hindsight caveat for DMS-BDS (L698). Every Section 6 number I compared against `experiments/aamas/section6/out/numbers.md` matches (DMS-BDS, DABP, selection, fractions at a bound, Table 1).
- Negative results are reported, for example DMS against DMS-SCFG on random dense (p = 0.054, DMS lower on 37 of 50, L676) and no effect of the split moment on the structured benchmarks (L696).

## Weaknesses

- **This file is behind the version the last reviewer scored.** 36 of the 51 edits in `publish/apply_edits.py` are missing: every edit outside Section 6. Section 6 is identical to the review-edits file, but the rest comes from Roie's v4, which never received those edits. So several points the reviewer called "resolved" are open again: damping at variable-nodes only, footnote 6, the Lemma 4.6 proof, the two-iteration delay in Theorem 4.10, the Beyond Trees citation, the consensus description, the meeting scheduling variables, the zoom figures, and the Figure 10 caption. The Conclusions are also the old ones. Section 1 of the detailed comments lists all 36.
- **Theorem 4.10 is false as worded.** I reproduced the reviewer's counterexample exactly: M_a=0, M_b=2, B_a=30, B_b=20, a single −25 received at iteration 23. The messages to X_j sit at 20 in iterations 21–24, and every later input satisfies the inequality, yet the message at 25 is −1 (L505–507, L517–519). The fix is one phrase (Section 3 below).
- **Four figures do not match the text or the stated algorithm.** Figure 10 (L1136) is still the file made with damping on function-to-variable messages: its unsplit λ=0.9 line dips to about −6.5 and recovers. Its caption still says "M_a = 8 and M_b = 0". The appendix zoom figures (L1091–1106) are the pre-fix files. In the random sparse one, "Opt merge" ends below DMS-SCFG, the opposite of L701. Figures 3–4 draw DABP on all five benchmarks, three of them from older June runs the text never mentions. The ternary figures (L1112–1126) are pre-fix, in engine units, with old line names.
- **Several sentences claim more than the measurements show.** Examples: "Thus the feedback loop of Section 4 ... is what most messages of these factor graphs settle into" (L671). "although most of its messages sit at a bound" (L673), which is false on graph coloring (0.19) and meeting scheduling (0.07). "To validate this hypothesis" (L608). "damping, which joins the two runs" (L701, L709). And the Conclusions (L716) say the new heuristics "have an advantage over previous versions", but the selection heuristic stays behind DMS-SCFG on every benchmark (L701).
- **The delayed split changes two things at once.** At the split the damping history is cleared, so the first messages after it are not damped. The file hides this: the sentence is commented out (L623), and the DMS-kDS entry (L649) does not say it. A control run exists (2026-09-30) and answers the reviewer's objection. It is not in the paper.
- **The positioning contradicts the group's own prior work.** The paper never cites Zivan et al. 2020 (Beyond Trees, `ZivanLG20`), which has the same lemniscate, the same 130/132 solutions, the BCTs of L232–239, and a convergence result for split trees with damping. The abstract (L117) and introduction (L161) still say no theory exists. The consensus work is described wrongly (L176).
- **The main text is about 0.9 page over the 8-page limit.** The Conclusions and References start on page 9 of 15, and all proofs are in the supplement.

## Detailed comments

### 1. Start from the right file

I ran `publish/apply_edits.py`'s edit list against this file. 15 edits are present (all in Section 6). 36 are missing, and each missing edit still finds its original text here, so none was replaced by a different fix:

- Front matter and framing: E02 title, E03 intro prior theory, E04 related work (Cohen + Beyond Trees), E05 Rebeschini.
- Background: E26 iteration units, E06 damping only at variable-nodes, E07 BCT citation.
- Section 4: E08 footnote 5, E09 no ties, E10 footnote 6, E11 scope sentence, E12 Lemma 4.6 tie case, E13a–e Theorem 4.10 wording and delay, E14 larger domains.
- Section 5: E15, E16a, E16b, E17, E18, E19, E20, E21, E22a (split state and damping reset), E22b (DMS-BDS).
- Conclusions: E36.
- Appendix: E37a–c (Lemma 4.6 proof), E13f (Theorem 4.10 proof), E38 (meeting variables), E40a–b (Figure 10 file and caption).

A dry run in the scratchpad (nothing in the project was written) shows the way back is short:
- All 36 edits apply to this file without a mismatch.
- The cut set of `publish/apply_cuts_v2.py` then applies 28 of its 35 replacements.
- The result compiles to 14 pages, with only the Conclusions spilling onto page 9.

The 7 cut replacements that fail are C4a, C4d, C4e, K15, K16b, K18 and K24. They fail because Roie rewrote Definition 4.1, two Section 4 sentences and nothing else, Shir and Or rewrote Theorem 4.9 today, and the Conclusions here are the old ones.

Warning: the stored replacement text of C4d and C4e is the old Theorem 4.9. Copying it would silently undo today's rewrite. Inline today's statement by hand instead.

### 2. The last review, point by point

| Reviewer's point | In this file | Where and what to change |
|---|---|---|
| Figure 10 does not follow Q-only damping | Not fixed. The Overleaf file is the R-damping version; `s6_unary17.pdf` was never uploaded. | Upload `experiments/aamas/section6/overleaf_upload/s6_unary17.pdf` (its unsplit λ=0.9 line rises from −16 to +1, the split λ=0.9 line reaches about 12 at 400). Use caption K22 (`apply_cuts_v2.py`), which prints the matrix with row and column labels and states X_i = X_2, X_j = X_1. L1136–1138. |
| Print the table orientation | Not fixed | Covered by K22. |
| Theorem 4.10 pending input at the window end | Not fixed | See Section 3, item 1. |
| Delayed split = split + damping reset | Not disclosed at all now | Re-apply E22a (L623), then add the control result (Section 5, item 1 below). |
| "feedback loop ... is what most messages settle into" | Still there, L671 | Replace with: "These fractions and the small final message changes are consistent with the feedback loop of Section 4 taking part in the stabilization, although they do not establish the hypotheses of the single-cycle theorems on the full graph." |
| Residual: which messages, which norm, which window | Final step only, "any message" (L671) | Use K21: variable-to-function messages, maximum norm after normalization, over the last 800 iterations (same counts 48, 49, 48, 26, 37). |
| "most messages at a bound" for MS-SCFG | Still unrestricted, L673 | Use K21b: "on the random benchmarks". |
| Lemma 4.6 proof: first pass may start above the limit; tie case | Not fixed: L398 says "remain constant", L800–801 says "increases ... until", L840–841 still has the leftover note | Re-apply E12, E37a–c. |
| Footnote 6 identities | Not fixed, L308 | Re-apply E10. |
| B_a < B_b proof "starts from d or from 2d" | Not fixed, L907 | "starts from d or from min{2d, B_b − M_a}". |
| Index t−1 should be t−2 in the constant-outside expansion | Not fixed, L379–380 | Replace R''^{t−1} by R''^{t−2} in the four terms after the second equals sign (L377 correctly has Q'^t = R̄_i + R''^{t−1}). |
| Definition 4.1 names finite-time convergence | Not fixed, L282 | Title it "Finite-time convergence of message differences", or add one sentence that "converges" means "becomes constant" throughout Section 4. |
| Random unaries break ties "categorically" | Still categorical, L280 and L1051 | "We assume no ties. In the experiments, small random unary costs make exact ties unlikely, but they do not rule them out." |
| Update schedule / pseudocode | Not given | Re-apply E26 and add: "With damping, each step updates all Q messages from the previous R messages, damped with the previous Q, and then all R messages from the new Q; the step-equals-two-iterations reading is exact only for λ = 0." Two sentences cost less space than an algorithm box. |
| Two-run explanation on odd-cycle graphs | The "approximately separate" sentence is commented out (L606), but "validate" (L608), "joins the two runs" (L701) and "the two runs are joined" (L709) remain | If E17 is re-applied, use the reviewer's wording for its second sentence: "The benchmark graphs contain odd cycles. There we observe the same alternation between two assignments, but this example does not establish its cause." Change L608 to "To use this observation". Delete the "which joins the two runs" clause at L701 (and in K10). At L709, write "With damping the alternation disappears". |
| Candidate set spans up to 2^n assignments | "good candidate set" is already there (L701) | No change beyond the row above. |
| Larger domains presented as established | Asserted, L574–576 | Re-apply E14 and add: "We state this as an interpretation; it is not proved here." Also fix "binary constraints" to "binary domains" (L574). |
| "opt" after a time cap | Name kept; "The exact search" at L701 | Rename to MS-SCFG-BB, or keep the name and write "the branch and bound search" at L701. Add the search time. For seed 0, `experiments/aaai/data_cuda/merge_timing.csv` gives 0.44, 300, 30, 22 and 0.47 s (sparse, dense, scale free, coloring, meeting). That is the time of 125 to 18,869 DMS steps, while MGM takes 2 to 4. |
| Medians over different subsets | Labeled correctly (L676) | Fine as is. A paired count on the runs where both converged would help if space allows. |
| Undefined Spearman correlations | Fixed (L696) | — |
| DABP protocol | Partly | Add whether the random sparse and graph coloring DABP runs restart (an earlier check found no restart in them), and that the reported cost is the current cost, not the best so far (the CSVs hold both). Use the one-row cost figure (K08, DABP only on sparse and coloring) so the older June runs are no longer drawn. |
| Multiple comparisons | Not handled | With Holm over Table 1's 26 cells at 0.05, 14 of the 15 stars survive; only random sparse k=600 (p = 0.0097) loses its star. State this in the caption. |
| Ternary results | Not labeled as older runs, pre-fix, old names, engine units | Remove them from this submission (L197 sentence and L1111–1132), or replace them with the post-fix runs in `experiments/aaai/ternary_plots/` plus a generator and units sentence. Removing saves space. |
| Abstract "has not yet been established" | Still there, L117 | See Section 7. |
| "Neither work" should allow Cohen et al.'s exterior-vector discussion | Missing entirely (E03, E04 not applied) | When re-applying E03 and E04, write: "Cohen et al. discuss how halving the tables changes the weight of incoming messages relative to the table costs; what is new here is the explicit recurrence and the thresholds for progress and persistence." |
| Conclusions overclaim | Still there, L714–716 | See Section 7. |
| 8 pages | 0.9 page over | See Section 8. |

### 3. Section 4 (theory)

What holds: Equation 1, the limits of Lemma 4.6 in both strict cases, the bounds of Lemma 4.8, the thresholds of Theorem 4.10, and the worked example all agree with exact rational simulation of the paper's schedule (earlier sessions: thousands of random tables; today: the example and the two items below). The core math does not need rebuilding. The problems are in the statements.

1. **Theorem 4.10 window (L505–507, L517–519).** Values received at iterations T−1 and T are already on their way to X_j when a four-iteration window T−3..T closes. They reach X_j at T+1 and T+2. Proposed statement: "Suppose that the differences of all messages sent to X_j in four consecutive iterations T−3, …, T are equal to the upper bound B_b − M_a. If every value δ received from iteration T−1 on satisfies [the inequality], then every difference sent to X_j after iteration T remains equal to B_b − M_a. Conversely, when a value that violates this inequality is received, the messages sent to X_j two iterations later are no longer at B_b − M_a." Do the same for the lower bound. In my check over 4000 random tables, the current wording has 83 counterexamples and the T−1 wording has none. The "four iterations later" at L515, L527 and L537–538 must become "two iterations after the value is received" (E13b, E13d, E13e).
2. **Theorem 4.9 as rewritten today (L483–484, L490–492).** "the cycle reaches one of its bounders and remains there" is true only in a weak sense. In my check (3000 random tables, δ changing every iteration, only δ + 2d ≥ ε assumed):
   - X_j selects a after the stated count in all 3000 runs.
   - At least one bound is active in every later iteration, in all 3000 runs.
   - In 418 runs, however, the messages to X_j keep changing. Example: M_a=0, M_b=7, B_a=26, B_b=24. The message to X_i sits at its cap 26, while the message to X_j takes the values 24, 22, 24, 21.

   A reader will take "remains there" to mean the messages stop. Proposed wording: "X_j selects the value a from then on, and in every later iteration at least one bound is active: the differences sent to X_j equal B_b − M_a, or the differences sent to X_i equal B_a − M_a."

   The proof (L915–937) shows positivity but not that a bound is reached within the count when the extra condition fails. Add after "or r_{k+1} ≥ r_k + ε": "More precisely, either the inner operation selects B_a − M_a, and the bound toward X_i is active, or r_{k+1} ≥ min{B_b − M_a, r_k + ε}; so a bound is reached within the stated number of passes, and since 2d + δ_k > 0 at least one of the two stays active."
3. **Lemma 4.6 tie case (L398).** "messages four iterations apart remain constant" holds only after the first pass. Use E12: "become constant after at most one complete pass". The tie case also contradicts the no-tie assumption of L280, because one of the interleaved sequences sits at exactly 0. Say so in one clause, or exclude the case.
4. **Labels and names.** Two definitions share `\label{def:message-difference-convergence}` (L283, L296), so rename the second to `def:assignment-convergence`. Definition 4.7 writes FT_{F'_{ij}} (L413), while Lemma 4.8 and the text write FT_{F_{ij}} (L429–439, L542, L561, L576). Pick one.
5. **Footnote 5 (L273).** "immediate convergence" is true for the assignment, not for the message (the message to X_i in the earlier check went 3, 3, 4, 4). Use E08.
6. **T1 remark (L334–336).** "reaches its cap first" is not true in time for every table (earlier check: false for 1054 of 6000 random message sequences; one example is M_a=3, M_b=24, B_a=27, B_b=39). "is capped by the smaller bound" says what is meant.

### 4. Section 5

- L588 claims both Cohen et al. and Section 6 "demonstrate" oscillation "between (often two)" solutions. Section 6 now shows exactly two assignments on 248 of 250 runs (L673), so the sentence can simply point there.
- L599–600: cite ZivanLG20 for the lemniscate and its 130/132 solutions (E16a). Fix "vise versa" twice (L600, L602; E16b).
- L604 (Roie's BCT paragraph) ends with a broken sentence ("The result is that if the different structure of BCTs cause ..."). Use E18 or the shorter K13.
- L608: "SyncBB" does not match Section 6, which ran a centralized branch and bound warm-started from MGM. Write "a centralized branch and bound search (Section 6)".
- L616 says damping combines "every message". In the implementation only variable-to-function messages are damped. Re-apply E06 in Section 3 and write "every variable-to-function message" here.
- L618: the sentence "when solving a cycle generated by a split MS-SCFG converges" has no subject. Its caveat (sufficient, not necessary; theorems stated for λ = 0) is commented out, so the paragraph now reads as if the theorems cover DMS. Re-apply E20. Fix "alternatives.By" (L619, E21).
- L623–625: Inter-DMS is introduced as interleaving three versions, but only two orders were run (Section 6, L649). One of them, MS-SCFG followed by a search, contains no DMS at all. Say "we evaluate two orders" here, and drop "demonstrate which of the versions was the most successful" (L625), which says nothing.

### 5. Section 6

The section is the strongest part of the file. It needs these changes:

1. **Report the reset control.** Proposed text after the DMS-kDS entry (L649) or at L679:

   > At the split the damping history is cleared, so the first messages after the split are not damped. Clearing the history alone does not cause the effect: when DMS clears it at k = 1000 without splitting, 100 iterations later 0.08, 0.61, 0.18, 0.01 and 0.05 of the messages are at a bound, as in DMS, and its final cost is not lower than that of DMS on any benchmark.

   Source: `experiments/aamas/splitting_explanation/results/exp1_reset_summary.md`, run 2026-09-30, 50 instances per benchmark. The run uses the lab engine of the exp1 study, like Figure 5. Its DMS-kDS fraction on graph coloring is 0.68, while the text's exp1_delayed run gives 0.66. Quote only the reset line from this run to avoid two numbers for the same thing.
2. **Fix the scale-free sentence (L696).** The table marks k=100 as significant, and the data agree (p = 2.6·10⁻⁵). Only k=200 is not significant (p = 0.026). The gain is also not monotone: 0.39, 0.27, 0.59, 0.65, 0.75. Proposed text: "significantly for every k on random sparse and random dense and for every k except 200 on scale free; on random dense the gain grows with k, from 0.22% to 0.40%, and on scale free from 0.27% at k = 200 to 0.75% at k = 2000."
3. **Add the missing p-values (L676):** graph coloring −75%, p = 2·10⁻¹⁰; meeting scheduling −62%, p = 1.2·10⁻¹⁴.
4. **L679, "At every k we tested, the split therefore takes effect ...".** Only k = 1000 is shown. Either cite the other k values in the supplement or drop "therefore".
5. **L707, "the split at the best recorded state ended lowest of all".** Add "(in hindsight)", since L698 already says it is not a competing version.
6. **L671 last sentence and L701/L709 two-run clauses:** see the table in Section 2.

### 6. Supplement

- **Zoom figures (L1091–1107).** Point them to `s6_sparse_zoom_2.pdf` and the other `s6_*_zoom_2.pdf` files, which are already in Overleaf and match the main text. Rewrite the captions with the current names (DMS-SCFG, DMS-kDS, DMS-BDS, MS-SCFG-opt). The current caption also has a stray period ("Opt versions. near the end").
- **Figure 10 (L1134–1140).** See the table in Section 2. The y-axis label of the old file reads "Belief Delta og Message". The new file's label "b[0] - b[1]" is code-like and could read b_{X_1}(a) − b_{X_1}(b).
- **The text before Figure 10 (L1063)** says "a single unary constraint that breaches FT", which is unclear. Say which value the outside message favors and that it is constant.
- **Meeting scheduling (L1045–1049).** State that the variables are the 20 meetings (E38). Otherwise "90 agents" conflicts with "each agent holds exactly one variable" (L197).
- **Proofs only in the supplement (L245 footnote).** The AAMAS instructions say reviewers need not read supplementary material. The main text should keep Equation 1 and one sentence on how each result follows from it. The last reviewer asked for exactly this.

### 7. Abstract, introduction, related work, conclusions

- **Abstract (L115–119).**
  - "standard Min-sum fails to converge" is too broad: say "often fails to converge on problems with cycles".
  - Replace the second paragraph with something like: "Recently, empirical evidence showed that splitting the function-nodes of the factor graph makes DMS converge far faster to high quality solutions. Earlier analyses explain this for a single split constraint and for split trees with strong damping, but not for the small cycles that the split creates inside a larger graph."
  - Replace the last sentence with the actual scope: a single split binary constraint without damping, measured on benchmarks, with a delayed split that lowers the cost on problems with random constraints.
- **Introduction.** L161: re-apply E03, with the Cohen et al. nuance from the table. L169 is fine.
- **Related work.** L174: re-apply E04. L176: re-apply E05 (consensus means averaging, not "as many agents as possible should agree").
- **DABP claims (L178–180).** "evidently, does not work well on benchmarks with structured constraints" rests on graph coloring, where the difference is not significant (39.2 against 34.4, p = 0.31, L703). K14 removes it; use K14.
- **Conclusions, first paragraph (L714).** "it often converges to more than one solution, which the algorithm not only oscillates between them..." is not grammatical, and it states the two-run mechanism as general. Proposed text: "without damping, Min-sum on a split factor graph often alternates between two assignments; on the lemniscate we show that these come from two offset runs that settle on different solutions."
- **Conclusions, second paragraph (L716).** "These heuristics were found to have an advantage over previous versions of the algorithm, especially in distributed settings" is not supported. The selection heuristic stays behind DMS-SCFG on every benchmark (L701). The delayed split helps only on the random benchmarks, by 0.2–0.75% (Table 1). And no experiment compares distributed settings. Replace it with K18, which says exactly that.

### 8. Page budget and a path to a submittable file

- Current state: 15 pages. Page 9 holds a full column and most of the second column of main text before the References.
- The measured path (dry run, Section 1): re-apply the 36 edits, apply the cut set, redo the 7 failed cuts by hand, and upload `s6_cost_row.pdf`, `s6_unary17.pdf` and the re-made `s6_bound_fraction.pdf` from `experiments/aamas/section6/overleaf_upload/`. That leaves only the Conclusions on page 9.
- The new items in this review add roughly 10–15 lines. The new Conclusions (K18 plus the one-sentence first paragraph) are shorter than the current ones by about the same amount.
- If more space is needed, the cheapest cuts are the ternary sentence and figures, and the random-split sentence at L676 (already cut C2).
- Before submitting, also replace `<<OpenReview submission id>>` (L60, prints on page 1) and drop the trailing period in the title (L94).

### What I checked, and what I did not

- **Read and compiled.** Read all 1153 lines. Compiled with tectonic: 0 errors, 15 pages. Rendered the figure files that Overleaf holds (Figures 3–5, Figure 10, one zoom figure, one ternary figure) and the new local Figure 10.
- **Edit status.** Checked each of the 51 edits by script. Dry-ran the re-application of the edits and cuts in the scratchpad, then compiled the result.
- **Exact simulation.** Used exact rational min-sum on the split cycle, on the paper's schedule, for the Theorem 4.10 and Theorem 4.9 items and the worked example. Scripts: `theorem_checks.py`, `check_edits.py` and `dryrun_reapply.py` in this session's scratchpad (not in the repo).
- **Numbers.** Compared the Section 6 numbers with `experiments/aamas/section6/out/numbers.md` and `delayed_split_sweep.md` (built from `data_paper_20260928`). Computed the Holm correction from those p-values.
- **Not re-run.** I did not re-run any benchmark experiment, DABP, or the branch and bound.
- **From earlier sessions, not re-checked today:**
  - the contents of ZivanLG20 and Cohen et al. 2020;
  - the AAMAS 8-page rule;
  - the restart in the older DABP runs, and no restart in the random sparse and graph coloring ones;
  - that the implementation damps only variable-to-function messages;
  - that the residual statistic covers variable-to-function messages, and its counts over the last 800 iterations;
  - the footnote 5 and T1 remark counterexamples.

## Questions for the authors

1. The delayed split also shortens the run after the split (4000 − k iterations). Does the gain in Table 1 remain when every DMS-kDS run gets the same number of iterations after its split? If it does, the "later split starts from a better solution" reading (L696) becomes much stronger.
2. With damping (λ > 0) on the single split cycle, do the thresholds of Theorems 4.9–4.10 still predict which bound is reached, only more slowly? Figure 10 suggests so. A one-line statement of what is observed would help the λ = 0 caveat.
3. On random dense the search stopped at the cap on 48 of 50 instances, and it improved on MGM on 1 instance. Is the "opt" result there any better than MGM's? If not, can the text say so?
4. Can the two-run picture be tested on a non-bipartite benchmark? For example, what fraction of variables in the 398 and 400 snapshots agree with a DMS-SCFG solution? That would support or refute the "candidate set" reading directly.
5. For the DABP lines that remain: were the models trained per instance during the run, as in Deng et al., or trained once and reused?

## Minor issues

- L94: the title is a question ending in a period.
- L60: submission ID placeholder prints on page 1.
- L232–233: the BCT paragraph has `\label{BoothB19}`, a citation-like label, and no citation (E07 adds ZivanLG20).
- L309: "guaranties" → "guarantees". "Under very mild conditions that we will specify below" should name the condition (E11).
- L345: space before the period in "much larger .".
- L460–461: the definition of "clipping" is commented out, but the appendix uses the word throughout (it defines the bracket at L773, which is enough if the main text no longer uses the word).
- L583: "combine versions" → "combined versions" (E15).
- L600, L602: "vise versa" (E16b).
- L619: "alternatives.By" (E21).
- L644: "4000 synchronous iterations" should say that one implementation step counts as two iterations (E26).
- L673: "Only with damping do the runs converge." By the paper's own criterion, 1 of the 250 MS-SCFG runs converged (graph coloring) and 2 of the 50 MS runs on random sparse, so write "Almost only".
- L757: "all differences generated on the cycle internal" has no verb.
- L840–841: leftover note "This boundary case will be stated separately in the lemma." (E37c).
- L1097: stray period in "Opt versions. near the end".
- L1137: "$a$ , the lower-cost" has a stray space; the whole caption is replaced by K22 anyway.

## Assessment by criterion

| Criterion | Rating | Note |
|---|---|---|
| Significance / contribution | Fair | A real DCOP question. But the theory covers one split binary cycle without damping, and the new heuristics gain 0.2–0.75% on random benchmarks only. |
| Originality | Fair | The recurrence and exact thresholds are new. The lemniscate, its solutions and BCTs come from uncited earlier work, which makes the novelty look overstated. |
| Soundness / validity | Fair | The core math checks out. Theorem 4.10 is false as worded, Theorem 4.9's new wording is loose, and four figures contradict the text or the algorithm. |
| Clarity / presentation | Fair | Section 6 reads well. Elsewhere there are broken sentences, typos, SyncBB vs branch and bound, and 0.9 page over the limit. |
| Related work / positioning | Poor | Beyond Trees is missing, and the text says no explanation "has been discovered since". The consensus work is misdescribed. |
| Reproducibility / transparency | Fair | Section 6 gives counts, tests and caps. Missing: the damped schedule, the damping reset, the search time, the DABP protocol, and code. |
| Ethics & limitations | Fair | No ethical issues arise. The λ = 0 limitation is stated in Section 6 but commented out in Section 5, and the Conclusions ignore it. |

## Justification

The score is 4 because this file lost most of the repairs that took the previous version from 4 to 5.5. The Section 6 improvements are all here, but the positioning, damping definition, proof repairs, Figure 10 and supplement fixes are not. It also still carries the reviewer's open items: the Theorem 4.10 window, the damping reset, and the overclaiming sentences. The deciding factor is soundness in the reader's eyes. A reviewer who checks Theorem 4.10, Figure 10 or the zoom figures will find a contradiction, and one who knows Beyond Trees will doubt the novelty claims.

Almost all of this is text work with drafts that already exist: the 36 edits, the cut set with K21/K21b/K22/K08, and the new items above. Only the reset control was needed as new evidence, and it has been run. With those changes, and the main text inside 8 pages, I would rate the paper about 6/10 (weak accept). Soundness and positioning would move to Good. What keeps it from 7 or higher is its scope: the theory is local, the link to large graphs is a correlation (the fraction of messages at a bound), and the heuristics give small gains or, for the selection, none over DMS-SCFG.
