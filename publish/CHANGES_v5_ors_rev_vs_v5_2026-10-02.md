# Everything changed in the working copy, against Roie's v5

Files: `min-sum_split_AAMAS2027_v5_ors_rev.tex` (the working copy, Overleaf commit `bf94661`) against `min-sum_split_AAMAS2027_v5.tex` (Roie's file, unchanged since his commit `c06e55d` of 1 Oct 13:46). Written 2 Oct 2026.

**25 places differ, +29 / −38 lines.** The working copy started as a byte copy of v5 on 1 Oct 14:08. Everything below was done after that. Roie's v5 was not edited. The abstract was changed on 2 Oct at 11:21 and put back at 12:27, so it is identical in both files.

Line breaks inside the diff blocks are for reading only. The exact text is in the companion file `v5_to_v5_ors_rev_2026-10-02.patch` (apply it to v5 and you get the working copy).

## The pushes

| Commit | When | What | Asked where |
|---|---|---|---|
| `505770b` | 2 Oct 10:47 | Figure 10: New figure file made with damping on the variable-to-function messages, and its caption. | your second window: "do 1 and 2 and push straight" (10:43) |
| `766140c` | 2 Oct 10:55 | Theorems 4.9 and 4.10: The two theorems, the text after them, the example, and the proof of 4.10 in the appendix. | this window: "make changes to 4.9 and 4.10 and push to v5_orsrev" |
| `a54f9fb` | 2 Oct 11:08 | Section 6 answers: Four answers to the AAAI reviewers: two captions, the DABP time factors, how k was chosen. | your third window: "you can change section 6 changes needed and push as well" (11:06) |
| `1a7c2c2` | 2 Oct 11:21 | Novelty and scope: Abstract, introduction, related work, BCT citation, Section 5, larger domains, conclusions, Figure 10 text. | your third window, after "yes, i pushed check my push" (11:17); no message there asked for these |
| `bf94661` | 2 Oct 12:19 | Abstract restored: The two abstract sentences of 1a7c2c2 put back as in v5; the abstract now equals Roie's. | this window: "revert only the abstract" |

## Overview

| # | Where | Lines | What changed | Push |
|---|---|---|---|---|
| 1 | Introduction, the paragraph on Cohen et al. (1 Introduction) | v5 161 → working copy 161 | Defines the SCFG acronym, and cites Cohen et al. and Zivan et al. 2020 (Beyond Trees) instead of "has not been discovered since". | `1a7c2c2` |
| 2 | Related work, first paragraph (2 Related Work) | v5 174 → working copy 174 | "This unexplainable success" becomes what the two earlier papers proved and what is new here: a split cycle that receives messages from the rest of the graph. | `1a7c2c2` |
| 3 | Backtrack Cost Trees (3 Background) | v5 235 → working copy 235 | Citation for backtrack cost trees (Zivan et al. 2020). | `1a7c2c2` |
| 4 | Theorem 4.9, end of the first case (4 The Effect of Splitting) | v5 485 → working copy 485 | "Difference and assignment convergence" was false when the outside value keeps changing; the theorem now claims that the assignment of X_j converges, which is what the proof gives. | `766140c` |
| 5 | Theorem 4.9, the extra condition (upper bound) (4 The Effect of Splitting) | v5 487–488 → working copy 487–488 | The condition used t̂, which this theorem never defines. It now holds for every iteration, and the conclusion says "from then on". | `766140c` |
| 6 | Theorem 4.9, the extra condition (lower bound) (4 The Effect of Splitting) | v5 493 → working copy 493 | Same fix for the lower bound. | `766140c` |
| 7 | Theorem 4.10, upper half, hypothesis (4 The Effect of Splitting) | v5 504 → working copy 504 | The condition now starts at t̂−2: a value received at t acts on the messages at t+2, so the two values received just before the window closes must be covered too. | `766140c` |
| 8 | Theorem 4.10, lower half, hypothesis (4 The Effect of Splitting) | v5 512–513 → working copy 512 | Same start of the condition, and the window is worded as in the upper half (copy F′, four consecutive iterations). | `766140c` |
| 9 | Theorem 4.10, lower half, converse (4 The Effect of Splitting) | v5 519 → working copy 518 | \> is a space in math mode, so the PDF printed no comparison sign. Now >. | `766140c` |
| 10 | The paragraph after Theorem 4.10 (4 The Effect of Splitting) | v5 531–532 → working copy 530–531 | "Four iterations following a violation" contradicted the theorem; now two iterations after the value is received, with one sentence on why the condition starts at t̂−2. | `766140c` |
| 11 | Before the example (4 The Effect of Splitting) | v5 535 → working copy removed | A sentence that broke off (its continuation was commented out) is removed. | `766140c` |
| 12 | Example, last sentences (4 The Effect of Splitting) | v5 557–560 → working copy 555–556 | The 4 → 24 increase holds for a single outside message; a value below −4 received in every iteration still replaces the assignment. Theorem 4.10 is about the messages, FT follows from them. | `766140c` |
| 13 | Extending to Larger Domains, first sentences (4 The Effect of Splitting) | v5 564 → working copy 560 | The theorems are for binary domains and are not extended; the rest of the paragraph is marked as informal, with a pointer to the measurements in Section 6. | `1a7c2c2` |
| 14 | Splitting with No Damping, the selection step (5 Damping, With or Without) | v5 598 → working copy 594 | Section 5 said SyncBB; the experiments ran a centralized branch and bound. Now: a complete search, SyncBB as the distributed option, what was run in the experiments. | `1a7c2c2` |
| 15 | Splitting with Damping, the paragraph that cites the theorems (5 Damping, With or Without) | v5 608–609 → working copy 604–605 | Commas in the sentence that cites the theorems; the caveat that the theorems are for λ = 0 (which was commented out) placed before the informal damping sentence; the typo "alternatives.By". | `1a7c2c2` |
| 16 | Splitting with Damping, Inter-DMS (5 Damping, With or Without) | v5 614 → working copy 610 | "MGM or SyncBB" becomes "MGM or the search described above". | `1a7c2c2` |
| 17 | Figure 3 caption (6 Experimental Evaluation) | v5 628 → working copy 624 | MS-SCFG-opt is drawn at iteration 400; the time of its search is not included. | `a54f9fb` |
| 18 | Algorithms, DABP (6 Experimental Evaluation) | v5 639 → working copy 635 | How DABP is put on the iteration axis: the measured time factors 6.9, 3.3, 5.6, 6.3 and 8.2 per benchmark. | `a54f9fb` |
| 19 | Figure 4 caption (6 Experimental Evaluation) | v5 646 → working copy 642 | Same note as in the Figure 3 caption. | `a54f9fb` |
| 20 | Splitting in the middle of a run (6 Experimental Evaluation) | v5 669 → working copy 665 | k = 1000 is the same value on every benchmark; no k was selected per benchmark; every k ran on the same 50 instances. | `a54f9fb` |
| 21 | Conclusions, first paragraph (7 Conclusions) | v5 704 → working copy 700 | "We prove, for a single split cycle"; the ungrammatical sentence about "converges to more than one solution" becomes "alternates between two solutions", with "to the best of our knowledge". | `1a7c2c2` |
| 22 | Conclusions, second paragraph (7 Conclusions) | v5 706 → working copy 702 | "An advantage over previous versions, especially in distributed settings" becomes what Section 6 shows: delayed split lower on random constraints; selection recovers most of the gap. | `1a7c2c2` |
| 23 | Proof of Theorem 4.10, last paragraph (Appendix: Proofs) | v5 1010–1014 → working copy 1006 | The proof now starts the induction from the four window iterations and uses the value received at t−2; tightness via monotonicity of the clipped update. | `766140c` |
| 24 | The text that describes Figure 10 (Appendix: Additional Experimental Results) | v5 1053 → working copy 1045 | Describes the new figure: the difference between the beliefs of X_1, the −17 against the first flipping threshold −16, why damping delays the reversal. | `1a7c2c2` |
| 25 | Figure 10, file and caption (Appendix: Additional Experimental Results) | v5 1124–1126 → working copy 1116–1117 | New file s6_unary17.pdf (damping on variable-to-function messages, 400 iterations) and a caption that prints the table, the mapping X_i = X_2, X_j = X_1, and the damping. | `505770b` |

## The changes, section by section

Red lines (−) are Roie's v5, green lines (+) are the working copy.

### 1 Introduction

#### 1. Introduction, the paragraph on Cohen et al.

Lines v5 161 → working copy 161. Push `1a7c2c2` (2 Oct 11:21).

Defines the SCFG acronym, and cites Cohen et al. and Zivan et al. 2020 (Beyond Trees) instead of "has not been discovered since".

```diff
- In \citet{CohenGZ20}, the splitting of nodes in the factor-graph on which the algorithm performs,
- which represent functions (i.e., function-nodes), was found to trigger rapid convergence of DMS. The
- DMS algorithm was performed on a new generated factor-graph in which each of the function-nodes in
- the original factor-graph was represented by two function-nodes. The costs for combination of
- assignments in the original function-nodes were divided between the two function-nodes in the new
- generated factor-graph. This caused rapid convergence to high quality solutions. However, although
- the success of this version of the algorithm (DMS-SCFG) was demonstrated empirically, a formal
- theoretic explanation of this phenomenon was not presented and has not been discovered since.
+ In \citet{CohenGZ20}, the splitting of nodes in the factor-graph on which the algorithm performs,
+ which represent functions (i.e., function-nodes), was found to trigger rapid convergence of DMS. The
+ DMS algorithm was performed on a new generated factor-graph in which each of the function-nodes in
+ the original factor-graph was represented by two function-nodes. The costs for combination of
+ assignments in the original function-nodes were divided between the two function-nodes in the new
+ generated factor-graph. This caused rapid convergence to high quality solutions. The resulting graph
+ is a \emph{split constraint factor graph} (SCFG) and the algorithm is DMS-SCFG. Its success was
+ demonstrated empirically, but the existing analyses \cite{CohenGZ20,ZivanLG20} cover a single split
+ constraint and split trees under strong damping, not the short cycles that the split creates inside
+ a larger graph.
```

### 2 Related Work

#### 2. Related work, first paragraph

Lines v5 174 → working copy 174. Push `1a7c2c2` (2 Oct 11:21).

"This unexplainable success" becomes what the two earlier papers proved and what is new here: a split cycle that receives messages from the rest of the graph.

```diff
- \citet{CohenGZ20} demonstrated that function-node splitting triggers rapid convergence of DMS. When
- used with standard MS without damping, in most cases splitting alternates between two low quality
- solutions. The DMS-SCFG version quickly converges to a much higher quality solution. This
- unexplainable success motivated this study.
+ \citet{CohenGZ20} demonstrated that function-node splitting triggers rapid convergence of DMS. When
+ used with standard MS without damping, in most cases splitting alternates between two low quality
+ solutions. The DMS-SCFG version quickly converges to a much higher quality solution. Their analysis
+ covers a single split constraint and explains the gain informally. \citet{ZivanLG20} introduced
+ backtrack cost trees and proved that on an SCFG generated from a tree, a large enough damping factor
+ makes DMS converge to the optimal solution. Neither work analyzes a split cycle that receives
+ messages from the rest of the graph, which is what we study.
```

### 3 Background

#### 3. Backtrack Cost Trees

Lines v5 235 → working copy 235. Push `1a7c2c2` (2 Oct 11:21).

Citation for backtrack cost trees (Zivan et al. 2020).

```diff
-         A {\em backtrack cost tree} (BCT) allows tracing for each belief the entries in the cost
- tables held by function-nodes that were used to compose it.% In other words, the components of the
- assignment's cost.
+         A {\em backtrack cost tree} (BCT) \cite{ZivanLG20} allows tracing for each belief the
+ entries in the cost tables held by function-nodes that were used to compose it.% In other words, the
+ components of the assignment's cost.
```

### 4 The Effect of Splitting

#### 4. Theorem 4.9, end of the first case

Lines v5 485 → working copy 485. Push `766140c` (2 Oct 10:55).

"Difference and assignment convergence" was false when the outside value keeps changing; the theorem now claims that the assignment of X_j converges, which is what the proof gives.

```diff
- and remains there, and $X_j$ selects the value assignment $a$. Thus, the algorithm reaches both
- difference and assignment convergence.
+ and remains there, and $X_j$ selects the value assignment $a$ from then on. Thus, the assignment of
+ $X_j$ converges.
```

#### 5. Theorem 4.9, the extra condition (upper bound)

Lines v5 487–488 → working copy 487–488. Push `766140c` (2 Oct 10:55).

The condition used t̂, which this theorem never defines. It now holds for every iteration, and the conclusion says "from then on".

```diff
- If, in addition, $\Delta_{\bar R_i^t}\geq(B_b-B_a)-d$ for every iteration $t > \hat{t}$, the bounder
- reached is $B_b$ and every difference sent to $X_j$ is equal to $B_b-M_a$.
+ If, in addition, $\Delta_{\bar R_i^t}\geq(B_b-B_a)-d$ for every iteration $t$, the bounder
+ reached is $B_b$ and, from then on, every difference sent to $X_j$ is equal to $B_b-M_a$.
```

#### 6. Theorem 4.9, the extra condition (lower bound)

Lines v5 493 → working copy 493. Push `766140c` (2 Oct 10:55).

Same fix for the lower bound.

```diff
- reaches one of its bounders and remains there, and $X_j$ selects the value assignment $b$. In
- addition, if $\Delta_{\bar R_i^t}\leq(B_b-B_a)-d$ in every iteration $t > \hat{t}$, the bounder
- reached is $B_a$ and $\Delta_{R^{'t}_j} = M_b-B_a$. If $B_a-B_b\leq d$, this additional inequality
- follows
+ reaches one of its bounders and remains there, and $X_j$ selects the value assignment $b$. In
+ addition, if $\Delta_{\bar R_i^t}\leq(B_b-B_a)-d$ in every iteration $t$, the bounder reached is
+ $B_a$ and, from then on, $\Delta_{R^{'t}_j} = M_b-B_a$. If $B_a-B_b\leq d$, this additional
+ inequality follows
```

#### 7. Theorem 4.10, upper half, hypothesis

Lines v5 504 → working copy 504. Push `766140c` (2 Oct 10:55).

The condition now starts at t̂−2: a value received at t acts on the messages at t+2, so the two values received just before the window closes must be covered too.

```diff
- Assume that iteration $\hat{t}$ follows four consecutive iterations in which $\Delta_{R^{'t}_j} =
- B_b-M_a$. If for every $t > \hat{t}$,
+ Assume that iteration $\hat{t}$ follows four consecutive iterations in which $\Delta_{R^{'t}_j} =
+ B_b-M_a$. If for every $t \geq \hat{t}-2$,
```

#### 8. Theorem 4.10, lower half, hypothesis

Lines v5 512–513 → working copy 512. Push `766140c` (2 Oct 10:55).

Same start of the condition, and the window is worded as in the upper half (copy F′, four consecutive iterations).

```diff
- Similarly, assume that at iteration $\hat{t}$ the differences of four consecutive messages sent to
- $X_j$ were all equal to the lower bound $M_b-B_a$. If for every $t > \hat{t}$,
+ Similarly, assume that iteration $\hat{t}$ follows four consecutive iterations in which
+ $\Delta_{R^{'t}_j} = M_b-B_a$. If for every $t \geq \hat{t}-2$,
```

#### 9. Theorem 4.10, lower half, converse

Lines v5 519 → working copy 518. Push `766140c` (2 Oct 10:55).

\> is a space in math mode, so the PDF printed no comparison sign. Now >.

```diff
- In contrast, if at some iteration $t > \hat{t}$ $\Delta_{\bar R_i^t}\> -\max\{2d,\;(B_a-B_b)+d\}$,
- then $\Delta_{R^{'t+2}_j} > M_b-B_a$.
+ In contrast, if at some iteration $t > \hat{t}$ $\Delta_{\bar R_i^t} > -\max\{2d,\;(B_a-B_b)+d\}$,
+ then $\Delta_{R^{'t+2}_j} > M_b-B_a$.
```

#### 10. The paragraph after Theorem 4.10

Lines v5 531–532 → working copy 530–531. Push `766140c` (2 Oct 10:55).

"Four iterations following a violation" contradicted the theorem; now two iterations after the value is received, with one sentence on why the condition starts at t̂−2.

```diff
- inequality, $\Delta_{R_j}$ remains at the bound; four iterations following a violation, the
- corresponding difference sent to $X_j$ moves away from the bound.
+ inequality, $\Delta_{R_j}$ remains at the bound; two iterations after a violating value is received,
+ the
+ corresponding difference sent to $X_j$ moves away from the bound. A value received at iteration $t$
+ reaches the messages sent to $X_j$ at iteration $t+2$, which is why the conditions start at
+ $\hat{t}-2$.
```

#### 11. Before the example

Lines v5 535 → working copy removed. Push `766140c` (2 Oct 10:55).

A sentence that broke off (its continuation was commented out) is removed.

```diff
- Note that the difference between a value of $\Delta_{\bar R_i}$ received in a
```

#### 12. Example, last sentences

Lines v5 557–560 → working copy 555–556. Push `766140c` (2 Oct 10:55).

The 4 → 24 increase holds for a single outside message; a value below −4 received in every iteration still replaces the assignment. Theorem 4.10 is about the messages, FT follows from them.

```diff
- Nevertheless, $FT_{F'_{ij}}$ can also increase.
- Theorem~\ref{theo:conv} establishes that it remains $-24$ when $\Delta_{\bar R_i} \geq
- -\min\{4,12\}=-4$.
- Thus, when $\Delta_{\bar R_i} < -4$ it grows.
- Thus, the feedback loop generated as a result of the split increases the absolute difference
- required to replace the assignment of $X_j$  from $4$ to $24$. However, the absolute difference
- required for increasing or decreasing this flipping threshold remains $4$ and is not affected by the
- feedback loop.% ORS-REV T4 clarified
+ Nevertheless, $FT_{F'_{ij}}$ can also increase. By Theorem~\ref{theo:conv}, the messages stay at the
+ bound, and with them $FT_{F'_{ij}}$ stays at $-24$, as long as every value of $\Delta_{\bar R_i}$
+ received is at least $-\min\{4,12\}=-4$; when values below $-4$ are received, it grows.
+ Thus, the feedback loop generated by the split increases the absolute difference that a
+ \emph{single} outside message needs in order to replace the assignment of $X_j$ from $4$ to $24$.
+ However, the value that decides whether this flipping threshold moves up or down remains $-4$, as in
+ the unsplit factor graph: a value below $-4$ received in every iteration still replaces the
+ assignment, only after more iterations.% ORS-REV T4 clarified
```

#### 13. Extending to Larger Domains, first sentences

Lines v5 564 → working copy 560. Push `1a7c2c2` (2 Oct 11:21).

The theorems are for binary domains and are not extended; the rest of the paragraph is marked as informal, with a pointer to the measurements in Section 6.

```diff
- \paragraph{Extending to Larger Domains:} The above analysis considered an SCFG with binary
- constraints. However, a similar feedback loop accounts for the algorithm behavior when the variables
- have larger domains.
+ \paragraph{Extending to Larger Domains:} The above analysis considered binary domains, and we do not
+ extend the theorems beyond them. Informally, a similar feedback loop accounts for the algorithm's
+ behavior when the variables have larger domains, and Section~\ref{Sec:experiments} measures it
+ there.
```

### 5 Damping, With or Without

#### 14. Splitting with No Damping, the selection step

Lines v5 598 → working copy 594. Push `1a7c2c2` (2 Oct 11:21).

Section 5 said SyncBB; the experiments ran a centralized branch and bound. Now: a complete search, SyncBB as the distributed option, what was run in the experiments.

```diff
- To validate this hypothesis, we propose two versions of the algorithm that take advantage of our
- observation on the behavior of the algorithm in the lemniscate example (Figure~\ref{Fig:scfg}).
- Given a factor graph, we generate its SCFG and perform a number of MS iterations that allow the
- algorithm to reach some iteration $t$ where it already alternates between two solutions (400 in our
- experiments). We take the assignments selected by the algorithm in the iterations, $t-2$ and $t$
- (398 and 400 in our case), and generate a new problem including only these two values in each domain
- variable. Then, we use either MGM \cite{MaheswaranTBPV04} or SyncBB \cite{HirayamaY97} to select the
- solution of this reduced problem.
+ To validate this hypothesis, we propose two versions of the algorithm that take advantage of our
+ observation on the behavior of the algorithm in the lemniscate example (Figure~\ref{Fig:scfg}).
+ Given a factor graph, we generate its SCFG and perform a number of MS iterations that allow the
+ algorithm to reach some iteration $t$ where it already alternates between two solutions (400 in our
+ experiments). We take the assignments selected by the algorithm in the iterations, $t-2$ and $t$
+ (398 and 400 in our case), and generate a new problem including only these two values in each domain
+ variable. Then, we use either MGM \cite{MaheswaranTBPV04} or a complete search to select the
+ solution of this reduced problem; a distributed algorithm such as SyncBB \cite{HirayamaY97} can be
+ used, and in our experiments we used a centralized branch and bound search
+ (Section~\ref{Sec:experiments}).
```

#### 15. Splitting with Damping, the paragraph that cites the theorems

Lines v5 608–609 → working copy 604–605. Push `1a7c2c2` (2 Oct 11:21).

Commas in the sentence that cites the theorems; the caveat that the theorems are for λ = 0 (which was commented out) placed before the informal damping sentence; the typo "alternatives.By".

```diff
- That being said, it seems that with respect to SCFGs damping provides another advantage. In
- Section~\ref{sec:split}, we show that when solving a cycle generated by a split MS-SCFG converges to
- one of its bounds if the messages that enter it satisfy one of the two sets of conditions in
- Theorem~\ref{theo:ft2} for enough consecutive iterations. By Theorem~\ref{theo:conv}, the cycle then
- remains at this bound as long as the messages received from outside the cycle satisfy the
- corresponding persistence condition. %These conditions are sufficient, not necessary, and the
- theorems are stated for $\lambda = 0$; we do not extend them to damping, whose effect on the
- feedback loop is shown numerically in Figure~\ref{Fig:split_tail_unary17_it200} in the supplementary
- material. When the entering messages keep jumping from one side of the threshold to the other,
- neither theorem applies, and the cycle may never settle at a bounder.
- Otherwise, they may jump back and forth and prevent the feedback loop from generating the separation
- between the minimal assignment selection and its alternatives.By giving most of the weight to the
- message that was sent in the previous iteration, damping makes the changes in $\Delta$ between
- consecutive messages more gradual, thus preventing erratic behavior.
+ That being said, it seems that with respect to SCFGs damping provides another advantage. In
+ Section~\ref{sec:split}, we show that, when solving a cycle generated by a split, MS-SCFG converges
+ to one of its bounds if the messages that enter it satisfy one of the two sets of conditions in
+ Theorem~\ref{theo:ft2} for enough consecutive iterations. By Theorem~\ref{theo:conv}, the cycle then
+ remains at this bound as long as the messages received from outside the cycle satisfy the
+ corresponding persistence condition. %These conditions are sufficient, not necessary, and the
+ theorems are stated for $\lambda = 0$; we do not extend them to damping, whose effect on the
+ feedback loop is shown numerically in Figure~\ref{Fig:split_tail_unary17_it200} in the supplementary
+ material. When the entering messages keep jumping from one side of the threshold to the other,
+ neither theorem applies, and the cycle may never settle at a bounder.
+ Otherwise, they may jump back and forth and prevent the feedback loop from generating the separation
+ between the minimal assignment selection and its alternatives. These conditions are sufficient, not
+ necessary, and the theorems are stated for $\lambda = 0$; we do not extend them to damping. By
+ giving most of the weight to the message that was sent in the previous iteration, damping makes the
+ changes in $\Delta$ between consecutive messages more gradual, thus preventing erratic behavior.
```

#### 16. Splitting with Damping, Inter-DMS

Lines v5 614 → working copy 610. Push `1a7c2c2` (2 Oct 11:21).

"MGM or SyncBB" becomes "MGM or the search described above".

```diff
- When the last version to run was MS-split (with no damping) we selected the assignments of two
- iterations and generated a problem that was solved by MGM or SyncBB as described above. In some
- cases, we used the anytime mechanism proposed in~\cite{ZivanOP14} to select the best beliefs to be
- used in the splitting point.
+ When the last version to run was MS-split (with no damping) we selected the assignments of two
+ iterations and generated a problem that was solved by MGM or by the search described above. In some
+ cases, we used the anytime mechanism proposed in~\cite{ZivanOP14} to select the best beliefs to be
+ used in the splitting point.
```

### 6 Experimental Evaluation

#### 17. Figure 3 caption

Lines v5 628 → working copy 624. Push `a54f9fb` (2 Oct 11:08).

MS-SCFG-opt is drawn at iteration 400; the time of its search is not included.

```diff
- \caption{Mean solution cost per iteration when solving the random sparse, random dense and scale
- free problems.}
+ \caption{Mean solution cost per iteration when solving the random sparse, random dense and scale
+ free problems. MS-SCFG-opt is drawn at iteration 400; the time of its search is not included.}
```

#### 18. Algorithms, DABP

Lines v5 639 → working copy 635. Push `a54f9fb` (2 Oct 11:08).

How DABP is put on the iteration axis: the measured time factors 6.9, 3.3, 5.6, 6.3 and 8.2 per benchmark.

```diff
- The versions that combine algorithms are the orders of Inter-DMS (Section~\ref{Sec:with}) that we
- ran: MS-SCFG followed by a search over the two selected assignments, and DMS followed by DMS-SCFG,
- with the split either at a fixed iteration or at the best state that the anytime mechanism recorded.
- MS-SCFG and DMS-SCFG are the MS-split and DMS-split of Section~\ref{Sec:with}. \textbf{MS}, standard
- Min-sum on the original factor graph. \textbf{MS-SCFG}, Min-sum on the symmetric split constraint
- factor graph, in which every function-node is replaced by two copies holding half of its cost table.
- \textbf{MS-SCFG-MGM} and \textbf{MS-SCFG-opt} (Section~\ref{Sec:MS-split}): MS-SCFG runs for 400
- iterations, the two assignments selected in iterations 398 and 400 define a problem with two values
- per variable, and this problem is solved by MGM-1 \cite{MaheswaranTBPV04}, run from each of the two
- assignments until no agent can improve, keeping the better result, or by a centralized branch and
- bound search over the reduced problem, warm-started from the MGM solution, with a time cap of 300
- seconds per instance. The search completed on every instance of random sparse, graph coloring and
- meeting scheduling and on 48 of the 50 scale free instances; on random dense it hit the cap on 48
- instances, where we report the best solution found within it. On one instance per benchmark the
- search took 0.4 to 300 seconds, as long as about 250 to 37{,}700 iterations of DMS, and MGM as long
- as 4 to 9. \textbf{DMS}, damped Min-sum on the original factor graph. \textbf{DMS-SCFG}, damped Min-
- sum on the symmetric SCFG. \textbf{DMS-$k$DS} (Section~\ref{Sec:with}): DMS on the original factor
- graph for $k$ iterations, then the graph is converted to an SCFG as described in
- Section~\ref{Sec:with}. \textbf{DABP} \cite{DengKL022}, Deep Attentive Belief Propagation, which
- learns damping factors and message weights on an SCFG with a $0.95/0.05$ split. DABP is centralized,
- trained, and its iterations require neural nets and a GPU; to plot it on the iteration axis we use
- the simulated runtime method \cite{SultanikLR08}, with times measured on an NVIDIA RTX 4090. We
- include it as a reference for a strong learned centralized version and do not analyze it further.
+ The versions that combine algorithms are the orders of Inter-DMS (Section~\ref{Sec:with}) that we
+ ran: MS-SCFG followed by a search over the two selected assignments, and DMS followed by DMS-SCFG,
+ with the split either at a fixed iteration or at the best state that the anytime mechanism recorded.
+ MS-SCFG and DMS-SCFG are the MS-split and DMS-split of Section~\ref{Sec:with}. \textbf{MS}, standard
+ Min-sum on the original factor graph. \textbf{MS-SCFG}, Min-sum on the symmetric split constraint
+ factor graph, in which every function-node is replaced by two copies holding half of its cost table.
+ \textbf{MS-SCFG-MGM} and \textbf{MS-SCFG-opt} (Section~\ref{Sec:MS-split}): MS-SCFG runs for 400
+ iterations, the two assignments selected in iterations 398 and 400 define a problem with two values
+ per variable, and this problem is solved by MGM-1 \cite{MaheswaranTBPV04}, run from each of the two
+ assignments until no agent can improve, keeping the better result, or by a centralized branch and
+ bound search over the reduced problem, warm-started from the MGM solution, with a time cap of 300
+ seconds per instance. The search completed on every instance of random sparse, graph coloring and
+ meeting scheduling and on 48 of the 50 scale free instances; on random dense it hit the cap on 48
+ instances, where we report the best solution found within it. On one instance per benchmark the
+ search took 0.4 to 300 seconds, as long as about 250 to 37{,}700 iterations of DMS, and MGM as long
+ as 4 to 9. \textbf{DMS}, damped Min-sum on the original factor graph. \textbf{DMS-SCFG}, damped Min-
+ sum on the symmetric SCFG. \textbf{DMS-$k$DS} (Section~\ref{Sec:with}): DMS on the original factor
+ graph for $k$ iterations, then the graph is converted to an SCFG as described in
+ Section~\ref{Sec:with}. \textbf{DABP} \cite{DengKL022}, Deep Attentive Belief Propagation, which
+ learns damping factors and message weights on an SCFG with a $0.95/0.05$ split. DABP is centralized,
+ trained, and its iterations require neural nets and a GPU; to plot it on the iteration axis we use
+ the simulated runtime method \cite{SultanikLR08}: with times measured on an NVIDIA RTX 4090, an
+ iteration of DABP takes 6.9, 3.3, 5.6, 6.3 and 8.2 times as long as an iteration of DMS on the five
+ benchmarks (in the order above), and the DABP curve is stretched by these factors. We include it as
+ a reference for a strong learned centralized version and do not analyze it further.
```

#### 19. Figure 4 caption

Lines v5 646 → working copy 642. Push `a54f9fb` (2 Oct 11:08).

Same note as in the Figure 3 caption.

```diff
- \caption{Mean solution cost per iteration when solving Graph coloring (left) and Meeting scheduling
- (right) problems.}
+ \caption{Mean solution cost per iteration when solving Graph coloring (left) and Meeting scheduling
+ (right) problems. MS-SCFG-opt is drawn at iteration 400; the time of its search is not included.}
```

#### 20. Splitting in the middle of a run

Lines v5 669 → working copy 665. Push `a54f9fb` (2 Oct 11:08).

k = 1000 is the same value on every benchmark; no k was selected per benchmark; every k ran on the same 50 instances.

```diff
- The blue line in Figure~\ref{Fig:bound} is DMS-$k$DS with $k = 1000$. Before the split it follows
- DMS, with 0.08, 0.61, 0.17, 0.01 and 0.05 of its messages at a bound; 100 iterations after the split
- 0.75, 0.94, 0.76, 0.66 and 0.56 are at a bound, more than DMS-SCFG has 100 iterations after the
- start, and the runs converge a median of 46, 32, 50, 118 and 134 iterations after the split. At
- every $k$ we tested the split takes effect as fast as at the start: 200 iterations after it, the
- fraction at a bound is within 0.01 of the fraction DMS-SCFG has 200 iterations after the start, or
- higher. At the split the damping history is cleared, so the first messages after it are not damped.
- Clearing the history without a split does not cause this change: in a control run that clears it at
- $k = 1000$, 100 iterations later 0.08, 0.61, 0.18, 0.01 and 0.05 of the messages are at a bound, as
- in DMS, and the final cost is not lower than that of DMS.
+ The blue line in Figure~\ref{Fig:bound} is DMS-$k$DS with $k = 1000$, the same value on every
+ benchmark; no $k$ was selected per benchmark, and every $k$ was run on the same 50 instances. Before
+ the split it follows DMS, with 0.08, 0.61, 0.17, 0.01 and 0.05 of its messages at a bound; 100
+ iterations after the split 0.75, 0.94, 0.76, 0.66 and 0.56 are at a bound, more than DMS-SCFG has
+ 100 iterations after the start, and the runs converge a median of 46, 32, 50, 118 and 134 iterations
+ after the split. At every $k$ we tested the split takes effect as fast as at the start: 200
+ iterations after it, the fraction at a bound is within 0.01 of the fraction DMS-SCFG has 200
+ iterations after the start, or higher. At the split the damping history is cleared, so the first
+ messages after it are not damped. Clearing the history without a split does not cause this change:
+ in a control run that clears it at $k = 1000$, 100 iterations later 0.08, 0.61, 0.18, 0.01 and 0.05
+ of the messages are at a bound, as in DMS, and the final cost is not lower than that of DMS.
```

### 7 Conclusions

#### 21. Conclusions, first paragraph

Lines v5 704 → working copy 700. Push `1a7c2c2` (2 Oct 11:21).

"We prove, for a single split cycle"; the ungrammatical sentence about "converges to more than one solution" becomes "alternates between two solutions", with "to the best of our knowledge".

```diff
- The main contribution of our work is a theoretical analysis of the dramatic effect of function-node
- splitting on the convergence of Min-sum. We prove that the feedback loops generated by the split
- cause separation between the minimal belief that the algorithm propagates and its alternatives. We
- further identify a property that was not reported before, that when Min-sum is applied without
- damping, it often converges to more than one solution, which the algorithm not only oscillates
- between them, but also has different nodes of the graph consider different solutions at the same
- iteration. This results in low quality solutions.
+ The main contribution of our work is a theoretical analysis of the dramatic effect of function-node
+ splitting on the convergence of Min-sum. We prove, for a single split cycle, that the feedback loops
+ generated by the split cause separation between the minimal belief that the algorithm propagates and
+ its alternatives. We further identify a property that, to the best of our knowledge, was not
+ reported before: when Min-sum is applied without damping, it often alternates between two solutions,
+ and different nodes of the graph follow different solutions at the same iteration. This results in
+ low quality solutions.
```

#### 22. Conclusions, second paragraph

Lines v5 706 → working copy 702. Push `1a7c2c2` (2 Oct 11:21).

"An advantage over previous versions, especially in distributed settings" becomes what Section 6 shows: delayed split lower on random constraints; selection recovers most of the gap.

```diff
- We use this understanding of the algorithm's behavior to propose two types of heuristics. The first:
- delay splitting until DMS identifies better solution candidates. The second: use MS-split in a
- limited number of iterations, to identify two alternatives in each domain, and use distributed
- search algorithms to select the better assignment among them. These heuristics were found to have an
- advantage over previous versions of the algorithm, especially in distributed settings. We hope in
- the future to investigate further more heuristics that stem from this novel understanding of the
- algorithm.
+ We use this understanding of the algorithm's behavior to propose two types of heuristics. The first:
+ delay splitting until DMS identifies better solution candidates. The second: use MS-split in a
+ limited number of iterations, to identify two alternatives in each domain, and use distributed
+ search algorithms to select the better assignment among them. Delaying the split lowers the final
+ cost on problems with random constraints, and selecting between the two alternatives recovers most
+ of the gap between undamped and damped split Min-sum. We hope in the future to investigate further
+ more heuristics that stem from this novel understanding of the algorithm.
```

### Appendix: Proofs

#### 23. Proof of Theorem 4.10, last paragraph

Lines v5 1010–1014 → working copy 1006. Push `766140c` (2 Oct 10:55).

The proof now starts the induction from the four window iterations and uses the value received at t−2; tightness via monotonicity of the clipped update.

```diff
- Applying the appropriate equivalence successively to the sequences generated
- by the first four messages proves persistence by induction over complete
- return passes. Because
- both statements are equivalences, a strict violation moves the next message
- away from the relevant bound, which proves tightness.
+ A message $\Delta_{R^{'t}_j}$ with $t \geq \hat{t}$ is the result of a complete pass that starts
+ from the message sent by $F'_{ij}$ at iteration $t-4$ and uses the value $\Delta_{\bar R_i^{t-2}}$
+ received at iteration $t-2 \geq \hat{t}-2$. Starting from the four iterations before $\hat{t}$ and
+ applying the appropriate equivalence to each pass proves persistence by induction. For tightness,
+ note that the right-hand side of Equation~\eqref{eq:pass} is non-decreasing in $r$ and that every
+ difference sent to $X_j$ lies between $L$ and $U$. A value that violates the inequality therefore
+ gives, two iterations after it is received, a message below $U$ (above $L$ in the second case),
+ whatever the message at the start of the pass was.
```

### Appendix: Additional Experimental Results

#### 24. The text that describes Figure 10

Lines v5 1053 → working copy 1045. Push `1a7c2c2` (2 Oct 11:21).

Describes the new figure: the difference between the beliefs of X_1, the −17 against the first flipping threshold −16, why damping delays the reversal.

```diff
-  Figure~\ref{Fig:split_tail_unary17_it200} returns to the setting of the example presented in
- Section 4, of a factor graph with a single function-node, where each variable has only 2 possible
- values, before and after splitting. We present the differences between the beliefs of the outgoing
- message ($\Delta_R$). In this case, we have a single unary constraint that breaches $FT$. However,
- in DMS versions it takes time until the unary constraint develops, and thus we see a change in trend
- during the run. It further demonstrates the erratic behavior of the algorithm when there is no
- damping and that, with splitting, the difference between beliefs in the outgoing messages grows to a
- much larger value, despite the optimization solution itself (which variable value minimizes cost)
- remaining the same.
+  Figure~\ref{Fig:split_tail_unary17_it200} returns to the setting of the example presented in
+ Section 4, of a factor graph with a single function-node, where each variable has only 2 possible
+ values, before and after splitting. We present the difference between the beliefs of $X_1$ for its
+ two values. The constant outside message, $-17$, lies below the first flipping threshold $2(M_a-
+ M_b)=-16$, so it eventually reverses the assignment that the cost table alone selects; with damping
+ the message builds up gradually, so the reversal takes longer. The figure further demonstrates the
+ erratic behavior of the algorithm when there is no damping and that, with splitting, the difference
+ between the beliefs grows to a much larger value, despite the optimization solution itself (which
+ variable value minimizes cost) remaining the same.
```

#### 25. Figure 10, file and caption

Lines v5 1124–1126 → working copy 1116–1117. Push `505770b` (2 Oct 10:47).

New file s6_unary17.pdf (damping on variable-to-function messages, 400 iterations) and a caption that prints the table, the mapping X_i = X_2, X_j = X_1, and the damping.

```diff
- \includegraphics[width=.85\textwidth]{split_tail_unary17_it200.pdf}
- \caption{The difference in the belief-cost ($\Delta_R$) for the 2 possible values of $X_1$
- ($D_{X_{1}}=\{a,b\}$). We ran over 200 Min-sum (MS) iterations on a problem similar to the example
- from Section 4 with costs $M_a = 8$ and $M_b = 0$, and a single unary constraint of $[0,17]$. We
- compared the unsplit factor graph and a symmetric split ($s=0.5$) factor graph using undamped and
- damped min-sum (with $\lambda\in\{0,0.5,0.9\}$). A positive difference here favors value $a$ , the
- lower-cost assignment, i.e., $\Delta_{R_j} = R_{j_b} - R_{j_a}$. The undamped and $\lambda=0.5$
- unsplit cost differences approach the small $+1$ cost difference, while the corresponding values
- including split reach a much larger internal margin of about $24$. \\
- For $\lambda=0.9$ both trajectories have not settled by iteration 200; the split cost-difference has
- only reached about $11$. Splitting introduces large oscillations without damping, while damping
- suppresses oscillations and slows the dynamics. Note that both the split and unsplit cases share the
- same goal (minimizing the same costs), so the change is an internal reparameterization effect.}
+ \includegraphics[width=.85\textwidth]{s6_unary17.pdf}
+ \caption{The difference between the beliefs of $X_1$ for its two values, $b_{X_1}(a) - b_{X_1}(b)$,
+ over 400 iterations on a single function-node $F_{12}$ between $X_1$ and $X_2$, where $X_2$ receives
+ a constant outside message with cost $17$ for $a$ and $0$ for $b$. Each copy of $F_{12}$ holds the
+ table $\bigl(\begin{smallmatrix} & X_2{=}a & X_2{=}b \\ X_1{=}a & 0 & 20 \\ X_1{=}b & 30 & 8
+ \end{smallmatrix}\bigr)$, and the unsplit table twice these values; in the notation of
+ Section~\ref{sec:split}, with $X_i = X_2$ and $X_j = X_1$, this is $M_a = 0$, $M_b = 8$, $B_a = 20$,
+ $B_b = 30$ and $\Delta_{\bar R_i} = -17$, and the plotted quantity is the sum over the two copies of
+ $-\Delta_{R_j}$. We compare the unsplit factor graph and its symmetric split ($s=0.5$) with
+ $\lambda\in\{0,0.5,0.9\}$, damping applied to the variable-to-function messages as in all our
+ experiments. Under this outside message $b$ is the lower-cost value of both variables, so a positive
+ difference favors the optimal assignment. Without the split the difference rises from $-16$ to $+1$,
+ the margin of the original table; with the split it reaches $24$, since the message of each copy is
+ clipped at its lower bound $M_b - B_a = -12$ (Equation~\eqref{eq:pass}). For $\lambda=0.9$ neither
+ run has settled by iteration 400; the split difference has reached about $12$. Splitting introduces
+ large oscillations without damping, damping suppresses them and slows the dynamics, and both graphs
+ minimize the same cost, so the change is an internal effect of the split.}
```

## Also added to the Overleaf project

- `s6_unary17.pdf`: the new Figure 10 (damping on the variable-to-function messages, 400 iterations), pushed in `505770b`.

## Not in the working copy (from the optional list)

- The footnote explaining the $(B_b-M_b)+2d$ term in Theorem 4.9.
- The Background sentence that damping is applied only at variable-nodes.
- The duplicate label of Definition 4.2.
