# Review of v5_ors_rev against the AAAI-27 reviews

| | |
|---|---|
| **Decision** | Reject, as it stands |
| **Recommendation** | Weak Reject |
| **Overall score** | 4/10, notable flaws outweigh the merits as the file stands |
| **Reviewer confidence** | 4/5 (theorem claims checked by exact simulation; the benchmark runs were not repeated) |

Reviewed file: `min-sum_split_AAMAS2027_v5_ors_rev.tex`, a byte copy of Roie's `min-sum_split_AAMAS2027_v5.tex` at Overleaf commit `c06e55d` (his push of 2026-10-01 13:46). The copy was pushed as `da35c7b` and v5 itself is unchanged. Line numbers below are lines of that file. Date of the review: 2026-10-01.

## What v5 is, and how I checked it

- v5 is the text I pushed in `11932df` (Section 6 and everything outside Section 4) plus Roie's rewrite of Theorems 4.9 and 4.10 and of the example. The two files differ in 17 hunks, all in L461–560. His second push (13:46) changed the converse of Theorem 4.10 from `t+4` to `t+2` and rewrote the example paragraph.
- The file compiles with tectonic: 15 pages, 0 errors. The main text ends on page 9, so it is about one page over the AAMAS limit of eight.
- Theorem checks use exact rational min-sum on the paper's own schedule (every node sends once per iteration from what it received in the previous one, no damping), on random tables with M_a = 0 < M_b < B_b < B_a. About 3,900 tables per check. A value `Δ_R̄_i^t` received at iteration `t` acts on the messages to X_j at iteration `t+2` (Definition 4.7 of the paper says the same).
- The AAAI reviews were read in full from the OpenReview paste of 2026-09-25 (jFq3 rating 4, XqaZ rating 5, and the AI review). The paste lost the formulas, so for the theorem points I rely on the surrounding words and on my earlier exact checks of the lemmas.

## 1. Theorem 4.10 and its example in Roie's version

**Verdict: not correct yet.** The converse is right now. The persistence half is false as worded, and three things around it do not match it.

What is right:
- The thresholds `max{−2d, (B_b−B_a)−d}` and `min{−2d, (B_b−B_a)−d}`.
- The converse as he now states it: a violating value received at `t` pushes `Δ_R'^{t+2}_j` below (above, for the lower half) the bound. It held in 3,941 of 3,941 upper-half tests and in 3,921 of 3,921 lower-half tests, whatever the other values were.
- Every number in the example (below).

What is wrong:

**(a) The persistence half leaves out the values that are still on their way.** The message at iteration `t` is computed from the message of the same copy at `t−4` and the value received at `t−2`. His window is the four iterations `t̂−4, …, t̂−1`, and his condition covers only values received after `t̂`. So the messages at `t̂` and `t̂+1` depend on values received at `t̂−2` and `t̂−1`, and the message at `t̂+2` on the value received at `t̂`. Nothing checks them.

The paper's own example breaks the claim. Table `M_a = 0, M_b = 2, B_a = 30, B_b = 20` (threshold −4). The messages `Δ_R'_j` are 20 in iterations 21 to 24, so `t̂ = 25`. The outside value is 0 at every iteration, except −25 received at iteration 24. Every value received after `t̂` is 0, which is at least −4, so the hypothesis holds. The theorem says `Δ_R'^t_j = 20` for all `t > 25`. In fact:

```
iteration  : 21  22  23  24  25  26  27  28  29  30
Δ R'_j     : 20  20  20  20  20  -1  20  20  20   3
```

| Check (exact simulation, copy F′ only, values at `t̂−2 … t̂` free) | Result |
|---|---|
| Upper half, his condition (values after `t̂`) | the claim fails in **3,898 of 3,941** tables; every failure has a violating value among those received at `t̂−2, t̂−1, t̂` |
| Upper half, failures where only the value at `t̂−2` violates | 120. So starting the condition at `t̂−1` is still not enough |
| Lower half, his condition | fails in **3,853 of 3,921** tables |
| Condition for every `t ≥ t̂−2`, claim for every `t ≥ t̂` (fix A1–A4) | **0 failures** in 3,941 and in 3,921 |
| Converse at `t+2`, any other values | 0 failures in 3,941 and in 3,921 |

**(b) The paragraph after the theorem still says four iterations** (L531: "four iterations following a violation"). It contradicts the theorem it describes. Fix A5.

**(c) The proof in the appendix was not updated** (L1010–1014). It still talks about "the first four messages" and "the next message". It cannot prove the new statement, and the new statement is false as worded. Fix A8.

**(d) The lower half is worded differently from the upper half** ("at iteration t̂ the differences of four consecutive messages … were"), and has no comparison sign (`\>` is a space in math mode, so the printed formula has none; L519). Fix A3, A4.

**(e) A sentence breaks off before the example** (L535: "Note that the difference between a value of Δ received in a", and the rest is commented out). Fix A6.

**(f) The example.** The numbers are right: `Δ_R'^t_j` is 2, 6, 10, 14, 18, 20 at iterations 1, 5, 9, 13, 17, 21; `Δ_R''^{21}_i` is 22; the flipping threshold goes −4, −8, −12, −16, −20, −24 at iterations 1, 5, …, 21 and stays −24. The new closing sentences are the problem:
- "increases the absolute difference required to replace the assignment of X_j from 4 to 24" is true only for a **single** outside message. A repeated value does replace it. A value of −5 received at every iteration from iteration 21 on makes X_j select b at iteration 103, −10 at iteration 35, −30 at iteration 23; −4 does not (400 iterations checked). The old text said "single" and "repeated"; the rewrite lost both words, and a constant outside value is exactly what reviewer jFq3's counterexample used.
- "Theorem 4.10 establishes that it [the flipping threshold] remains −24" and "Thus, when Δ < −4 it grows" are links that the theorem does not make. Theorem 4.10 is about the messages. The flipping threshold follows from them through `FT = −d − s` (proof of Lemma 4.8), and "it grows" is Lemma 4.8 for a value repeated in every iteration. Fix A7.

**(g) Theorem 4.9, next to it.** Three problems:
- It ends "Thus, the algorithm reaches both difference and assignment convergence". With the outside value alternating −13, −10 on the table `M_a = 0, M_b = 7, B_a = 26, B_b = 24` (both values satisfy `Δ + 2d ≥ ε` with ε = 1), the messages to X_j alternate 23, 20, 23, 20 forever after the 128 iterations the theorem allows. X_j selects a, but the differences do not converge. Fix A9.
- `t̂` appears in the second half ("for every iteration t > t̂") and is not defined there. The extra condition has to hold from the start of the climb. Fix A10, A11.
- `(B_b − M_b) + 2d` is not explained (XqaZ asked). It equals `(B_b − M_a) + d`. Fix A12.

## 2. The AAAI reviews, point by point

24 points. **7 fixed, 7 partly, 9 open, 1 not applicable.** The fix names refer to section 3.

| # | AAAI point | Who | In v5_ors_rev | Status | Fix |
|---|---|---|---|---|---|
| 1 | Lemma 4.6 and Theorems 4.7–4.8 incorrect as written; counterexample with a constant outside value | jFq3 W1 (rebuttal 1), AI W1 | Shir's rewrite fixed these (exact simulation, earlier): the Lemma 4.8 inequality, the limits of Lemma 4.6, the persistence thresholds. Roie's rewrite then made Theorem 4.10 false as worded (section 1) and ended Theorem 4.9 with a false claim. | PARTLY | A1–A13 |
| 2 | The damping argument in Section 5 rests on those results | jFq3 W1, XqaZ Q4 | The caveat that the theorems are for λ = 0 is commented out (L608), so Section 5 reads as if they explained damping. | OPEN | C1–C3 |
| 3 | Larger-domain claims | jFq3 W1 | L564–566 still state the larger-domain picture as established. | OPEN | C6 |
| 4 | 'Opt merge': SyncBnB versus the artifact's centralized search; completion, time, gap, label | jFq3 W2 | Section 6 now says centralized branch and bound, 300 s cap, complete on all but 48 random dense and 2 scale free instances, and gives the search time. Section 5 still says SyncBB (L598, L614). The label is still 'opt'. No optimality gap. Drawn at iteration 400. | PARTLY | C4, C5, D3, D4 |
| 5 | 'Every K improves' (DMS@600 on random sparse) | jFq3 W3, rebuttal 3 | Table 1 reports every k with signs and p-values, and the sentence next to it matches the table. | FIXED | — |
| 6 | Convergence not shown by the mean cost; residuals, assignment-change rates, stable-run fractions | jFq3 W4 | Section 6 reports converged runs and medians, the final message change (below 1e-6 in 48, 49, 48, 26 and 37 of 50 DMS-SCFG runs) and the runs with no assignment change. | FIXED | — |
| 7 | How far the theory generalizes (one binary cycle versus the full graphs) | XqaZ W1, Q2 | The Discussion of Section 6 says the theory covers a single cycle without damping. Section 4, the abstract and the introduction do not say it. | PARTLY | B2, B4, E1 |
| 8 | What 'convergence' means; structured benchmarks still oscillate; title and abstract unqualified | XqaZ W2, Q1; AI W2 | Roie added Definitions 4.1 and 4.2, and Section 6 defines 'converged' for the cost. The abstract and the Conclusions still say converge without scope, and Theorem 4.9 now claims convergence of the algorithm. | PARTLY | A9, A13, B2, E1 |
| 9 | (B_b − M_a) in one theorem, (B_b − M_b) in the other | XqaZ W3, Q3 | Theorem 4.9 uses (B_b − M_b) + 2d with no explanation in the main text. | OPEN | A12 |
| 10 | Notation heavy; a notation table or a clearer running example | XqaZ W4 | No table. The running example exists, but its closing sentences are muddled (single versus repeated value). | OPEN | A7 |
| 11 | How k was chosen; same instances for tuning and evaluation | XqaZ W5, Q5 | Every k is reported and k = 1000 is drawn for every benchmark, but the text never says that nothing was tuned. | PARTLY | D6 |
| 12 | Speed claim on an iteration axis; how DABP time is converted; time-to-quality | XqaZ W6, Q6; AI W6 | Section 6 says 'times measured on an NVIDIA RTX 4090' and nothing else: no ratios, no cost of an SCFG iteration. | OPEN | D1, D5 |
| 13 | DABP and DABP-NoSplit identical apart from the split? | XqaZ Q7 | Not stated. Figures 3 and 4 draw DABP on all five benchmarks, and three of them are June runs on other instances. | OPEN | D2 and the figure decision (section 4) |
| 14 | The recurrence needs more than the stated ordering; piecewise derivation | AI W3 | Shir's one-pass recurrence with clipping, with its assumptions stated (Equation 1 and the appendix). | FIXED | — |
| 15 | Novelty: Zivan et al. 2020 not cited; the 'no explanation' claim | AI W4 | Still not cited. The abstract (L117), the introduction (L161) and the related work (L174) still say no explanation exists or call the success unexplainable. | OPEN | B1, B3, B5, B6 |
| 16 | Postprocessing gain not attributed to the oscillation-derived candidates; controls | AI W5 | No controls. The text now says 'good candidate set' and that the search stays behind DMS-SCFG, which is accurate but not tested. | OPEN (needs runs) | section 4 |
| 17 | No common resource measure | AI W6 | The search time is stated (0.4 to 300 s, about 250 to 37,700 DMS iterations). The cost of an SCFG iteration and the DABP stretch are not. MS-SCFG-opt is drawn at iteration 400. | PARTLY | D1, D3–D5 |
| 18 | Define the split formally in the Background, with the sum property | AI suggestion | No definition; SCFG is first used at L161. | OPEN | B7 |
| 19 | Observation 4.1 uses a primed message in the unsplit graph | AI minor | Unprimed now (L304). | FIXED | — |
| 20 | What happens to the messages and the damping history at the split | AI minor | Section 5: each copy gets half of the old message. Section 6: the history is cleared, so the first messages are undamped. | FIXED | — |
| 21 | Normalization constant; invariance of the differences | AI minor | α in Section 3 and the footnote of Definition 4.1. | FIXED | — |
| 22 | Labels @200 versus @400 | AI minor | New legends (MS-SCFG-opt); the text says iterations 398 and 400. | FIXED | — |
| 23 | SyncBB and SyncBnB for the same method | AI minor | SyncBnB is gone; SyncBB in Section 5 now disagrees with the centralized search of Section 6. | PARTLY | C4, C5 |
| 24 | Adaptive factor-wise splitting policy | AI suggestion | Not in the paper. | n/a (an optional future-work sentence) | — |

Two of the open points are what ended the AAAI paper: reviewers jFq3 and the AI review wrote that the theorems are "incorrect as written", and the AI review wrote that the paper does not cite Zivan et al. (2020), Beyond Trees, while saying no explanation exists. In v5 the first is back in a new form (Theorem 4.10, Theorem 4.9) and the second is untouched.

## 3. Exact fixes

34 fixes. Every OLD text occurs exactly once in `v5_ors_rev`. All of them applied together compile (0 errors, no undefined citations). Nothing has been applied to Overleaf. Line numbers are those of the file as it is now.

### A. Theorem 4.10, its example and Theorem 4.9 (Roie's rewrite)

**A1. Theorem 4.10, upper half: the condition has to start two iterations before the window ends** (line 504)

```diff
- Assume that iteration $\hat{t}$ follows four consecutive iterations in which $\Delta_{R^{'t}_j} = B_b-M_a$. If for every $t > \hat{t}$,
+ Assume that iteration $\hat{t}$ follows four consecutive iterations in which $\Delta_{R^{'t}_j} = B_b-M_a$. If for every $t \geq \hat{t}-2$,
```


**A2. Theorem 4.10, upper half: conclusion and converse** (lines 509–510)

```diff
- then for all such $t$, $\Delta_{R^{'t}_j} = B_b-M_a$.
- In contrast, if at some iteration $t > \hat{t}$, $\Delta_{\bar R_i^t}< -\min\{2d,\;(B_a-B_b)+d\}$, then
+ then for every $t \geq \hat{t}$, $\Delta_{R^{'t}_j} = B_b-M_a$.
+ In contrast, if at some iteration $t \geq \hat{t}-2$, $\Delta_{\bar R_i^t}< -\min\{2d,\;(B_a-B_b)+d\}$, then
```


**A3. Theorem 4.10, lower half: same window wording as the upper half, same start of the condition** (lines 512–513)

```diff
- Similarly, assume that at iteration $\hat{t}$ the differences of four consecutive messages sent to
- $X_j$ were all equal to the lower bound $M_b-B_a$. If for every $t > \hat{t}$,
+ Similarly, assume that iteration $\hat{t}$ follows four consecutive iterations in which $\Delta_{R^{'t}_j} = M_b-B_a$. If for every $t \geq \hat{t}-2$,
```


**A4. Theorem 4.10, lower half: conclusion, converse, and the missing greater-than sign** (lines 518–519)

```diff
- then for all such $t$, $\Delta_{R^{'t}_j} = M_b-B_a$.
- In contrast, if at some iteration $t > \hat{t}$ $\Delta_{\bar R_i^t}\> -\max\{2d,\;(B_a-B_b)+d\}$, then
+ then for every $t \geq \hat{t}$, $\Delta_{R^{'t}_j} = M_b-B_a$.
+ In contrast, if at some iteration $t \geq \hat{t}-2$, $\Delta_{\bar R_i^t} > -\max\{2d,\;(B_a-B_b)+d\}$, then
```

Note: in the source, \> is a medium space in math mode, so the printed formula has no comparison sign.


**A5. The paragraph after Theorem 4.10 still says four iterations** (lines 531–532)

```diff
- inequality, $\Delta_{R_j}$ remains at the bound; four iterations following a violation, the
- corresponding difference sent to $X_j$ moves away from the bound.
+ inequality, $\Delta_{R_j}$ remains at the bound; two iterations after a violating value is received, the
+ corresponding difference sent to $X_j$ moves away from the bound. A value received at iteration $t$ reaches the messages sent to $X_j$ at iteration $t+2$, which is why the conditions start at $\hat{t}-2$.
```


**A6. A sentence breaks off before the example (the rest of it is commented out)** (lines 535–536)

```diff
- Note that the difference between a value of $\Delta_{\bar R_i}$ received in a
- %single iteration
+ %single iteration
```


**A7. Example: the last sentences need 'single', and the link to Theorem 4.10 and Lemma 4.8 must be exact** (lines 557–560)

```diff
- Nevertheless, $FT_{F'_{ij}}$ can also increase.
- Theorem~\ref{theo:conv} establishes that it remains $-24$ when $\Delta_{\bar R_i} \geq -\min\{4,12\}=-4$.
- Thus, when $\Delta_{\bar R_i} < -4$ it grows.
- Thus, the feedback loop generated as a result of the split increases the absolute difference required to replace the assignment of $X_j$  from $4$ to $24$. However, the absolute difference required for increasing or decreasing this flipping threshold remains $4$ and is not affected by the feedback loop.% ORS-REV T4 clarified
+ Nevertheless, $FT_{F'_{ij}}$ can also increase. By Theorem~\ref{theo:conv}, the messages stay at the bound, and so $FT_{F'_{ij}}$ stays at $-24$, as long as every value of $\Delta_{\bar R_i}$ received is at least $-\min\{4,12\}=-4$. When a value below $-4$ is received in every iteration, $FT_{F'_{ij}}$ increases (Lemma~\ref{Lem:ft}).
+ Thus, the feedback loop generated as a result of the split increases the absolute difference that a \emph{single} outside message needs in order to replace the assignment of $X_j$ from $4$ to $24$. A repeated value below $-4$ still replaces it, as in the unsplit factor graph, only more slowly.% ORS-REV T4 clarified
```

Note: numbers in the example are right (checked by exact simulation); a repeated -5 from iteration 21 makes X_j select b at iteration 103, a repeated -10 at 35.


**A8. Appendix, end of the proof of Theorem 4.10: it still proves the old statement** (lines 1010–1014)

```diff
- Applying the appropriate equivalence successively to the sequences generated
- by the first four messages proves persistence by induction over complete
- return passes. Because
- both statements are equivalences, a strict violation moves the next message
- away from the relevant bound, which proves tightness.
+ A message $\Delta_{R^{'t}_j}$ with $t \geq \hat{t}$ is the result of a complete pass that starts from the message sent by $F'_{ij}$ at iteration $t-4$ and uses the value $\Delta_{\bar R_i^{t-2}}$ received at iteration $t-2 \geq \hat{t}-2$. Starting from the four iterations before $\hat{t}$ and applying the appropriate equivalence to each pass proves persistence by induction. For tightness, note that the right-hand side of Equation~\eqref{eq:pass} is non-decreasing in $r$ and that every difference sent to $X_j$ lies between $L$ and $U$. A value that violates the inequality therefore gives, two iterations after it is received, a message below $U$ (above $L$ in the second case), whatever the message at the start of the pass was.
```


**A9. Theorem 4.9: 'difference and assignment convergence' is false when the outside value keeps changing** (line 485)

```diff
- and remains there, and $X_j$ selects the value assignment $a$. Thus, the algorithm reaches both difference and assignment convergence.
+ and remains there, and $X_j$ selects the value assignment $a$ from then on.
```

Note: counterexample: M_a=0, M_b=7, B_a=26, B_b=24, outside value alternating -13, -10 (both satisfy the hypothesis): the messages to X_j alternate 23, 20, 23, 20 forever.


**A10. Theorem 4.9, upper half: $\hat{t}$ is not defined here** (lines 487–488)

```diff
- If, in addition, $\Delta_{\bar R_i^t}\geq(B_b-B_a)-d$ for every iteration $t > \hat{t}$, the bounder
- reached is $B_b$ and every difference sent to $X_j$ is equal to $B_b-M_a$.
+ If, in addition, $\Delta_{\bar R_i^t}\geq(B_b-B_a)-d$ for every iteration $t$, the bounder
+ reached is $B_b$ and, from then on, every difference sent to $X_j$ is equal to $B_b-M_a$.
```


**A11. Theorem 4.9, lower half: same** (line 493)

```diff
- In addition, if $\Delta_{\bar R_i^t}\leq(B_b-B_a)-d$ in every iteration $t > \hat{t}$, the bounder reached is $B_a$ and $\Delta_{R^{'t}_j} = M_b-B_a$.
+ In addition, if $\Delta_{\bar R_i^t}\leq(B_b-B_a)-d$ in every iteration $t$, the bounder reached is $B_a$ and, from then on, $\Delta_{R^{'t}_j} = M_b-B_a$.
```


**A12. Theorem 4.9: explain the $(B_b-M_b)+2d$ term (reviewer XqaZ asked whether it is intentional)** (line 484)

```diff
- iterations from initialization (recall that $4$ is the size of the cycle), the algorithm reaches one of its bounders
+ iterations from initialization (recall that $4$ is the size of the cycle),\footnote{The term $(B_b-M_b)+2d$ equals $(B_b-M_a)+d$, the distance from the smallest possible first difference, $-d$, to the bound $B_b-M_a$.} the algorithm reaches one of its bounders
```


**A13. Definition 4.2 reuses the label of Definition 4.1** (lines 295–296)

```diff
- \begin{definition}[Assignment Convergence]
- \label{def:message-difference-convergence}
+ \begin{definition}[Assignment Convergence]
+ \label{def:assignment-convergence}
```


### B. Abstract, introduction, related work, Background (novelty, scope, the split)

**B1. Abstract: the claim that no theory exists** (line 117)

```diff
- However, while this success was empirically validated, a theoretical understanding of this phenomenon has not yet been established.
+ However, earlier analyses cover only a single split constraint and split trees under strong damping, not the short cycles that splitting creates inside a larger graph.
```


**B2. Abstract: say what is proved and where, and what the heuristics achieve** (line 119)

```diff
- We prove and demonstrate that the small cycles that are generated as a result of such splitting create feedback loops that differentiate between the solution to which the algorithm converges and alternatives. This understanding allows us to propose new heuristics that further improve the algorithm.
+ We prove, for a single split cycle without damping, and measure on five benchmark families that the small cycles that are generated as a result of such splitting create feedback loops that separate the selected assignment from its alternatives. This understanding allows us to propose two heuristics: delaying the split lowers the final cost of split DMS on problems with random constraints, and a search over the two assignments that undamped split Min-sum alternates between recovers most of the gap to the damped version.
```


**B3. Introduction: the prior-theory claim, with the missing citation (Beyond Trees)** (line 161)

```diff
- However, although the success of this version of the algorithm (DMS-SCFG) was demonstrated empirically, a formal theoretic explanation of this phenomenon was not presented and has not been discovered since.
+ The resulting graph is a \emph{split constraint factor graph} (SCFG, defined in Section~\ref{Sec:background}), and the algorithm is DMS-SCFG. Its success was demonstrated empirically and explained only in part: \citet{CohenGZ20} analyzed a single split constraint, and \citet{ZivanLG20} proved that on an SCFG generated from a tree, a large enough damping factor makes DMS converge to the optimal solution. Neither work gives the recurrence of a split cycle that receives messages from the rest of the graph, nor the thresholds for progress and persistence that follow from it.
```

Note: ZivanLG20 is already in DisCSP_refs.bib.


**B4. Introduction: 'bound the variable assignment and its pace until convergence' is more than Theorems 4.9 and 4.10 say** (line 163)

```diff
- We establish the bounds that are generated by this effect, which bound the variable assignment and its pace until convergence.
+ We establish the bounds that this effect generates and the thresholds that messages from outside the cycle must cross to change the assignment.
```


**B5. Related work: 'unexplainable success'** (line 174)

```diff
- The DMS-SCFG version quickly converges to a much higher quality solution. This unexplainable success motivated this study. 
+ The DMS-SCFG version quickly converges to a much higher quality solution. Their analysis covers a single split constraint, on which DMS converges after the first iteration, and explains the gain informally: the halved cost tables give more weight to the differences between the costs in the incoming messages. \citet{ZivanLG20} introduced backtrack cost trees (Section~\ref{Sec:background}) and proved that on an SCFG generated from a tree, a large enough damping factor makes DMS converge to the optimal solution; the example of two adjacent split constraints that we analyze in Section~\ref{Sec:MS-split} is analyzed there with BCTs. Neither work gives the recurrence of a split cycle that receives messages from the rest of the graph, which is what we study. 
```


**B6. Background: cite the source of backtrack cost trees** (line 235)

```diff
- A {\em backtrack cost tree} (BCT) allows tracing 
+ A {\em backtrack cost tree} (BCT) \cite{ZivanLG20} allows tracing 
```


**B7. Background: define the split formally (AI review: before its theoretical use, with the sum property)** (lines 232–233)

```diff
- \paragraph{Backtrack Cost Trees:}
- \label{BoothB19}
+ \paragraph{Splitting:}
+ A \emph{split constraint factor graph} (SCFG) is obtained from a factor graph by replacing every function-node $F$ that holds the cost table $f$ with two function-nodes $F'$ and $F''$ that are connected to the same variable-nodes and hold tables $f'$ and $f''$ with $f'(a)+f''(a)=f(a)$ for every assignment $a$, so the objective is unchanged. In the symmetric split $f'=f''=f/2$. \citet{CohenGZ20} divide every entry in a ratio drawn uniformly from $[0.4,0.6)$, and DABP uses $0.95/0.05$.
+ 
+ \paragraph{Backtrack Cost Trees:}
+ \label{BoothB19}
```


### C. Section 5 and the larger-domain paragraph (damping, SyncBB)

**C1. Section 5: restore the caveat that the theorems are for $\lambda = 0$ (commented out in v5)** (line 608)

```diff
- %These conditions are sufficient, not necessary, and the theorems are stated for $\lambda = 0$;
+ These conditions are sufficient, not necessary, and the theorems are stated for $\lambda = 0$;
```

Note: removing the % re-activates the rest of that line, which ends with 'neither theorem applies, and the cycle may never settle at a bounder.'; it refers to the supplement figure (Figure 10), which still has to be replaced by s6_unary17.pdf.


**C2. Section 5: the sentence after the caveat repeats it ('Otherwise, ...') and contains the typo 'alternatives.By'** (line 609)

```diff
- Otherwise, they may jump back and forth and prevent the feedback loop from generating the separation between the minimal assignment selection and its alternatives.By giving most of the weight
+ By giving most of the weight
```


**C3. Section 5: grammar of the sentence that cites the theorems** (line 608)

```diff
- In Section~\ref{sec:split}, we show that when solving a cycle generated by a split MS-SCFG converges to one of its bounds if
+ In Section~\ref{sec:split}, we show that, without damping, a cycle generated by a split converges to one of its bounds if
```


**C4. Section 5: SyncBB, while Section 6 ran a centralized branch and bound (jFq3 weakness 2)** (line 598)

```diff
- Then, we use either MGM \cite{MaheswaranTBPV04} or SyncBB \cite{HirayamaY97} to select the solution of this reduced problem.
+ Then, we use either MGM \cite{MaheswaranTBPV04} or a centralized branch and bound search, which when it completes returns the solution that SyncBB \cite{HirayamaY97} would find, to select the solution of this reduced problem.
```


**C5. Section 5, Inter-DMS paragraph: same** (line 614)

```diff
- solved by MGM or SyncBB as described above
+ solved by MGM or by the branch and bound search described above
```


**C6. Section 4, larger domains: scope it, and mark what is not proved (jFq3 weakness 1 asks to revise the larger-domain claims)** (lines 564–566)

```diff
- \paragraph{Extending to Larger Domains:} The above analysis considered an SCFG with binary constraints. However, a similar feedback loop accounts for the algorithm behavior when the variables have larger domains.
- 
- In the binary case $R'_j$ holds a belief for each of the two values of $X_j$, so it has a single difference $\Delta_{R'_j}$. The feedback loop increases this difference until $B_b$ takes over as the minimum in the second column; the difference is then pinned at $B_b - M_a$. With a larger domain $R'_j$ holds a belief for each value of $X_j$, one is the selected assignment and the rest are alternatives, so there is now a separate difference between the selected assignment and each alternative. The feedback loop grows all of them, and each is pinned once some entry of $F'_{ij}$ takes over as the minimum that generates that alternative's belief; that entry is the alternative's bounder, exactly as $B_b$ is in the binary case. A domain of size $|D|$ thus has up to $|D|-1$ bounders, one capping each alternative's difference, in place of the single binary one; the smallest of these differences, the alternative closest to the selected assignment, sets the flipping threshold $FT_{F_{ij}}$.
+ \paragraph{Larger Domains:} The theorems above are stated for binary domains, and we do not extend them. The same feedback loop, however, operates with larger domains, and Section~\ref{Sec:experiments} measures it there. With $|D|$ values, $R'_j$ holds a separate difference between the selected value of $X_j$ and each alternative. The feedback loop grows all of them, and each is pinned once some entry of $F'_{ij}$ takes over as the minimum that generates that alternative's belief; that entry is the alternative's bounder, exactly as $B_b$ is in the binary case. The smallest of these pinned differences, the alternative closest to the selected value, sets the flipping threshold $FT_{F_{ij}}$.
```

Note: text from the earlier 51-edit list (E14); its OLD string is the paragraph as it is in v5.


### D. Section 6 (axis, DABP, k)

**D1. Section 6: say how DABP is put on the iteration axis (XqaZ question 6, AI review weakness 6)** (line 639)

```diff
- to plot it on the iteration axis we use the simulated runtime method \cite{SultanikLR08}, with times measured on an NVIDIA RTX 4090.
+ to plot it on the iteration axis we use the simulated runtime method \cite{SultanikLR08}. With times measured on an NVIDIA RTX 4090, an iteration of DABP takes 6.9, 3.3, 5.6, 6.3 and 8.2 times as long as an iteration of DMS on random sparse, random dense, scale free, graph coloring and meeting scheduling, and we stretch the DABP curve by these factors.
```

Note: ratios from experiments/aaai/data_paper_20260928/dabp_timing.csv (column ratio); plot_final.py stretches DABP by 2 x ratio on the paper axis.


**D2. Section 6: DABP and DABP-NoSplit (XqaZ question 7)** (line 693)

```diff
- without the split (DABP-NoSplit) its cost is
+ without the split (DABP-NoSplit, which uses the same network and driver and skips only the built-in split step) its cost is
```

Note: from the docstring of AttentiveNoSplitEngine in experiments/aaai/code/engines.py; confirm that the model is not trained differently before submitting.


**D3. Figure 3 caption: the search time is not on the axis** (line 628)

```diff
- \caption{Mean solution cost per iteration when solving the random sparse, random dense and scale free problems.}
+ \caption{Mean solution cost per iteration when solving the random sparse, random dense and scale free problems. MS-SCFG-opt is drawn at iteration 400 and does not include the time of the search.}
```


**D4. Figure 4 caption: same** (line 646)

```diff
- \caption{Mean solution cost per iteration when solving Graph coloring (left) and Meeting scheduling (right) problems.}
+ \caption{Mean solution cost per iteration when solving Graph coloring (left) and Meeting scheduling (right) problems. MS-SCFG-opt is drawn at iteration 400 and does not include the time of the search.}
```


**D5. Section 6, setup: an iteration on the split graph costs about twice an iteration on the original (AI review weakness 6)** (line 634)

```diff
- All algorithms run for 4000 synchronous iterations, and in every iteration the cost of the assignment selected by the current beliefs is evaluated on the original problem.
+ All algorithms run for 4000 synchronous iterations, and in every iteration the cost of the assignment selected by the current beliefs is evaluated on the original problem. An iteration on a split factor graph takes about 1.8 to 2 times as long as an iteration on the original one, so the curves compare iterations and not running time.
```

Note: 1.80 to 1.97 was measured on a Mac, seed 0 (2026-09-25), not on the machine of the DABP timings: re-measure on the RTX box before submitting.


**D6. Section 6: how k was chosen and on which instances (XqaZ question 5, jFq3 weakness 3)** (line 669)

```diff
- The blue line in Figure~\ref{Fig:bound} is DMS-$k$DS with $k = 1000$.
+ The blue line in Figure~\ref{Fig:bound} is DMS-$k$DS with $k = 1000$, the same value on every benchmark; no value of $k$ was selected per benchmark, and every value was run on the same 50 instances.
```


### E. Conclusions

**E1. Conclusions: scope of the proof, and no 'converges to more than one solution'** (line 704)

```diff
- We prove that the feedback loops generated by the split cause separation between the minimal belief that the algorithm propagates and its alternatives. We further identify a property that was not reported before, that when Min-sum is applied without damping, it often converges to more than one solution, which the algorithm not only oscillates between them, but also has different nodes of the graph consider different solutions at the same iteration. This results in low quality solutions.
+ We prove that, in a single split cycle without damping, the feedback loops generated by the split cause separation between the minimal belief that the algorithm propagates and its alternatives, and we measure the same effect on five benchmark families. We further identify a property that was not reported before, that when Min-sum is applied without damping, it often alternates between two assignments, with different nodes of the graph following different ones of the two solutions at the same iteration. This results in low quality solutions.
```


**E2. Conclusions: 'an advantage over previous versions, especially in distributed settings' is not what Section 6 shows** (line 706)

```diff
- These heuristics were found to have an advantage over previous versions of the algorithm, especially in distributed settings.
+ On problems with random constraints, delaying the split gives a lower final cost than splitting from the start, and selecting between the two assignments recovers most of the gap between undamped and damped split Min-sum.
```


## 4. What text cannot fix

- **Controls for the postprocessing** (AI review W5, the most serious experimental gap). Three runs, 50 instances per benchmark, each takes minutes: (1) MS on the unsplit graph, assignments of iterations 398 and 400, then MGM; (2) two random values per variable, then MGM; (3) MGM on the full domains, started from the MS-SCFG assignment at iteration 400. They show whether the gain comes from the two values that undamped split Min-sum alternates between, or from the added optimizer.
- **DABP lines** on random dense, scale free and meeting scheduling are June runs on other instances, without tie-break fractions, and their messages restart at library iteration 1000. Either draw DABP only on random sparse and graph coloring (the one-row figure of the earlier cut list does this) or rerun them. Until then fix D2 should not be applied.
- **The supplement figure of the single cycle** (Figure 10) is still the old file made with the wrong damping, with the old caption. Replace it with `s6_unary17.pdf` and the caption K22 of the earlier list. Fix C1 refers to that figure.
- **The ternary figures** are from before the axis fix and have the old line names. Remove them or rerun them.
- **Page limit.** The main text ends on page 9 (limit: 8 plus references). The 34 fixes add about 50 lines in total (A 13, B 31, C −10, D 13, E 3; roughly half a page), so the cuts of `publish/CUT_LIST_2026-09-30.html` are still needed; several of them must be redone by hand because they target sentences that changed.
- **Not AAAI-driven but still open** from the earlier 51-edit list: E06 (damping only at variable-nodes), E26 (iteration units), E08–E10, E12, E15–E19, E21, E22a/b, E36–E38, E40a/b.
- A **time-to-quality plot** (XqaZ) would settle the speed claim. D1 and D5 only say honestly what the iteration axis is.

## 5. Rating

**4/10, weak reject, as the file stands.** The empirical reporting answers most of what the AAAI reviewers asked for (jFq3 W3 and W4, most of W2, the minor points). But the two reasons that decided the AAAI outcome are still there: a theorem that a reviewer can break with the paper's own example (Theorem 4.10), and a missing citation that makes the novelty claim false (Beyond Trees). The abstract and Conclusions still say more than the single-cycle theory supports, and the controls for the postprocessing do not exist.

Credit for Roie's edits: the two new definitions answer XqaZ's "what is convergence"; the footnote on normalization answers the AI review; the converse at `t+2` is correct.

| Criterion | Rating | Note |
|---|---|---|
| Significance / contribution | Fair | a real DCOP question; the theory is one binary cycle without damping |
| Originality | Fair | recurrence and thresholds are new; the prior-work framing is wrong without Beyond Trees |
| Soundness / validity | Fair | the core recurrence is right; Theorem 4.10 is false as worded and Theorem 4.9 overclaims |
| Clarity / presentation | Fair | the example paragraph and the broken sentences hurt; Section 6 reads well |
| Related work / positioning | Poor | Beyond Trees missing; "unexplainable" and "has not been discovered" |
| Reproducibility / transparency | Fair | Section 6 gives counts, caps and tests; DABP protocol and the axis are not stated |
| Ethics & limitations | Fair | the λ = 0 limit is stated in Section 6 and commented out in Section 5 |

**What moves it to about 6/10:** fixes A (the theorem and its proof), B (the citation and the abstract), C1–C3 (the damping caveat) and D1–D6, plus the three controls and a decision on the DABP lines. Fixes A and B alone remove the two AAAI-deciding objections.

## 6. Not checked

- The benchmark runs and the data behind Section 6 were not repeated; I compared Section 6 with the data files in an earlier review and it matched except the sentence that is now fixed.
- The contents of Zivan et al. 2020 and Cohen et al. 2020 come from my earlier reading of the papers (2026-09-29), not from today.
- The formulas in the AAAI reviews were lost in the paste. For the theorem points (jFq3 W1, AI W1, W3) I checked the current statements against exact simulation instead of against the reviewers' formulas.
- Fix D5 uses 1.8 to 2 as the cost of an SCFG iteration, measured on a Mac with seed 0. Re-measure it on the machine of the DABP timings. Fix D2 follows the docstring of `AttentiveNoSplitEngine`; confirm that the model is trained the same way.
- Scripts: `v5_thm_checks.py`, `v5_thm_checks2.py`, `v5_thm49_check.py`, `v5_fixes.py` in the session scratchpad.
