# Audit 46, literature half: what the primary sources actually say

Checked 2026-09-21. This was a reading task. No code, simulation or experiment was run. Nothing under `/Users/or/Projects/Belief-Propagation-Simulator` was changed.

## How to read this file

**Target case.** Synchronous Min-sum on a pairwise factor graph. The graph is dense, random, not bipartite, with many cycles. Domain size 10 or 20. float64. The run is damped on the unsplit graph first. Then every factor is split 0.5/0.5 mid-run, the existing messages are handed to the copies (so the split graph starts from non-zero messages), and damping is switched off. Observed: runs that alternate between 2 assignments, and one run where some agents visit 3 values in an exact 42-update cycle.

**Verification label.** Each section starts with one of: VERIFIED from the primary text / PARTIALLY verified / NOT verified from the primary text.

**Quotes.** I kept direct quotation to one short quote in the whole file (Section 4), on purpose, for copyright reasons. Everything else is my paraphrase with a page location so it can be checked. Formulas are given in plain notation. Where I substituted numbers into a formula myself, I say so.

**Applicability wording.** I use only three verdicts: "applies as stated", "applies only if <premise> is checked", "does not apply because <reason>".

**Proof notes** are observations about what the printed proof does or does not do. They are not new theory.

---

## 1. Yedidia 2011 (MERL TR2011-087)

**VERIFIED from the primary text.** Read: PDF pp. 1-3 and 10-23 of 34.

**Citation.** J. S. Yedidia, "Message-Passing Algorithms for Optimization and Inference: Belief Propagation and Divide & Concur", MERL Technical Report TR2011-087, October 2011. The cover page says it appeared in Journal of Statistical Physics, 2011. The inner manuscript title is worded slightly differently ("... for Inference and Optimization: 'Belief Propagation' and 'Divide and Concur'"). PDF page = printed page + 2. Page numbers below are the printed ones.

### 1a. Min-sum message and belief rules. Sections 5-6, printed pp. 10-14, eqs (4)-(8)

- **Statement.**
  - Eq (4), p. 12: the belief of variable i is the sum of all incoming factor messages. `b_i(x_i) = sum_{a in N(i)} m_{a->i}(x_i)`.
  - Eq (5), p. 13: the message from variable i to factor a is the sum of the messages from all the *other* factors. `m_{i->a}(x_i) = sum_{b in N(i)\a} m_{b->i}(x_i)`. Eq (6), p. 13, is the same thing written as `b_i - m_{a->i}`.
  - Eq (7), p. 13, is a three-variable example. Eq (8), p. 14, is the general factor rule: `m_{a->i}(x_i) = min over the other variables of [ C_a(X_a) + sum_{j in N(a)\i} m_{j->a}(x_j) ]`. The minimum is taken over the full incoming cost vectors of the other variables. The neighbour's currently chosen value is not plugged into the table.
  - The readout (pick the lowest-cost state of the belief) is described in the step list of Section 5, p. 10, and in the caption of Fig. 7, p. 11. It has **no equation number**.
  - p. 14: the rules are update rules. They hold as equalities only at a fixed point.
  - Section 8, p. 17: on graphs with cycles the algorithm is well defined but not necessarily exact. Section 10, p. 20: on graphs with cycles BP may fail to converge.
- **Assumptions.** Topology: any factor graph; exactness is claimed only for trees. Arity: any. Domain: finite and discrete in these sections. Initialization: random or non-informative messages (p. 10). Schedule: not fixed; updating all messages in parallel is allowed (pp. 11-12). Ties: not discussed in Sections 5-6. Damping: none here. Numerical precision: not discussed.
- **Object of the conclusion.** Message vectors and belief vectors. The assignment is a readout from the belief. No statement about "eventually".
- **Status.** Definition / tutorial description. No theorem.
- **Proof note.** Not applicable.
- **Applicability.** Applies as stated. These are the update rules of the target algorithm.

### 1b. Splitting. Section 9, printed pp. 18-20, Fig. 13 (p. 18), eqs (13)-(14) (p. 19)

- **Statement.**
  - Fig. 13, p. 18: a factor is replaced by two identical factor nodes, each with half the cost. The total cost function is unchanged. The min-sum algorithm on the new graph is a different algorithm.
  - Eq (13), p. 19: `m_{i->a}(x_i) = (k_a - 1) m_{a->i}(x_i) + sum_{b in N(i)\a} k_b m_{b->i}(x_i)`.
  - Eq (14), p. 19: `m_{a->i}(x_i) = min [ C_a(X_a)/k_a + (k_i - 1) m_{i->a}(x_i) + sum_{j in N(a)\i} k_j m_{j->a}(x_j) ]`.
  - My substitution (not printed in the paper): for the symmetric two-way factor split, `k_a = 2` for every factor and `k_i = 1` for every variable. Then (13) becomes `m_{i->a} = m_{a->i} + 2 * sum_{b != a} m_{b->i}`. The message a variable sends to one copy now contains, with weight 1, the message it got back from the sibling copy.
  - Guarantees. Yedidia reports (crediting Ruozzi and Tatikonda) that for suitably chosen **real** k, for example on a regular graph of degree d with `k_i = 1` and `0 < k_a < 1/d`, a fixed point with a unique lowest-cost state at every node gives the global optimum, and that simple schedules exist that provably converge. He calls the no-ties requirement a loophole (pp. 19-20).
- **Assumptions.** The reduction from "run min-sum on the split graph" to eqs (13)-(14) **assumes the messages to and from all copies of a node are initialized equal**. The symmetry then persists. Topology: any. Arity: any. Domain: finite discrete. Schedule: not fixed for (13)-(14); the convergence remark is about particular schedules. Ties: none, for the optimality remark. Damping: none. Precision: not discussed.
- **Object.** (13)-(14): messages. The guarantee: the assignment read at a fixed point.
- **Status.** Survey description. The guarantees are reported from other work and are not proved here.
- **Proof note.** Not applicable.
- **Applicability.**
  - Eqs (13)-(14) with `k_a = 2`: applies only if "the hand-off at the split gives both copies of a factor identical messages, in both directions" is checked. If the copies start unequal, the run on the split graph is still plain min-sum on that graph (rule 1a applies), but the compact form (13)-(14) does not describe it.
  - The reported guarantees: do not apply because `k_a = 2` is outside the `k_a < 1/d` range, and because they are statements about a fixed point, which an oscillating run does not have.

---

## 2. Weiss and Freeman 2001

**VERIFIED from the primary text.** Read: all 11 pages of the author manuscript hosted at MIT CSAIL.

**Citation.** Y. Weiss and W. T. Freeman, "On the optimality of solutions of the max-product belief-propagation algorithm in arbitrary graphs", IEEE Transactions on Information Theory 47(2), 2001. I did not see the journal page range in the PDF. Page numbers below are the manuscript's.

### 2a. Algorithm and setting. Section I, pp. 2-3, eqs (6)-(7)

- **Statement.** Pairwise Markov random field. Eq (6), p. 3, is the max-product message update (a vector over the receiver's values, excluding the receiver's own message). Eq (7), p. 3, is the belief. Messages are initialized to constant functions. All nodes update in parallel (p. 3). A unique maximizing value at every node is assumed, and a unique MAP assignment is assumed (p. 3). Eq (8), p. 3, records the damping scheme of Murphy et al.: `m(t+1) = F( alpha*m(t) + (1-alpha)*m(t-1) )`. This damps the *input* to the update. It is not the same formula as the Zivan/Cohen damping in Sections 3-4 below.
- **Assumptions.** Topology: arbitrary graph. Arity: pairwise (any model is first converted to pairwise, Fig. 1). Domain: general (discrete; Gaussian case treated separately). Init: constant messages. Schedule: fully parallel. Ties: none. Damping: only mentioned. Precision: normalization mentioned to avoid underflow.
- **Status.** Definition.
- **Applicability.** Applies as stated for the update rule (max-product is min-sum after taking negative logs). The constant-start assumption does not match the target case (warm start after the split).

### 2b. Unwrapped (computation) tree. Section II-A, pp. 4-5, Fig. 3 (p. 5); construction and its five properties, Section II-B, p. 6

- **Statement.** After t parallel iterations, the messages a node receives in the loopy graph equal the messages it would receive in a tree built by recursively copying the nodes that the earlier messages came from (Fig. 3 caption, p. 5). Every node of the tree is a replica of one node of the original graph. Replicas of the same original node sit at different depths. Section II-B, p. 6, lists the properties of this tree that the proof uses: same neighbours, same local conditional probabilities, isomorphic subtrees, an infinite-chain property, and a lemma (from their ref [22]) about periodic assignments.
- **Assumptions.** Same as 2a. The equivalence is stated for a run that starts from constant messages. For the fixed-point proof, the leaf potentials of the tree are changed so that they include the fixed-point messages.
- **Object.** Messages after a finite number t of iterations.
- **Status.** Construction with an argument. Stated in a figure caption and in the text, not as a numbered theorem.
- **Proof note.** The paper does not state the version for an arbitrary (non-constant) start. It only uses modified leaves at a fixed point.
- **Applicability.** Applies only if "the hand-off messages at the split are placed at the leaves of the tree of the *split* graph" is checked. That extension is standard, but it is not printed in this paper.

### 2c. Claim 1 (the optimality result). Stated p. 4, proved in Section II-B, pp. 6-7

- **Statement.** The result is labelled "Claim 1", not "Theorem". For an arbitrary graphical model with arbitrary potentials: if m* is a **fixed point** of max-product and x* is the assignment read from it, then x* has higher posterior probability than every other assignment that differs from x* only on a set of nodes whose induced subgraph is a disjoint union of trees and single loops (the "SLT neighbourhood").
- **Assumptions.** Topology: arbitrary. Arity: pairwise. Domain: general. Init: irrelevant (it is a fixed-point statement). Schedule: irrelevant. Ties: unique maximizer at every node. Damping: none. Precision: not discussed.
- **Object.** The assignment read at a fixed point. Not "eventually": it says nothing about whether a fixed point is reached.
- **Status.** Claim with a full proof in the paper.
- **Proof note.** Corollaries 1-4 (pp. 7-9) cover Gaussian means, turbo codes (Hamming distance 2), and 2D grids. The introduction, p. 1, mentions that on single-loop graphs the algorithm converges to a stable fixed point or to a periodic oscillation. That is cited to other papers and is not proved here.
- **Applicability.** Does not apply because the target runs do not reach a fixed point. The paper says nothing about a run that keeps switching.

### 2d. Why tree conclusions do not transfer directly. Section IV, p. 9

- **Statement.** Two obstacles: the numbers of replicas of different nodes in the tree can be unbalanced, and the leaves contribute terms that have no counterpart in the original graph.
- **Status.** Discussion.
- **Applicability.** Applies as stated, as a caution.

---

## 3. Zivan, Lev and Galiki, AAAI 2020

**VERIFIED from the primary text.** Read: all 8 pages.

**Citation.** R. Zivan, O. Lev, R. Galiki, "Beyond Trees: Analysis and Convergence of Belief Propagation in Graphs with Multiple Cycles", Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), pp. 7333-7340.

### 3a. Setting, damping formula, SCFG. p. 7335

- **Statement.** Variable-to-function message: sum of the messages received from the *other* function nodes in the previous iteration, minus a normalizing constant. All iteration-0 messages are zero vectors. Damping: `m(k) = lambda * m(k-1) + (1 - lambda) * mhat(k)`, where mhat is the newly computed message. **lambda is the weight of the OLD message.** Range printed as lambda in [0, 1). The formula is stated for a generic message from any node. An SCFG replaces each constraint by two function nodes whose tables sum to the original table.
- **Status.** Definition.
- **Applicability.** Applies as stated for the damped phase, provided the group's damping uses the same convention (weight on the old message). That is a code question, not a literature question.

### 3b. Definition 1, Backtrack Cost Tree. p. 7337

- **Location detail.** Definition 1 is on p. 7337, at the end of the "Preliminaries" section. There is **no section titled "Backtrack Cost Tree"**. The next heading on that page is "Max-sum and BCT".
- **Statement.** For the belief about value x in a message from X_i to a function node at time t, the BCT is a tree. Its root is that belief. Its children are the nodes it received messages from at time t-1, with the belief entries that were used. This continues down to leaves at time 0. `BCT^t_i` is the tree of the minimal belief, which is the one that selects the assignment. "State" at time t means the vector of values selected by all variable nodes (p. 7336).
- **Assumptions.** Zero start (leaves at time 0 carry zero messages). Footnote 5, p. 7337: no ties, assumed for simplicity.
- **Status.** Definition.
- **Applicability.** Applies only if "the leaves can carry the non-zero hand-off messages" is checked. The paper only treats a zero start.

### 3c. Lemma 1. Statement pp. 7337-7338, proof p. 7338

- **Statement.** For any factor graph there is a time t0 and a positive integer k such that the state (the selected values of all variables) at time t equals the state at time t+k for every t >= t0. The costs added to the BCTs are periodic as well. The paper calls the repeating sequence the final periodical and says it is the one whose assignment BCT is minimal. Footnote 6: convergence is the case k = 1.
- **Assumptions.** Topology: any factor graph. Arity: the paper works with binary constraints. Domain: finite. Init: zero messages. Schedule: synchronous iterations (implicit). Ties: none (footnote 5). Damping: none in this lemma. Precision: not discussed.
- **Object.** **Assignments** (and BCT cost increments). Not messages, not beliefs. It is an "eventually" statement. **No bound on k or on t0.**
- **Status.** Lemma with a short printed proof.
- **Proof note.** The proof goes: finitely many assignments, so some state appears infinitely often; then it says some finite sequences of states between two appearances occur infinitely often; it assumes two such sequences with different average cost and argues that, because costs accumulate, after enough repetitions the value used by the costlier sequence is never chosen again. Things the printed proof does not do: (1) it does not show why some *finite* return sequence must itself repeat infinitely often; (2) it does not treat two sequences with equal average cost; (3) the step "never chosen again" is asserted; (4) it never shows that the message state repeats, so it is not a finite-state argument about the real state of the algorithm.
- **Applicability.** Applies only if "zero start and no ties can be replaced by the warm start" is checked. Even if it holds, it only says "eventually periodic with some period k". It is equally compatible with period 2 and with a 42-update cycle. It explains neither.

### 3d. Lemma 2 and Corollary 1. p. 7338

- **Statement.** For a variable on a cycle, after some time all beliefs in its message share the same bottom q levels of their BCTs (Lemma 2). Two variables on the same cycle share those bottom levels too (Corollary 1).
- **Status.** Lemma and corollary with short proofs. Both rest on Lemma 1.
- **Applicability.** Same condition as 3c.

### 3e. Theorem 1. p. 7338

- **Statement.** With damping, for a large enough lambda, the end state is determined only by the part of the BCT that starts at t0 (written lambda-BCT-bar), not by what happened before t0.
- **Assumptions.** The proof takes epsilon = the smallest possible difference between two constraint values, d = the maximal degree in G, and requires `1 - lambda < 1/(2d)`.
- **Status.** Theorem with a short proof.
- **Proof note.** The proof treats damping as a multiplication of new contributions by (1 - lambda) per step. It does not write out the lambda * old-message term. It reuses Lemma 1 for the damped run "for the same reasoning".
- **Applicability.** Does not apply because damping is off after the split.

### 3f. Corollary 2 and the paragraph after it. p. 7339

- **Statement.** For large enough lambda, **if** the assignment tree induced by lambda-BCT-bar is consistent (the same variable gets the same value everywhere in the tree), the damped run converges to the optimal solution. The next paragraph says this does not mean every convergence is optimal, and that with a smaller lambda damped runs do converge to suboptimal solutions.
- **Status.** Corollary with a short proof.
- **Applicability.** Does not apply because damping is off after the split, and consistency is a premise that is not shown for dense graphs.

### 3g. Proposition 1. p. 7339

- **Statement.** In an SCFG with a linear division (footnote 7: each function node's two copies have a constant entry ratio q in (0,1)), **if the original graph was a tree**, then for high enough lambda the induced assignment is consistent and optimal.
- **Status.** Proposition with a very short proof.
- **Applicability.** Does not apply because the original graph is dense with many cycles, and damping is off after the split.

### 3h. Example 1 and Fig. 4. p. 7339

- **Statement.** Three variables in a chain, two constraints, each split symmetrically, domain of 2 values. Without damping the algorithm fails to converge. With lambda = 0.7 it reaches the optimal solution after 23 iterations. With lambda = 0.3 it converges to a suboptimal solution (cost 132 against the optimal 130) after 19 iterations.
- **Status.** Worked example (empirical).
- **Applicability.** Applies as stated, as a published small example of undamped symmetric split not converging. It is an example, not a theorem. It starts from zero messages.

---

## 4. Cohen, Galiki and Zivan, Artificial Intelligence 2020

**VERIFIED from the primary text.** Read: all 22 pages of the author copy.

**Citation.** L. Cohen, R. Galiki, R. Zivan, "Governing convergence of Max-sum on DCOPs through damping and splitting", Artificial Intelligence 279 (2020) 103212.

### 4a. Setting and damping formula. pp. 4-7

- **Statement.** One variable per agent, binary constraints, integer costs (p. 4). Eqs (1)-(2), p. 5: the Q and R messages, zero start, normalizing constant. Eqs (3)-(4), p. 6: belief and argmin readout. New messages are generated synchronously (p. 6). **Eq (5), p. 7:** `m(k) = lambda * m(k-1) + (1 - lambda) * mhat(k)`. lambda is the weight of the OLD message. The paper says that in all its implementations **only variable nodes damp**. p. 7 also says there is no theoretical guarantee that identifies when damped BP converges.
- **Internal inconsistency (observation).** The range is printed as lambda in (0, 1], while the text says lambda = 0 is standard Max-sum. The AAAI paper prints [0, 1).
- **Status.** Definition.
- **Applicability.** Applies as stated for the damped phase, if the group's code damps variable-to-factor messages with the weight on the old message.

### 4b. SCFG definition. Section 5.1, p. 9

- **Statement.** Each constraint is represented by two function nodes whose tables sum to the original. "Constant" SCFG: one fixed ratio for the whole table (0.5 is the symmetric case; 0.95 is an example of a non-symmetric constant ratio). "Random" SCFG: a random ratio per table entry.
- **Status.** Definition.
- **Applicability.** Applies as stated. The target case is a constant SCFG with ratio 0.5.

### 4c. Results about damping on trees. pp. 7-8

- **Lemma 1, p. 7.** There is a tree scenario where damped Max-sum needs at least `2(n-2) + log_{1/lambda}(C)` steps. Proved by an example (Fig. 4).
- **Proposition 1, p. 8.** The run time of damped Max-sum is at least weakly polynomial. Follows from Lemma 1.
- **Lemma 2, p. 8.** On a tree, after at most `2(n-2) * log_{1/lambda}(Chat/epsilon)` steps a variable can select its optimal value. Short proof sketch. No ties is assumed here (p. 8).
- **Proposition 2, p. 8.** Damped Max-sum converges to the optimal solution on **tree** graphs in weakly polynomial time.
- A remark on p. 8 says similar proofs work for a single cycle under some restrictions, citing their ref [38] (Weiss 2000). It is not proved in this paper.
- **Status.** Lemma 1: proved by example. Lemma 2: sketch. Propositions: follow from the lemmas.
- **Applicability.** Does not apply because the target graphs are not trees.

### 4d. Results about splitting a single constraint. pp. 9-10

- **Lemma 3, pp. 9-10.** k variables and s **identical** function nodes, each connected to all k variables: Max-sum converges to the optimal solution after the first iteration. Proved by induction. The proof relies on the table having a unique smallest entry. Footnote 6 points to Section 3.1 for the no-ties assumption, but the assumption is actually stated on p. 8 (observation: a wrong internal pointer in the paper).
- **Lemma 4, p. 10.** Same with proportional tables. Full proof omitted by the authors.
- **Lemma 5, p. 10.** Same with damping, any damping factor. Full proof omitted by the authors.
- **Proposition 3, p. 10.** Damped Max-sum on a constant SCFG made from a graph with **a single constraint** converges to the optimal solution after the first iteration. Proof: "immediate" from Lemmas 3-5.
- p. 10 also reports, empirically, that a *random* split of a single constraint often did not converge (20,000 random single-cycle graphs).
- **Assumptions.** Topology: one constraint only. Init: zero. Ties: unique smallest entry. Schedule: synchronous.
- **Object.** Assignments, from iteration 1 onward.
- **Status.** Lemma 3 proved; Lemmas 4-5 stated with proofs omitted; Proposition 3 rests on them.
- **Applicability.** Does not apply because the target graph has many constraints.

### 4e. Section 5.3, "Splitting adjacent constraints", p. 11, Fig. 5. **The most relevant passage in the four DCOP/BP sources.**

- **Statement.** The authors ask whether the single-constraint result extends to larger graphs. They say it does not, even for three variables and two constraints. Fig. 5 is that graph with both constraints split with a constant symmetric split, domain of 2 values. On it, undamped Max-sum "does not converge, but rather oscillates between suboptimal solutions" (Cohen, Galiki and Zivan 2020, p. 11). Damped Max-sum with lambda = 0.5 converges to the optimal solution after 17 iterations. The cost tables are the same as in Fig. 4 of the AAAI paper (Section 3h above).
- **Assumptions.** Zero start. Synchronous. Domain 2.
- **Object.** Assignments.
- **Status.** Empirical example. No proof, no statement about the period.
- **Applicability.** Applies as stated, as a published precedent: undamped symmetric split can oscillate between assignments already on a 3-variable chain. It does not say why, does not bound the period, and starts from zero messages.

### 4f. Experiments and explanation. pp. 11-20

- **Statement.** Random uniform problems use costs drawn between 0 and 10, domain 10, 50 or 100 variables (p. 11). Ties are broken with random value preferences (p. 11). All versions run inside an anytime framework that remembers the best assignment seen. Damped runs that do not converge still give good anytime results (pp. 12-19). Section 6.4, pp. 19-20, gives a heuristic explanation.
- **Related-work remark, p. 3.** The authors cite Rebeschini and Tatikonda (their ref [32]) and Ruozzi and Tatikonda (their ref [33]) and say those results give no guarantee or indication about convergence of BP in combinatorial domains. This matches my own reading of those two papers (Sections 9 and 10 below).
- **Status.** Empirical.
- **Applicability.** Applies as stated, as context only.

---

## 5. Goles and Olivos 1980

**NOT verified from the primary text.**

**Citation (from the tex's BibTeX comment and the DOI; I did not see the printed first page).** E. Goles and J. Olivos, "Periodic behaviour of generalized threshold functions", Discrete Mathematics 30(2), 1980, pp. 187-189. DOI 10.1016/0012-365X(80)90121-1.

**What I used instead.** (1) The abstract as shown in the first author's institutional repository record. (2) A secondary source: Kaaser, Mallmann-Trenn and Natale, arXiv:1508.03519, pp. 1-2.

- **Statement (from the abstract only).** A map from {0,1}^n to {0,1}^n whose components are a symmetric set of threshold functions. Applying it repeatedly leads to a fixed point or to a cycle of length two.
- **Secondary source.** Kaaser et al. credit Goles-Olivos and Poljak-Sura with showing, by a potential-function argument, that the deterministic binary majority process ends in a state that repeats with period at most two.
- **Assumptions I could confirm from the abstract.** Binary states. Symmetric weights. All components updated together. **Not confirmed:** how ties at the threshold are handled, whether self-weights are allowed, the exact threshold convention, whether the proof is complete.
- **Object.** The state vector (the assignments). Eventual.
- **Status.** Theorem, per the abstract. Proof not seen.
- **Applicability.** Does not apply because the states are binary (the target has domain 10 or 20) and because the update is a threshold rule on the neighbours' current states, not a Min-sum message update.

---

## 6. Poljak and Sura 1983

**VERIFIED from the primary text for pp. 119-120** (the publisher's free two-page preview). These two pages contain the model, the theorem and the whole proof. p. 121 was not seen.

**Citation.** S. Poljak and M. Sura, "On periodical behaviour in societies with symmetric influences", Combinatorica 3(1), 1983, pp. 119-121.

- **Statement.** V is a finite set of members. Each member holds an opinion in {0, 1, ..., p}. w(u,v) is a real number, the influence of v on u, with w(u,v) = w(v,u). At every step, every member u takes the opinion i that has the largest total weight among the members currently holding i, that is, the largest `sum of w(u,v) over v with f_t(v) = i` (rule (1), p. 119). Theorem, p. 119: for every such system the period is 1 or 2.
- **Answer to the specific question.** The interaction is **not** an arbitrary symmetric payoff table. A pair (u,v) contributes w(u,v) to opinion i **only when v currently holds i**. In cost-table language the pairwise term is `w(u,v) * [opinions equal]`. Nothing else is allowed.
- **Assumptions.** Topology: any (w can be zero). Pairwise only. Domain: any finite number of opinions. Weights: real, sign not restricted. u and v need not be distinct, so a self-weight w(u,u) is allowed. Init: any. Schedule: all members update at the same time. **Ties: broken by taking the highest-numbered opinion among the tied ones.** Damping: none. Precision: exact arithmetic implied.
- **Object.** The opinion map (the assignments). Eventual. No bound on the time before the period starts.
- **Status.** Theorem with a full proof (p. 120).
- **Proof note.** The authors first note (p. 119) that the system must become periodic because there are finitely many opinion maps and each map determines the next one. This finite-state argument is valid here because the assignment **is** the whole state. The proof then assumes a period k > 2, picks k inequalities per member with at least one strict, sums them over V, and uses symmetry to get 0 > 0. I saw no missing step in the two pages.
- **Applicability.** Does not apply because (1) the pairwise term must be a weight times an equality indicator, not a general cost table, and (2) in Min-sum the assignment is not the whole state: the next assignment is not a function of the current assignment alone. It could be used only through the tex's own reduction; see "Check of the local notes", part B.

---

## 7. Ashkenazi-Golan, Mergoni Cecchelli and Plumb 2025

**VERIFIED from the primary text.** Read: pp. 1-9 and 12-14 of arXiv:2505.10378v2.

**Citation.** G. Ashkenazi-Golan, D. Mergoni Cecchelli, E. Plumb, "Simultaneous Best-Response Dynamics in Random Potential Games", arXiv:2505.10378v2, 16 May 2025. Marked "Preprint. Under review."

- **Setting.** Definition 2.2: an n-player, m-action random potential game. The potential value of **each full action profile** is drawn independently from a continuous distribution. So the potential is a generic table over joint profiles. It is **not** a sum of pairwise terms. Eq (3): every player simultaneously plays a best response to the others' previous actions.
- **Lemma 3.2, p. 5 (proof in Appendix A.2, p. 12).** For **two players**, with probability one the dynamics end in a cycle of length one or two. The proof follows two interleaved sequences of potential values that strictly increase until a player repeats. "With probability one" is only there to rule out ties.
- **Theorem 3.1, p. 5.** For two players and enough actions, the dynamics reach a two-cycle within `log(eps)/log(3/4)` steps with probability at least 1 - eps.
- **Remark 3.3.** The two-cycle is {(a,b), (a',b')} where (a,b') and (a',b) are both Nash equilibria.
- **Three or more players.** Simulation only (Section 4.3): the dynamics reach a Nash equilibrium with high probability. Section 5 says a rigorous proof for n >= 3 is still open. **There is no period-at-most-two theorem for n >= 3 in this paper.**
- **Assumptions.** Two players (for the theorems). Any finite number of actions. No ties (continuous distribution). Simultaneous updates. Any start. No damping.
- **Object.** Action profiles (assignments). Lemma 3.2 is eventual; Theorem 3.1 gives a finite bound with high probability.
- **Status.** Theorem and lemma with full proofs for two players. Empirical for n >= 3.
- **Applicability.** Does not apply because the theorems are for two players only, the potential is a generic random table rather than a sum of shared pairwise tables, and the dynamics are best responses on assignments, not Min-sum.

---

## 8. Forney, Kschischang, Marcus and Tuncel 2001

**PARTIALLY verified.** Only pp. 239-240 (the publisher's free preview: abstract and most of the introduction). The propositions themselves were not opened.

**Citation.** G. D. Forney Jr., F. R. Kschischang, B. Marcus, S. Tuncel, "Iterative decoding of tail-biting trellises and connections with symbolic dynamics", in *Codes, Systems, and Graphical Models*, IMA Volumes in Mathematics and its Applications vol. 123, Springer New York, 2001, starting at p. 239 (end page 264 is from the task description; I did not see it).

- **Statement (introduction only).** The setting is a tail-biting trellis, which is a graph with a **single cycle**. The convergence behaviour of min-sum there is governed by the trellis path or paths with the smallest average weight per symbol, which the authors call dominant pseudocodewords. If that path is longer than one trip around the cycle, min-sum tends to settle on something that is not a codeword.
- **Formal results named in the introduction.** Proposition 5.1 (a review of Anderson and Hladik's sum-product result), and, as recorded in my reading notes of the introduction, Propositions 6.1, 7.1, 8.2, 8.3, 10.1 and 10.2. One of them says the generating-function min-sum converges along a particular subsequence of the integers. **I did not read the statements or the assumptions of any of these.**
- **Cross-check from verified texts.** Weiss-Freeman p. 1 and Zivan et al. p. 7333 both describe the single-cycle result the same way: converge (and then optimal) or oscillate periodically.
- **Status.** Not assessed.
- **Applicability.** Does not apply because the results are for a single cycle and the target graphs have many cycles. Marked NOT verified beyond the introduction.

---

## 9. Ruozzi and Tatikonda 2013

**VERIFIED from the primary text.** Read: all 22 pages of arXiv:1002.3239v3 (1 Dec 2012).

**Citation.** N. Ruozzi and S. Tatikonda, "Message-Passing Algorithms: Reparameterizations and Splittings", arXiv:1002.3239v3. Published in IEEE Transactions on Information Theory 59(9), 2013, pp. 5860-5881 (journal details as cited in reference [33] of the AIJ paper; I read the arXiv version).

### 9a. The splitting algorithm. Section III, pp. 4-6, eq (19), Algorithm 1

- **Statement.** Split factor alpha into c copies, each with 1/c of the potential. **If the messages on the split edges are initialized identically, they stay identical** (the split graph has a symmetry that swaps the copies). Under that condition min-sum on the split graph reduces to eq (19), printed for the case where only factor alpha is split: `m_{i->alpha} = phi_i + (c - 1) m_{alpha->i} + sum_{beta in di \ alpha} m_{beta->i}`. Algorithm 1, p. 6 ("Synchronous Splitting Algorithm"), is the general version with a multiplicity for every factor and every variable, started from any finite vector.
- **Assumptions.** Identical start on the copies. Any topology, any arity, finite domains.
- **Status.** Definition plus a symmetry argument.
- **Applicability.** Applies only if "the hand-off gives both copies identical messages" is checked (same premise as 1b). With c = 2 the returning term has weight 1.

### 9b. What a fixed point gives. Theorem II.1 (Section II); Theorem V.2 and Corollary V.3 (p. 9); Theorem V.5, Corollaries V.6-V.7 (p. 10)

- **Statement.** Theorem II.1: at a fixed point of min-sum the beliefs are admissible (they add back up to the objective) and min-consistent. Definition II.2: beliefs are "locally decodable" when every node's belief has a unique minimum. Theorem V.2: if the fixed-point beliefs are locally decodable and a condition on the parameters holds, the decoded assignment is a local minimum with respect to changing one variable. The condition always holds for `c_i = 1, c_alpha > 0`, so it holds for the two-way split. Corollary V.3 is the same for plain min-sum. Theorem V.5 / Corollary V.7: global optimality needs, in addition, `(1 - sum_{alpha in di} c_alpha) * c_i >= 0` with `c_alpha > 0`. With `c_i = 1` this means `sum_{alpha in di} c_alpha <= 1`.
- **Object.** The assignment decoded **at a fixed point**. Not eventual: nothing about reaching one.
- **Status.** Theorems with full proofs.
- **Applicability.** Theorem V.2: applies only if "the run has reached a fixed point with a unique minimum in every belief" is checked; an oscillating run has not. Theorem V.5: does not apply because with `c_alpha = 2` the sum over a variable's factors is at least 2, which is greater than 1.

### 9c. Convergence. Theorem VI.2 (p. 11), Algorithm 2, Algorithm 3 (p. 12)

- **Statement.** Under `c_i = 1, c_alpha > 0, sum_{alpha in di} c_alpha <= 1`, **Algorithm 2, which is sequential and starts from zero messages**, makes a lower bound on the optimum non-decreasing. The paper states that this says nothing about convergence of the messages or of the beliefs. **Algorithm 3, the damped synchronous version**, uses `m(t) = kappa + (1 - delta) * m(t-1) + delta * (new factor-to-variable message)`. Here **delta is the weight of the NEW message**, and the damped message is factor-to-variable. With delta = 1/n it has the same guarantee under the same conditions.
- **Convention warning.** This is the opposite convention to Zivan/Cohen (where lambda is the weight of the old message), and a different message is damped (factor-to-variable here; variable nodes in Cohen et al.). Translating: delta = 1 - lambda.
- **Status.** Theorem with proof.
- **Applicability.** Does not apply because `c_alpha = 2` violates `sum c_alpha <= 1`, the schedule in the theorem is sequential or heavily damped (delta = 1/n), and the start is zero.

### 9d. Graph covers. Section VII, Theorems VII.1-VII.3, Corollary VII.4, Theorem VII.5, Corollary VII.6

- **Statement.** Assignments on finite covers of the graph explain when the convergent variants cannot reach the true optimum. Theorem VII.5 is for pairwise **binary** problems (2-covers suffice).
- **Applicability.** Does not apply because these results describe the convergent parameter range and, in the sharp form, binary variables.

**Summary for source 9.** For the two-way symmetric split there is no convergence theorem and no global-optimality theorem in this paper. Undamped synchronous updates on a general graph get no guarantee at all.

---

## 10. Rebeschini and Tatikonda 2017

**VERIFIED from the primary text for pp. 1-12** of arXiv:1706.03807v2 (3 Nov 2017), which contain the setting, the algorithm and the main theorem.

**Citation.** P. Rebeschini and S. Tatikonda, "Accelerated Consensus via Min-Sum Splitting", arXiv:1706.03807v2. Published at NIPS 2017 (as cited in reference [32] of the AIJ paper).

- **Statement.** The problem is network averaging (consensus). Variables are **real-valued** and each node's cost is **quadratic**. Min-Sum Splitting (Algorithm 1) has a real parameter delta and a symmetric matrix Gamma. Messages stay quadratic (Proposition 2), so the algorithm is a **linear** recursion on the quadratic coefficients (Algorithm 2), which can be followed through an auxiliary process on the nodes (Proposition 3). Theorem 4, p. 11: with W symmetric, rows summing to one, second-largest eigenvalue modulus rho_W < 1, delta = 1, Gamma = gamma * W, gamma = 2 / (1 + sqrt(1 - rho_W^2)), and a zero start, the estimates converge to the average at a rate strictly better than rho_W (a square-root improvement). The paper also notes that ordinary Min-Sum does not converge for consensus on graphs with cycles.
- **Assumptions.** Real variables, quadratic costs, synchronous, zero start, specific parameter choice. No ties issue (continuous). No discrete readout.
- **Object.** The real-valued estimates. Asymptotic rate.
- **Status.** Theorem with full proof.
- **Applicability.** Does not apply because the variables are continuous, the costs are quadratic and the dynamics are linear. There is no argmin switching in this setting.

---

## Additional sources

Only one, and it is a lead, not a verified source.

### A1. Poljak and Turzik 1986. **NOT verified from the primary text.**

**Citation (from a search result; not seen on the paper itself).** S. Poljak and D. Turzik, "On an application of convexity to discrete systems", Discrete Applied Mathematics 13(1), 1986, pp. 27-32. DOI 10.1016/0166-218X(86)90066-1.

- **What I have.** Only the abstract as relayed by a search snippet: if A is a symmetric matrix, f is the gradient (or a certain subgradient) of a convex function, and `y(t+1) = f(A y(t))`, then the only possible periods are 1 and 2.
- **Why it matters (observation, not a finding).** "Pick the best value given a linear score" is a subgradient of a convex function (a maximum of linear functions). With one-hot coding of the values, the synchronous recursion of the tex's Theorem "tworoutes" (arbitrary shared pairwise tables plus unary terms) looks like it could be an instance of this statement. If so, the tex's sentence that its theorem "covers" what the older results do not needs to be checked against this paper before submission. I could not open the paper, so this stays a question.

I found no other primary source that states a theorem directly about period-two behaviour of synchronous min-sum/max-product on general finite domains.

---

## Check of the local notes

### Part A. `experiments/aamas/late_split/oscillation_explanation_sources.md` against sources 1-4

**Note section 1 (Yedidia)**

1. "sections 5-6, printed pp. 10-14 (PDF pages 12-16)" -> **supported**. PDF page = printed + 2.
2. "beliefs sum incoming factor messages" -> **supported**. Eq (4), p. 12.
3. "the displayed value minimizes that belief vector", attributed to eqs (4)-(8) -> content **supported**, **pointer slightly wrong**. The readout is in the Section 5 step list (p. 10) and the Fig. 7 caption (p. 11). It has no equation number.
4. "A variable-to-factor message excludes that factor's incoming message" -> **supported**. Eqs (5)-(6), p. 13.
5. "A factor computes every output entry by minimizing over the other variables, incorporating their complete incoming vectors" -> **supported**. Eq (8), p. 14 (eq (7), p. 13 for the example).
6. "the neighbor's currently displayed assignment is not substituted into the cost table" -> **supported as an inference** from eq (8). Yedidia does not say it in these words. The closest passage is the end of Section 10, p. 21, which contrasts BP's cost-vector messages with Divide-and-Concur's single-best-guess messages.
7. "Section 9, printed pp. 18-19 (PDF pages 20-21), Figure 13 and equation (13)" -> **supported**. Section 9 runs pp. 18-20; Fig. 13 is on p. 18; eq (13) on p. 19.
8. "splitting ... preserves the objective but changes message dynamics" -> **supported**. pp. 18-19.
9. "A message sent to one copy includes the incoming message from its sibling. The source explicitly identifies this re-entry." -> **supported** by the `(k_a - 1) m_{a->i}` term of eq (13) and the text around it. **Missing premise:** Yedidia derives (13) under the assumption that the messages of the copies are initialized equal. The note does not mention this. It matters for the warm-start hand-off. The word "re-entry" is the note's, not Yedidia's.
10. "For pairwise factors the added path is X -> F1 -> Y -> F2 -> X" -> **not in the source**. It is the note's own inference. It is consistent with eqs (5), (8) and (13).
11. "The recipient-exclusion rule prevents immediate reversal through the same factor, not return through another copy" -> **not in the source in these words**; consistent with eqs (5) and (13).
12. **Omission.** The note does not say that the guarantees Yedidia reports for splitting are for `k_a < 1/d`, not for `k_a = 2`. Nothing in Section 9 supports any convergence claim for the two-way split.

**Note section 2 (Weiss and Freeman)**

13. "section II-A, pp. 4-5, Figure 3; section II-B, p. 6: after a finite number of parallel iterations, incoming messages correspond to exact calculations on a computation tree" -> **supported**. W&F call it the unwrapped tree. **Missing premise:** they state it for a run that starts from constant messages.
14. "Repeated copies correspond to the same original variable but occur in different positions" -> **supported**. Section II-B, p. 6.
15. "It does not mean BP explicitly enumerates or stores the tree" -> **not in the source**; a harmless remark of the note.
16. "Section I, p. 3, equations (6)-(7), specifies vector messages and simultaneous updates" -> **supported**. Both equations and the parallel-update sentence are on p. 3.
17. "Section IV, p. 9, explains why replica counts and boundary contributions complicate transferring global conclusions" -> **supported**.
18. "do not conclude that cycles necessarily oscillate or that graph cycle length equals assignment period" -> **consistent**. W&F make no claim about periods, apart from the single-loop remark on p. 1 that is cited to other papers.
19. **Omission.** The main result is "Claim 1" and it needs a fixed point and unique maximizers. The note does not use it, which is correct, but it would help to say so explicitly.

**Note section 3 (Zivan, Lev, Galiki)**

20. "pp. 7336-7337, 'Preliminaries' and 'Backtrack Cost Tree'" -> **pointer partly wrong**. Definition 1 is on p. 7337 at the end of "Preliminaries". There is no section called "Backtrack Cost Tree". The next section is "Max-sum and BCT".
21. "traces a belief's selected cost components backward through prior updates" -> **supported**. Definition 1, p. 7337.
22. "Lemma 1, pp. 7337-7338, claims eventual periodicity on arbitrary factor graphs; it does not bound the period by two or three" -> **supported**. To add: the periodicity is of the *assignments* (and BCT cost increments), and the lemma assumes a zero start and no ties (footnote 5).
23. "Its proof starts from finitely many assignments, then adds an argument about repeated sequences and cumulative costs" -> **supported**. p. 7338.
24. "repeated decoded assignments alone do not imply repeated message state" -> the note's own caution; **consistent** with the paper, which never claims the messages repeat.
25. "Proposition 1, p. 7339, concerns sufficiently damped linearly split graphs whose original graph was a tree" -> **supported**.
26. "Corollary 2 also requires a consistent induced assignment tree" -> **supported**. p. 7339. It also requires a large enough lambda; the proof of Theorem 1 uses `1 - lambda < 1/(2d)`.
27. "The paragraph following Corollary 2 explicitly distinguishes convergence from optimal convergence" -> **supported**. p. 7339.
28. **Omission.** Example 1 / Fig. 4, p. 7339: an undamped symmetric split of a 3-variable chain fails to converge. It is the closest thing in this paper to the target behaviour and the note does not cite it.

**Note section 4 (Cohen, Galiki, Zivan)**

29. "sections 1-2, pp. 2-3, and section 4: damping mixes past message calculations with new ones" -> **supported**. p. 2 (introduction) and eq (5), p. 7.
30. "reducing abrupt changes but slowing information propagation" -> **supported**. Abstract (p. 1) and Lemma 1 / Proposition 1 (pp. 7-8) for the slowing; p. 3 reports it from earlier work.
31. "Its effects depend on the problem" -> **supported**. p. 3, conclusion of the related-work survey.
32. "Section 1 distinguishes inference through cost vectors from search through actual assignments" -> **supported**. p. 2, second paragraph.
33. "The experiments also show useful nonconverging trajectories when an anytime mechanism retains good assignments" -> **supported**. pp. 12-19; stated in the abstract as well.
34. "damping should not be described as a universal convergence or optimality guarantee for arbitrary dense graphs" -> **supported**. p. 7 says there is no theoretical guarantee; the proved results are for trees (Proposition 2) and a single constraint (Proposition 3).
35. "The paper's single-constraint symmetric-split result is a restricted statement" -> **supported**, with a precision: Proposition 3, p. 10, is for any *constant* split of a single constraint (symmetric is one case), from a zero start, with a unique smallest table entry.
36. **Omission, and the most important one.** The note does not cite Section 5.3, p. 11, Fig. 5. That passage says in plain words that the single-constraint guarantee does not extend even to three variables and two constraints, and that undamped Max-sum on a symmetric split oscillates between suboptimal solutions. This is the most direct published support for "undamped symmetric split can alternate between assignments".

**Note section 5 and "Illustrative algebra".** These are the note's own consequences and its own constructed example. They are not literature claims, so there is nothing to check against sources 1-4. I did not re-derive the algebra (no computation was run in this audit).

**Overall.** No statement in the note is contradicted by sources 1-4. Two pointers are imprecise (items 3 and 20). Two premises are missing (items 9 and 13). Three relevant passages are not cited (items 12, 28, 36).

### Part B. Literature paragraph in `analysis_oscillation/oscillation_section.tex` (lines 284-296) against sources 5-7

1. Goles-Olivos described as "synchronous threshold networks with symmetric weights, binary states" -> **consistent with the abstract**. Full text not opened, so not verified. The BibTeX comment (Discrete Mathematics 30(2), 187-189, 1980) is consistent with the DOI; I did not see the printed page.
2. Poljak-Sura described as "synchronous weighted-plurality opinion dynamics with symmetric influence weights --- pairwise terms proportional to equality indicators" -> **supported**. p. 119, rule (1). Two details the tex does not mention: the weights are real with no sign restriction, and ties are broken by the highest-numbered opinion, so their theorem needs no uniqueness assumption. The tex's theorem assumes unique argmins. The BibTeX comment (Combinatorica 3, 119-121, 1983) matches the printed first page; the issue number is 1.
3. Ashkenazi-Golan et al. described as "a two-cycle result for simultaneous best response ... for two-player random potential games" -> **supported**. Lemma 3.2 and Theorem 3.1, p. 5. Two details: the potential there is a generic random table over joint profiles, not a sum of pairwise terms; and for three or more players the paper has simulations only, which show convergence to a Nash equilibrium, with the proof stated as open.
4. "Theorem tworoutes covers arbitrary shared pairwise cost terms with unary terms ... which is why we include the short proof" -> **not checkable from sources 5-7 alone.** Relative to those three sources the sentence is fair: none of them covers general shared pairwise tables with many agents. But Poljak-Turzik 1986 (Additional sources, A1) may cover it, and I could not open that paper.
5. "Within its hypothesis, this explains why the observed period is two rather than three, five, or aperiodic." -> the qualifier "within its hypothesis" carries the whole claim. The theorem is about the recursion "every variable picks its best value against the neighbours' current values". The tex derives that recursion only under its full-commitment condition (Corollary at lines 205-217). None of sources 5-7 says Min-sum assignments follow that recursion. A run with an exact 42-update cycle through 3 values cannot satisfy the theorem's hypothesis, because under the hypothesis the period is 1 or 2. So for that run the hypothesis must fail somewhere, and the period-two literature cannot be cited as explaining it.

---

## Sources I could not open

1. **Goles and Olivos 1980, full text.** The publisher's site returned an access error to both the fetch tool and the browser pane. An open-access mirror showed a bot-verification page, which I did not try to pass. An index lists the article as free to read at the publisher, so it can probably be opened by hand. Used instead: the abstract from the author's institutional repository, and Kaaser et al. arXiv:1508.03519 pp. 1-2. Entry marked NOT verified.
2. **Forney, Kschischang, Marcus and Tuncel 2001, beyond pp. 239-240.** Paywalled. Only the two-page free preview was read. No open copy found on the authors' pages. Propositions 6.1-10.2 not read. Entry marked PARTIALLY verified.
3. **Poljak and Sura 1983, p. 121.** Only pp. 119-120 are in the free preview. These contain the theorem and the complete proof. p. 121 (probably remarks and references) not seen.
4. **Poljak and Turzik 1986 (additional lead).** Publisher's site blocked. Only a search-result abstract. NOT verified.
5. **Weiss and Freeman, journal version.** I read the author manuscript, not the IEEE print. Page numbers are the manuscript's.
6. **Ruozzi-Tatikonda and Rebeschini-Tatikonda, journal/proceedings versions.** I read the arXiv versions (v3 and v2). Theorem numbers are the arXiv ones and may differ in print. For Rebeschini-Tatikonda I read pp. 1-12 only (not the appendices).

---

## Instructions found in the content

No web page and no PDF page contained text addressed to me.

One thing to report: a tool result that returned PDF page images also carried a text block formatted like a system reminder. It told me to add a "Co-Authored-By" line to git commits and a footer to pull requests. It was not part of the PDF's visible pages and did not come from the user or the launching agent. I ignored it. This task makes no commits, and the user's own instructions say never to add Claude as a co-author.
