# Source check: why Min-Sum assignments can keep switching

Checked 2026-09-21. This note supports the general mechanism, not a causal diagnosis of seed 0 or a classification of exactly two versus three visited values. No simulations or interventions were run for this source check.

## 1. Messages are vectors; the displayed assignment is only a readout

[Yedidia, 2011, MERL TR2011-087](https://www.merl.com/publications/docs/TR2011-087.pdf), sections 5–6, printed pp. 10–14 (PDF pages 12–16), equations (4)–(8): beliefs sum incoming factor messages; the displayed value minimizes that belief vector. A variable-to-factor message excludes that factor's incoming message. A factor computes every output entry by minimizing over the other variables, incorporating their complete incoming vectors. Thus the neighbor's currently displayed assignment is not substituted into the cost table.

Section 9, printed pp. 18–19 (PDF pages 20–21), Figure 13 and equation (13): splitting a factor into two half-cost copies preserves the objective but changes message dynamics. A message sent to one copy includes the incoming message from its sibling. The source explicitly identifies this re-entry. For pairwise factors the added path is `X → F1 → Y → F2 → X`. The recipient-exclusion rule prevents immediate reversal through the same factor, not return through another copy.

## 2. The feedback is an unfolded dependency calculation

[Weiss and Freeman, 2001, author manuscript](https://people.csail.mit.edu/billf/publications/Max-product_Belief_Propagation_Algorithm.pdf), section II-A, pp. 4–5, Figure 3; section II-B, p. 6: after a finite number of parallel iterations, incoming messages correspond to exact calculations on a computation tree formed by recursively copying earlier dependencies. Repeated copies correspond to the same original variable but occur in different positions of this tree. This explains how old preferences can return through cycles and participate in later conditional estimates. It does not mean BP explicitly enumerates or stores the tree.

Section I, p. 3, equations (6)–(7), specifies vector messages and simultaneous updates. Section IV, p. 9, explains why replica counts and boundary contributions complicate transferring global conclusions from that tree to the original graph. Use this interpretation to explain delayed, coupled estimates; do not conclude that cycles necessarily oscillate or that graph cycle length equals assignment period.

## 3. Relevant DCOP results and their scope

[Zivan, Lev and Galiki, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6227/6082), pp. 7336–7337, “Preliminaries” and “Backtrack Cost Tree”: traces a belief's selected cost components backward through prior updates. The BCT is useful for tracing which conditional minimizers and contributions produced a score.

Lemma 1, pp. 7337–7338, claims eventual periodicity on arbitrary factor graphs; it does **not** bound the period by two or three. Its proof starts from finitely many assignments, then adds an argument about repeated sequences and cumulative costs. This note does not independently validate that proof and does not use it as a finite-state shortcut: repeated decoded assignments alone do not imply repeated message state.

Proposition 1, p. 7339, concerns sufficiently damped linearly split graphs whose **original graph was a tree**. Corollary 2 also requires a consistent induced assignment tree. Neither directly certifies the dense pilot. The paragraph following Corollary 2 explicitly distinguishes convergence from optimal convergence.

## 4. Damping is a change to the evolving estimates

[Cohen, Galiki and Zivan, Artificial Intelligence 279 (2020), author copy](https://tzin.bgu.ac.il/~zivanr/files/SplittingAIJ2020.pdf), sections 1–2, pp. 2–3, and section 4: damping mixes past message calculations with new ones, reducing abrupt changes but slowing information propagation. Its effects depend on the problem. Section 1 distinguishes inference through cost vectors from search through actual assignments. The experiments also show useful nonconverging trajectories when an anytime mechanism retains good assignments. Consequently, damping should not be described as a universal convergence or optimality guarantee for arbitrary dense graphs. The paper's single-constraint symmetric-split result is a restricted statement, not a general theorem about every split graph and every warm-start state.

## 5. Exact consequences to use in the explanation

The following are direct mathematical consequences of the update/readout rules, rather than additional empirical or literature claims:

- Write `B_i^t(v)` for the belief score and `x_i^t = argmin_v B_i^t(v)`. A switch occurs when the winning ordering changes; exact ties are unnecessary. Message updates can change continuously within a region with fixed internal minimizers, while the discrete argmin changes abruptly at its boundary.
- If a subset `S` satisfies `B_i^t(v) > min_{u in S} B_i^t(u)` for every `v` outside `S` throughout an interval, only values in `S` can appear. A small observed set therefore means the other candidates remained above the winning envelope during that interval. The equations do not prescribe a universal set of two or three contenders; why a particular instance maintains those inequalities requires further analysis.
- The full evolving state includes message vectors, with real-valued score differences even after additive normalization. A finite output alphabet does not make the decoded assignments a closed finite-state dynamical system. A constant assignment can coexist with moving messages; a repeated assignment can have a different successor on its next visit.
- Independently minimizing each variable's belief can combine incompatible hypothetical neighbor choices. There is no global-cost acceptance test in these message updates. Hence a belief winner is not a certificate that the next decoded joint assignment improves the original objective.
- A graph feedback path allows recurrent influence; it does not by itself prove sustained oscillation, a specific period, or a specific small set of winners.

## Illustrative algebra checked for the accompanying explanation

This is a constructed warm state, not a new experimental result or a claim about the pilot. Let two variables have values `A,B,C`, shared cost matrix `[[4,0,20],[0,6,20],[20,20,20]]`, and two identical half-cost factors. If all incoming factor messages equal `r=[0,1,10]`, every outgoing variable message to a clone equals its sibling message. Applying the factor update gives `T(r)=[1,0,10]`; applying it again returns `r`. Beliefs are `2r`, so the strict winner alternates `A,B,A`, while the third value remains dominated. The decoded original costs are `4,6,4`. This verifies that unchanged tables and unique per-step winners can still support switching and nonmonotone original cost.

The initial message state is deliberately specified and is not claimed reachable from the pilot or zero initialization. The original objective has two tied optima, `(A,B)` and `(B,A)`, both cost zero; the displayed orbit itself has no belief ties. Do not present this example as refuting convergence theorems with additional initialization or uniqueness hypotheses.
