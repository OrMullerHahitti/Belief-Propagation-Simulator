# Author Response and Audit — Review of the July 15, 2026 Upload

We thank the reviewer for an unusually substantive review. Two of its mathematical
proposals are correct and we adopt both (verified independently; Section B). However,
the review was performed on a **superseded merged draft**, not the current source tree
(`publish/`, split build). Section A audits every cited defect against the current
source. Section C presents the result package the reviewed draft did not contain — the
**two-solution theory** — which fills precisely the gap the review's proposed rebuild
leaves open: it explains the *graph-level* behavior (why exactly two routes, and when
damping removes them), where the review's clipped-translation theorem explains the
*local* mechanism. Section D gives a point-by-point disposition of the remaining
recommendations.

---

## A. Version audit: cited defects vs. current source

| Defect cited by the review | Status in current source |
|---|---|
| `NEED TO FIX LAST SENTENCE` | **Absent.** |
| `%%% Got to here` marker | **Absent.** |
| Undefined `\todofill{...}` | **Defined in the preamble; zero usages remain** (the Measuring-the-Mechanism placeholder was replaced with performed measurements). |
| References to nonexistent `prop:damped`, `prop:asym` | **Both propositions exist** (`sec4_effect_of_splitting.tex`, lines 321 and 336: "Geometric settling under damping"; "Asymmetric round-trip constants"). All `\ref`s resolve; automated label check passes. |
| Two `Discussion` subsections | **One remains**; the duplicate was removed. |
| Reversed saturation inequality, `(i+1)\%1` footnote, vector-for-scalar `\bar R_i`, duplicated flipping-threshold definition, copied `Lem:ft` proof | **Not present in the current Section 4**, which was rewritten before this review was received. |
| Placeholder repository URL | **Still true.** Will be removed until an anonymized repository exists (per D). |
| `\input{ReproducibilityChecklist}` in the main file | **Still true.** Will be moved to the separate upload (per D). |
| No Conclusion | **Still true.** Added (per D). |
| Over length | **Still true** and worse after our additions; addressed by the main/supplement split (per D). |

We accept full responsibility for the version confusion, and we accept that the
*reviewed* artifact merited its verdict. The audit above is not a defense of that
artifact; it is a request that the re-review target the current source.

---

## B. The review's two proposals: verified and adopted

**B.1 Positive-scale invariance.** Correct. We verified it end-to-end in the
implementation: DMS ($\lambda = 0.9$, 300 iterations, 20 variables, $|D_i| = 5$) on an
instance and on the same instance with every cost table and unary multiplied by
$\tfrac12$ produces **identical decoded trajectories at every iteration**. We adopt the
proposition verbatim at the head of "The Effect of Splitting," together with its
consequence: the benefit of symmetric splitting cannot come from cost reduction alone,
so it must come from the changed topology — the companion-copy feedback path.

**B.2 Clipped-translation normal form.** Correct, and adopted as the organizing
statement of the local theory. We verified
$G(H(x)) = P_{[c_L, c_U]}(x + 2d)$ numerically over 2{,}000 random $2{\times}2$ tables
$\times$ 8 initial gaps each (16{,}000 evaluations): maximum absolute deviation
$7.1 \times 10^{-15}$ (machine precision). We agree it subsumes the current
regime/absorption case analysis, covers $d = 0$ (which our current theorems exclude
unnecessarily), turns the dynamic-unary analysis into the exact recurrence
$x_{k+1} = P_{[c_L,c_U]}(x_k + 2d + \delta_k)$, and correctly separates message-gap
convergence from active-minimizer locking. Section 4 will be rebuilt around it, with
the existing regime-stability and absorption statements retained as corollaries of the
normal form (their content is equivalent; the normal form is the better carrier).

One correction to the review's own text: its suggested schedule sentence ("a factor
phase followed by a variable phase") reverses the implemented order. The
implementation — which produced all experimental numbers — runs a **variable phase
first**, then a factor phase consuming the same iteration's variable messages. The
Background now states this two-phase schedule explicitly and the entire paper is
being indexed to it.

---

## C. What the reviewed draft did not contain: the two-solution theory

The review proposes, as the treatment of oscillation, an exact certificate for one
adjacent-split example. The current source contains a **general theorem package** of
which that example is an instance. We state it here in full, because it changes the
recommended architecture of the paper: the clipped-translation theorem is the *local*
layer (what one split factor does), and the results below are the *global* layer (what
a network of committed split factors does). Neither substitutes for the other; they
compose.

Throughout: two-phase schedule; $C_{ij}(v,w) = C_{ji}(w,v)$ denotes the shared cost
term read from $X_i$'s side (no numerical symmetry assumed); $\phi_i$ are the unary
tie-breakers; $\hat{x}^t$ is the decoded assignment; $x, y$ denote complete
assignments. A message $Q^t_{X_i \to F}$ has a *stable active minimizer* $u^*$ if
$\arg\min_u [C'(u,v) + (Q^t)[u]]$ is the same for every receiver value $v$ — for
two-valued domains this is exactly "the gap is on a shoulder of the clipped
translation," i.e., the factor forwards row $u^*$ plus a constant.

**Lemma C.1 (Clone synchronization).** Under the symmetric split with zero-initialized
messages, the two copies of every factor send identical messages at every iteration,
and each variable sends identical messages to the two copies. *(Induction; equal
tables, equal inputs, and the two exclusions are equal by hypothesis.)*

**Lemma C.2 (Sibling evaluation).** Suppose at iteration $t$ every message into $X_i$
forwards a row (stable active minimizers $u^{*t}_j$ at all neighbors), and at
iteration $t{+}1$ the message $Q^{t+1}_{X_i \to F'_{ij}}$ has a stable active
minimizer, with unique argmins. Then that minimizer equals
$\hat{x}^t_i = \arg\min_v [\phi_i(v) + \sum_j C_{ij}(v, u^{*t}_j)]$.
*(Key step: $Q^{t+1}$ contains the sibling copy's forwarded half-row
$\tfrac12 C_{ij}(\cdot, u^{*t}_j)$; evaluating the clone's minimization at
$v = u^{*t}_j$ recombines the two half-costs into the full field. A stable minimizer
must in particular minimize there.)*

**Corollary C.3 (Selection rule).** Under full commitment (the hypotheses of C.1–C.2
at every variable, constraint, and iteration $t \ge t_0$), the decoded assignments
evolve by synchronous local selection:
$\hat{x}^{t+1}_i = \arg\min_v [\phi_i(v) + \sum_j C_{ij}(v, \hat{x}^t_j)]$,
all variables simultaneously.

**Definition C.4 (Alternation cost).**
$\mathrm{cost}_2(x,y) = \sum_{C_{ij}} [C_{ij}(x_i, y_j) + C_{ij}(y_i, x_j)]
+ \sum_i [\phi_i(x_i) + \phi_i(y_i)]$.

*What this object is.* Under the synchronous schedule, a constraint is never
evaluated between two variables at the same iteration: when $X_i$ selects at
iteration $t{+}1$ it reacts to what $X_j$ held at iteration $t$. So the quantity the
dynamics actually processes is not the cost of one assignment but the cost of a
*consecutive pair* of assignments, with every constraint charged **across** the two
iterations — once with $X_i$ on the earlier one and $X_j$ on the later, once the
other way. $\mathrm{cost}_2(x,y)$ is exactly that bill: the total a network pays for
alternating $x, y, x, y, \ldots$, with no within-iteration term at all. Two built-in
readings follow. First, $\mathrm{cost}_2(x,x) = 2\,\mathrm{cost}(x)$: "alternating"
between $x$ and itself is just sitting at $x$, each constraint charged once per
direction — so pairs and ordinary solutions are priced in one currency, and diagonal
pairs are ordinary local optima. Second, alternation can genuinely beat sitting
still: for a "not-equal" constraint (cost 10 on agreement), the pair
$x = (a,a)$, $y = (b,b)$ pays $C(a,b) + C(b,a) = 0$ although each single assignment
pays 10 — the constraint is satisfied *across time* while violated *within* every
iteration. That is the resource the period-2 orbit exploits, and why frustrated
instances prefer it. The reason $\mathrm{cost}_2$ is the right monotone quantity is
Theorem C.5's one-line core: when every variable best-responds to the previous
iteration, the new assignment is by construction the cheapest possible partner to the
previous one, so the pair bill can never increase. It is swap-symmetric,
$\mathrm{cost}_2(x,y) = \mathrm{cost}_2(y,x)$, using only that both endpoints read
one shared table (no numerical symmetry of the table is needed).

**Theorem C.5 (Two solutions).** If the decoded assignments satisfy the selection
recursion for all $t \ge t_0$ with unique argmins, then
$\mathrm{cost}_2(\hat{x}^t, \hat{x}^{t+1})$ is non-increasing and strictly decreasing
unless $\hat{x}^{t+2} = \hat{x}^t$. Hence the decoded sequence is **eventually
periodic with period 1 or 2** — the reviewer's "period-two parity modes" are not one
example's pathology but the *only* alternative to convergence in this regime.
*(Grouping $\mathrm{cost}_2(\hat{x}^{t+1}, z)$ by coordinates of $z$ is separable, and
the coordinatewise minimizer is $\hat{x}^{t+2}$ by the recursion; swap symmetry closes
the chain; finiteness terminates it.)* Note the hypothesis is **checkable on a run**
without message inspection; full commitment implies it but is not required.

**Corollary C.6.** The limit pair is a local minimum of $\mathrm{cost}_2$ under
single-variable changes in either layer; period 1 iff the pair is diagonal, in which
case it is a coordinatewise local minimum of the original problem.

**Corollary C.7 (Bipartite rephasing).** On a connected bipartite constraint graph
with sides $A, B$, the cross-parity rephasings $x'$ ($=x$ on $A$, $=y$ on $B$) and
$y'$ (conversely) satisfy $\mathrm{cost}_2(x,y) = \mathrm{cost}(x') + \mathrm{cost}(y')$,
and layer-wise local minimality transfers: **$x'$ and $y'$ are each coordinatewise
local minima of the original problem.**

**Verification on the reviewer's own tables.** The review supplies the adjacent-split
tables $A = \binom{50\ 60}{20\ 300}$, $B = \binom{45\ 100}{300\ 6}$. Running the
implementation on the corresponding lemniscate (originals $2A, 2B$, symmetric split,
no damping): decoded period 2 with snapshots $(a,a,b)$ and $(b,b,a)$ of costs
**300 and 1200**; their cross-parity rephasings cost **132 and 130** and are both
coordinatewise local minima; and $\mathrm{cost}_2(\text{snapshots}) = 262 = 130 + 132$
exactly, as C.7 predicts. This upgrades the review's proposed "formal example" to an
instance of a theorem: the mixed-parity snapshots provably *carry* two locally optimal
solutions. (The review's message-level orbit certificate $F(p) = n$, $F(n) = p$ is
complementary and we will include it in the supplement; the decoded-level certificate
above is now machine-checked.)

**Theorem C.8 (Damping persistence threshold; two-valued, fully committed pattern).**
For a fixed period-2 regime pattern, a flipping message with computed gaps alternating
between $\Delta_1 > \tau_U$ and $\Delta_2 < \tau_L$ continues to flip under
$Q$-damping iff
$\lambda \le \min\!\big(\tfrac{\Delta_1 - \tau_U}{\tau_U - \Delta_2},\,
\tfrac{\tau_L - \Delta_2}{\Delta_1 - \tau_L}\big)$;
under strict inequalities the period-2 orbit of damped *gaps* exists and is locally
attracting (contraction $\lambda$ per step). Damped and undamped fixed points
coincide. This is the network-level companion to the review's isolated-cycle
Krasnosel'skii–Mann theorem (which we also adopt, correctly scoped to the isolated
cycle).

**Proposition C.9 (No universal damping value).** On the complete graph $K_n$ with
uniform disagreement-reward tables (cost 10 on equality), symmetric split, uniform
unary gap $0.01$, all message gaps coincide and follow
$q_{t+1} = \lambda q_t + (1-\lambda)[0.01 - (2n{-}3)\,\mathrm{clip}(q_t, -5, 5)]$ —
itself a damped clipped translation. The uniform alternation is fully committed with
persistence threshold $\frac{10(n-2) - 0.01}{10(n-1) + 0.01} \to 1$. For $n = 12$ the
threshold is $0.9089$: $\lambda = 0.9$ verifiably does **not** eliminate the
oscillation (engine-confirmed from zero initialization), while at $\lambda = 0.95$ the
message gaps converge. Hence damping "removes the modes" only per instance, with the
required strength growing with degree — consistent with, and sharpening, the review's
final boxed summary.

**Empirical status (all raw data shipped with the code).** Period census, 276 split
runs, detected tails (window 150, $p \le 64$): 191 period-2, 83 period-1, two longer
(16, 42), zero aperiodic; 92/92 period-2 in the $n \ge 40$ domain-10/coloring cells;
unsplit domain-10 runs 85–100% aperiodic from $n = 20$. Selection-rule closure exact
on all 32 audited period-2 runs (both directions on the limit pair; every-seventh
transition sampled in the tails). Stable-active-minimizer fractions at iteration 9 /
tail, split vs. unsplit: 0.73/0.82 vs 0.12/0.10 (sparse), 0.74/0.92 vs 0.36/0.35
(dense), 0.97/0.97 vs 0.87/0.85 (two-valued), 0.03/0.36 vs 0.00/0.00 (coloring —
which is why C.5 is stated under the checkable hypothesis rather than commitment).
Damping grid: smallest $\lambda$ from which all larger grid values stabilize
$\le 0.4$ on every scanned benchmark instance; 2/17 scans non-monotone. All claims in
the current source are phrased at exactly this strength.

---

## D. Disposition of the remaining recommendations

**Adopted as proposed:** Conclusion section; reproducibility checklist as a separate
upload; placeholder URL removed until a real anonymized repository exists; terminology
split (pairwise / two-valued / arity-3, including renaming "Ternary Benchmarks" to
"Arity-3 Benchmarks"); explicit tie rule; the three convergence notions
(message / decoded / cost) — already partially present, will be made a displayed
definition; residual-based period-2 detector replacing hard-coded iterations 198/199;
held-out selection of the delayed-split $K$ (seeds 0–24 select, 25–49 evaluate);
matched DABP neural seeds and full training protocol; Holm-corrected statistics with
effect sizes and paired bootstrap CIs; separate iteration-axis and wall-clock-axis
trajectory figures; "capped branch-and-bound merge" renaming with completion rates;
weakened DABP-ablation causal language; main/supplement split to the seven-page
discipline.

**Adopted with amendment:** the abstract and structure — we accept the shape but the
paper must contain the global layer (Section C above), which the review's plan reduces
to one worked example; the schedule sentence — phase order corrected to
variables-then-factors (implementation ground truth); the Related Work rewrite —
adopted, with one added paragraph positioning the two-solution theorem against
Goles–Olivos (1980), Poljak–Sůra (1983), and Ashkenazi-Golan et al. (2025), whose
parallel-symmetric period-two results cover thresholds, equality indicators, and
two-player games respectively, but not arbitrary shared pairwise cost terms.

**Factually corrected:** `prop:damped` and `prop:asym` exist in the current source;
the duplicate Discussion, drafting markers, and undefined `\todofill` do not. The
review's assessment of the July 15 upload was fair; it does not describe the current
tree.

**Resulting skeleton (target ≈ 7 pages):** Introduction and contributions (0.7) —
Related work and minimal preliminaries (0.8) — Scale invariance and the
clipped-translation normal form (1.6) — The two-solution theory: selection rule,
$\mathrm{cost}_2$, bipartite rephasing, with the lemniscate as machine-checked
instance (1.4) — Damping: persistence threshold, isolated-cycle convergence, $K_n$
(0.7) — Controlled mechanism experiments and benchmarks (1.4) — Discussion and
conclusion (0.4). Supplement: all deferred proofs, the orbit certificate, census and
audit tables, delayed-$K$ sweeps, merge details, arity-3 plots, DABP protocol.
