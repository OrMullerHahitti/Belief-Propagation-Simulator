# Pro consultation: splitting, damping, and next checks

Date: 15 September 2026. Consulted visible model: **6 Pro**, in the user's
**ors research** project, using the signed-in Chrome session. Both answers
completed. After the one focused follow-up, the power selector visibly showed
**Instant, 1 of 5**, and the closed composer selector showed **Instant**.

[Complete conversation: Derive Min Sum Theory](https://chatgpt.com/g/g-p-69bebd20f0b88191bac2e4cef7ad5b07-ors-research/c/6aa95385-4780-83eb-851e-cc51a8fe8c95)

The complete first answer was captured through the page's Copy response action.
This file is a structured record, not a verbatim transcript. The browser's
content-export command was unavailable. The conversation preserves the original
prompt, complete proofs, and Pro's attached standalone script. That script was
not downloaded or executed; verification used independently written local code.

## Prompt and evidence supplied

The user requested an independent pure-reasoning attempt and a prioritized plan,
grounded in the AAAI splitting research rather than DABP. The prompt supplied
the exact Q/R recurrence and gauge convention; the three-variable path map,
attracting two-cycles and rational basin-entry certificates; sibling-feedback
controls; the triangle and fixed-damping counterexamples; the 32-seed sparse
split-pulse result and its stability qualifications; and the new commitment,
sign-balanced Jacobian, suboptimal four-cycle, and coherent-message candidates.

The primary question was whether graph, cost, split, and damping conditions can
prove entry into a cheaper stable solution, rather than merely certify a saved
trajectory. Convergence, strict decoding, and original-cost quality were
explicitly separated. The prompt requested adversarial checking and at most
five next experiments/proof tasks.

## First answer: mathematical conclusions

### Local attraction extends to finite alphabets

At a fixed point, assume every conditional factor minimizer and every decoded
variable label is unique. For each factor, its reparameterized table has
min-marginals equal to the variable beliefs, modulo constants. Its unique
global minimum therefore uses the decoded labels at both endpoints. Referencing
each Q to its sender's decoded label removes the negative reference derivative.
The full distinct-clone Jacobian becomes a nonnegative integer matrix.

Its active graph is acyclic if and only if it is nilpotent, if and only if the
specified fixed point is locally asymptotically stable for any fixed old-Q
damping below one. A cyclic graph has a Perron eigenvalue at least one that
damping cannot make smaller than one. At eigenvalue one, an affine neighborhood
contains a fixed-point segment, so attraction to that individual point fails.

This does not preclude convergence toward a fixed set. Pro independently gave
a one-edge example where damping suppresses an alternating mode inside one
unchanged active cell with strictly decoded beliefs. Nor does nilpotency imply
one-step norm contraction: nonzero nilpotent matrices can produce transients.

Local verification: [BEST_LABEL_GAUGE.md](BEST_LABEL_GAUGE.md), its exact ternary
test, and all six exact domain-10 endpoint certificates confirm these claims.
In the chosen-label gauge, all six matrices are 0/1 DAGs. Previously visible
signed cycles in the label-zero gauge disappear.

### Commitment provides a capture condition, not a global escape rule

Pro confirmed the necessary and sufficient consistent-commitment condition

`Delta_i(a) > max(w_e,1-w_e) * H_ij(a)`.

The live commitment cell is a convex polyhedron in Q. Inside it the undamped
map is constant. If its candidate fixed point also satisfies every strict row
inequality, the cell is invariant under fixed damping below one; decoding is
strict and the terminal cost is exactly the candidate assignment's original
cost.

For a changed weight, if a current row margin is `m0>0` but its candidate's
margin is `mstar<0`, then while the old pattern persists its margin is
`lambda^t*m0 + (1-lambda^t)*mstar`. This proves finite exit from that commitment
cell. It does not prove exit from the assignment or its basin. A zero limiting
margin does not guarantee finite exit.

### The cycle has a later, global threshold

Pro's four-cycle analysis distinguishes the first commitment boundary `7/8`
from the global uniqueness boundary `31/32`. A partly committed, locally
attracting bad fixed point survives at .95. A pulse can therefore change
active rows yet return to the same bad assignment after restoring .5.

Our independent [PRO_THRESHOLD_AUDIT.md](PRO_THRESHOLD_AUDIT.md) proves the
four-cycle boundary, exhibits the surviving bad branches, and handles the
actual stale-R warm restoration step. It also proves a cost/bias parameter
family, rather than only the numerical `J=4, h=1` fixture.

Pro additionally proposed a broader attractive-cycle theorem. For

`E(y) = J * sum_i 1[y_i != y_(i+1)] + sum_i h_i*y_i`,

with `n>=3`, `h_i>=0`, and `0<H=sum_i h_i<J`, let heavy and light capacities
be `c=alpha*J` and `d=(1-alpha)*J`. If `2*n*d<H`, it claims a unique fully
committed optimal fixed point and convergence from every finite initialization
for every fixed damping below one. The proof bounds light feedback, forces
heavy saturation around both directed cycles, and uses monotone lower/upper
trajectories plus uniqueness.

The pulse target lies strictly inside the equal-split good commitment cell,
so the pulse can end at a detected finite entry event. Equal splitting then
increases the target's worst commitment margin. This does not establish a
cold-start gain: this nonnegative-field family already favors the optimum
from zero. A sufficiently asymmetric fixed split also succeeds.

Pro gave an undamped finite-time bound with `delta=H-2*n*d>0`:

`n*(floor(J/delta)+1) + ceil((c-d)/(2*d)) + 2`.

It reported exact stand-alone .98-pulse entry counts of 48 undamped steps and
495 steps at damping .9 for the four-cycle bad-state initialization. These
counts are Pro's calculations, not native repository results. The arbitrary-
length theorem and time bound passed a separate independent audit, with the
initialization correction described below.

### The potential argument remains conditional

For fixed complementary weights, Q can sometimes be represented as an affine
function of coherent distributions over neighbor labels. If all outgoing
minimizations agree on one sender choice per variable, Q damping preserves
that representation and induces an EMA best-response step.

Expected original cost then changes exactly as `-eta*Gamma + eta^2*S`, with
`eta=1-lambda` and `Gamma>=0`. This gives conditional descent. A warm split
change can destroy coherence; commitment can also fail. Expected-cost descent
does not imply monotone decoded-cost descent. Pro recommended keeping this
secondary rather than using it to explain the pulse.

## Pro's proposed next checks

1. Audit the chosen-label-gauge theorem at strict fixed points, including a
   fixed-set counterexample. Negative entries in that gauge or strict damped
   contraction of a cyclic matrix would falsify the claim.
2. Check cycle thresholds, global bounds, and warm restoration on sizes 4, 6,
   and 10, with rational parameters, both sides of each threshold, and distinct
   clone initializations. Compare baseline, damping-only, fixed asymmetry, and
   pulses with and without damping.
3. Add one attached path, then mixed-sign unary fields. Require a cold-start
   baseline failure and a proved entry condition before increasing graph size.
   Compare with the strongest fixed split, not only .5/.9.
4. Test whether actual benchmark gains follow escape and capture. Classify
   isolated fixed points, fixed sets, and tied/boundary states. If most gains
   lack the proposed pattern, do not present full commitment as their general
   cause.
5. Freeze the rule before a fresh sparse/dense holdout. The 32 inspected seeds
   are development evidence for any new rule. Keep message budgets equal and
   retain all cost/stability failures. Do not claim an adaptive advantage if a
   strong fixed or fixed-timing control matches it.

## Focused follow-up

One follow-up supplied the exact completed-state schedule: after changing
weights, the first Q update uses retained old R; only the subsequent factor
phase uses the new weights. It asked whether the general capture condition
must check the actual proposed first Q, rather than current Q alone. It also
supplied the offset-growth failure, the successful per-update normalization
controls, the six exact endpoint certificates, and the open full-clone
single-edge cost-improvement example.

The complete follow-up was captured through Copy response. Pro explicitly
corrected its earlier Q-only capture claim for this schedule. With
`H(R)_i->f = u_i + sum_(g!=f) R_g->i`, the actual first update is

`Q_hat = lambda_new*Q + (1-lambda_new)*H(R_old)`.

The sufficient acceptance test is:

- `Q_hat` belongs to the new-weight strict commitment cell for y;
- y's stationary committed candidate belongs to that cell too;
- `E(y)` is below the incumbent original cost.

The first refreshed factor pass then forwards y's fixed rows. Subsequent
updates remain in the convex cell and converge to its candidate, with strict
decoding and terminal cost `E(y)`. Neither current Q's membership nor old R's
consistency with the new weights is required. The proposed Q can be checked
without changing stored messages. For a partly committed target, a separately
proved local basin can replace the fully committed cell; its own decoding and
convergence conditions must also be retained.

Pro revised the verification order to **numerical gauge parity, retained-R
switch parity, certified damping release, then performance comparisons**.
Normalizing stored messages must not silently refresh old R under new weights.
The raw arithmetic also needs comparison against direct excluded-recipient
summation; reducing offsets alone is not a general rounding-error proof.

Its specific next fixture is a four-cycle with one tunable anti-equality
diagonal and small mixed-sign dyadic unary gaps. Use exact native zero
initialization and enumerate all 16 assignments. Keep damping .9 initially;
test `.5 -> 63/64 -> .5`, returning only after the corrected capture check.
Compare equal update budgets with .5/.9, a prespecified fixed-weight sweep
including near-one settings, and matched fixed-timing pulses. Require a lower
terminal cost than both principal controls plus successful capture on fresh
nearby perturbations. If a fixed split matches the pulse, report mechanism
evidence rather than adaptive superiority.

## Independent audit of the broader cycle theorem

The separate audit is in
`results/damping_generalization_20260915/PRO_THEOREM_AUDIT.md`. It confirms the
uniqueness/global-convergence proof and the undamped time bound under the
stated attractive-cycle assumptions. The proof must explicitly use monotonicity
of the lower comparison sequence to keep heavy responses above the light
threshold while light responses lock.

The bound T starts from Q/R consistent with the new weights. A real warm
switch retaining old R requires **T+1** updates from intervention. Independent
rational checks on cycle sizes 4, 6, and 10 covered 72 consistent initial
states and six warm switches. A strict boundary counterexample confirms that
`H>2*n*d` cannot be weakened to equality. These are mathematical recurrence
checks, distinct from native floating-point performance claims.

## Source and novelty boundary

Pro cited Ruozzi and Tatikonda's
[Message-Passing Algorithms: Reparameterizations and Splittings](https://arxiv.org/abs/1002.3239).
The primary source's title, authors, and scope were checked. It studies
reparameterizations, splitting, graph covers, convergence, and optimality.
Neither that bibliographic check nor the Pro consultation establishes novelty
of the deductions here. A focused prior-art review remains necessary before
claiming a new theorem in the paper.

No manuscript, external publication, or core runtime was changed during this
consultation. Pro's advice is being reconciled with independently verified
mathematics and the real update schedule before adopting a controller design.
