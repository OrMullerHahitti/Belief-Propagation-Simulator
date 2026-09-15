# The active graph in the decoded-label gauge

Status: algebraic extension derived and independently checked on 15 September
2026. This theorem applies to any finite variable alphabets. It requires both
strict conditional factor minimizers and unique decoded belief minima at the
fixed point. The neutral fixed-manifold mechanism remains possible and is
explicitly outside the conclusion about attraction to an individual point.

## 1. Fixed-point consistency fixes the reference minimizers

Use ordinary synchronous Min-sum, with fixed cost tables and old-Q damping
0<=lambda<1. Factors can be split into any fixed collection of positive weighted
copies. There are no learned weights multiplying incoming Q or R messages.
Work in message differences, so arbitrary additive normalizations disappear.
Factor scopes contain distinct variables. Unary response differences are
constant and may be included or eliminated.

Let q* be a fixed point, and write B_i for its variable belief vector. The
fixed-point equation for every incident factor f is, up to an additive constant,

\[
Q^*_{i\to f}(a)+R^*_{f\to i}(a)=B_i(a).             \tag{1}
\]

Damping below one does not change this identity because its fixed-point equation
Q=lambda Q+(1-lambda)Q_raw implies Q=Q_raw.

For a pairwise factor f joining i and j, define the reparameterized table

\[
H_f(a,b)=C_f(a,b)+Q^*_{i\to f}(a)+Q^*_{j\to f}(b).
\]

Its row and column minima are B_i and B_j respectively, each up to an additive
constant. Suppose every B_i has a unique minimizing label x_i*. Every global
minimizer of H_f must therefore have first coordinate x_i* and second coordinate
x_j*. Thus H_f has the unique global minimum (x_i*,x_j*).

In particular, when computing R_(f->i)(x_i*), its unique selected sender is x_j*.
This is a consequence of fixed-point consistency and unique decoded beliefs;
it is not an additional alignment assumption.

For a higher-arity factor the same proof uses

\[
H_f(x_f)=C_f(x_f)+\sum_{i\in f}Q^*_{i\to f}(x_i).
\]

Its min-marginal in each coordinate is B_i plus a constant. Unique variable
belief minima force its unique global minimizer to be x_f*. Consequently the
reference response R_(f->i)(x_i*) selects x_j* for every other variable j in f.

## 2. An exact change of coordinates removes negative derivatives

For each outgoing Q of variable i, use its decoded best label as reference:

\[
y_{i\to f}(a)=Q_{i\to f}(a)-Q_{i\to f}(x_i^*),
\qquad a\ne x_i^*.
\]

Freeze this reference choice at q*: this is a fixed linear coordinate change
for the local analysis, not a changing tie-break or modification of the algorithm.
The transformation between any two fixed-reference difference representations
is invertible and integer-valued in both directions. For example, from reference
0 to reference k, y_a=q_a-q_k, with q_0=0 and y_k=0.

Assume every factor conditional minimization, for every receiving label, is
strict at q*. These inequalities persist in an open neighborhood, in which G
is affine. For a pair factor, let sigma_(f->i)(a) be the sending label selected
when receiving variable i is fixed to a. Section 1 gives
sigma_(f->i)(x_i*)=x_j*. Therefore

\[
\delta\left[R_{f\to i}(a)-R_{f\to i}(x_i^*)\right]
=\delta y_{j\to f}(\sigma_{f\to i}(a)),           \tag{2}
\]

where the right side is zero when the selected label is x_j*. Thus this factor
response has only zero or positive unit coefficients in these coordinates.
Q aggregation sums such expressions without negative weights. The full active
Jacobian N is consequently a nonnegative integer matrix (entries 0 or 1 for
ordinary factors with distinct endpoints and full distinct message coordinates).

For a higher-arity factor, the right side of (2) is the sum of the corresponding
delta y coordinates over all other variables. It is still nonnegative and
integer-valued. Thus the result is not confined to pairwise factors.

If J is the Jacobian in the old fixed-reference gauge and T changes coordinates,
then

\[
N=TJT^{-1}.                                           \tag{3}
\]

Eigenvalues and nilpotency are unchanged. The presence of cycles in the graph
of individual signed entries need not be unchanged. In particular, apparent
signed cycle cancellations in a reference-0 Jacobian can become an ordinary
acyclic dependency graph in this decoded-label gauge. A structural claim about
feedback should specify the coordinates in which dependencies are measured.

## 3. Local stability has an exact graph characterization

**Theorem.** At a fixed point of finite-domain Min-sum satisfying the strictness
and unique-decoding assumptions above, the following are equivalent:

1. The active dependency graph in the decoded-label gauge is acyclic.
2. The undamped fixed point is locally asymptotically stable.
3. The fixed point is locally asymptotically stable for any specified fixed
   damping lambda in [0,1).

When they hold, undamped local convergence takes finitely many updates. When
they fail, no damping below one makes this individual fixed point locally
asymptotically stable. No binary-domain, sign-balance, bipartiteness, or factor
replication assumption is required.

**Proof.** If the graph is acyclic, N is nilpotent. The undamped affine map
reaches its fixed point after finitely many steps from a sufficiently small
neighborhood. The damped local matrix lambda I+(1-lambda)N has all eigenvalues
equal to lambda, and powers tending to zero, proving local asymptotic stability.

If the graph contains a directed cycle, nonnegative integer edge weights imply
rho(N)>=1. Perron-Frobenius supplies a real eigenvalue rho>=1. Damping changes
it to lambda+(1-lambda)rho. If rho>1, this exceeds one and gives instability.
If rho=1, there is a nonzero v with Nv=v. Since the map is exactly affine in an
open neighborhood, every sufficiently close point y*+t v is also fixed. The
specified point therefore cannot be locally asymptotically stable. These cases
exhaust the possibilities. QED.

The equivalent integer-Jacobian observation in `THEORY_CANDIDATES.md` proves
that undamped local stability implies nilpotency even with tied decoded beliefs.
The unique-decoding assumption here supplies the stronger nonnegative gauge,
and hence rules out damping creating stability of an individual point.

## 4. Necessary interpretation limits

- This is a theorem about a strict fixed point, not the itinerary followed by
  an initialized run. It supplies no global basin-entry guarantee.
- Cycles with Perron value one can permit damping to attract an entire fixed
  manifold. The single-edge example in `BINARY_ACTIVE_STRUCTURE.md` has strict
  decoded beliefs, reduces cost 16 to zero with damping, and never leaves its
  active cell. It does not contradict this theorem: no individual point of
  its manifold is locally asymptotically stable.
- A factor may be partly committed while the adapted dependency graph is
  acyclic. Complete row forwarding is sufficient but not necessary for local
  finite-step convergence.
- If a conditional minimum ties, there need not be an open affine cell. If a
  decoded belief ties, the proof that reference responses select a common
  decoded label fails. Neither case is covered by this theorem.
- These claims concern mathematical message differences. Overflow, additive
  offset growth, and cancellation in raw finite-precision messages require
  independent implementation controls.

## 5. A complete local affine classification

The adapted nonnegative matrix N gives a sharper classification than just the
absence of new individually stable fixed points. Let rho be its spectral radius.

1. If rho<1, integrality implies nilpotency. Undamped nearby trajectories
   converge in finitely many steps; keeping damping positive can slow settling.
2. If rho>1, Perron-Frobenius supplies a real unstable eigenvalue surviving every
   lambda<1. Fixed damping cannot make this local affine system stable.
3. If rho=1, every eigenvalue mu satisfies |mu|<=1. For 0<lambda<1, strict
   convexity of the unit disk gives
   |lambda+(1-lambda)mu|<1 whenever mu differs from +1. The +1 eigenspace remains
   unchanged. If +1 is semisimple, the powers of the damped matrix converge to
   its spectral projection P onto this eigenspace, so

\[
F_\lambda^t(q^*+e)\longrightarrow q^*+Pe.
\]

   Bounded matrix powers let a sufficiently small neighborhood remain inside
   the strict active cell. Its trajectories converge to a fixed manifold,
   preserving strict decoded beliefs. If instead +1 has a nontrivial Jordan
   block, damping retains a nonzero nilpotent part in that block and its local
   affine powers do not converge for arbitrary perturbations.

The third case includes the exact single-edge improvement: damping removes the
-1 oscillation while preserving +1 fixed directions. It can also include other
unit-circle modes. Jordan blocks at eigenvalues moved strictly inside the unit
circle do not prevent convergence; semisimplicity is required only at +1.

This classification is local. If a growing perturbation leaves the active cell,
the nonlinear map can still enter another region. The classification does not
assert nonconvergence after such an excursion or predict the eventual cost.

## Verification

The proof uses the exact Min-sum fixed-point and min-marginal identities and
received an independent mathematical audit, including the higher-arity case.
All six saved domain-10 strict fixed-point certificates were also transformed
exactly to the decoded-label reference. Every transformed matrix has only 0/1
entries and an acyclic dependency graph, with nilpotency indices 3,3,4,4,3,3.
Previously observed signed SCCs disappear under this change of coordinates.
The result is recorded under
`results/damping_generalization_20260915/local_stability/decoded_gauge/`.
These checks corroborate the algebra but are not its proof.
