# Binary active structure, damping, and fixed manifolds

Status: exact algebraic derivation, 15 September 2026. This strengthens the
sign-balanced local theorem in `THEORY_CANDIDATES.md` for positively replicated
binary pairwise factors. The fixed-manifold example below is also an important
limit on interpreting either theorem. These are statements about the specified
synchronous Min-sum message map, not a theorem for arbitrary domains or schedules.

## 1. State, schedule, and assumptions

Every variable has labels 0 and 1. Every pairwise factor has two distinct
variable endpoints. An original table may be represented by several distinct
factor clones with costs `w_f C`, where each weight is strictly positive and
fixed. Original unary costs may also be split; their response differences are
constant. All messages below are differences, label 1 minus label 0, so additive
normalization offsets have been removed.

Write `q_(i,f)` for the outgoing difference from variable i to factor f. After
the unary responses are fixed and the R responses are consistent with the Q
state, a synchronous Q -> R -> Q update is

\[
G_{i,f}(q)=h_i+\sum_{g\ni i,\;g\ne f}r_{g\to i}(q_{j(g),g}).
\tag{1}
\]

Here h_i includes the fixed unary contributions when f is pairwise. One may
eliminate all outgoing Qs to unary factors: they never affect a unary R, hence
never feed back into (1). If they are retained, they add zero Jacobian columns
and feed-forward rows only. Their limiting values are determined by the pairwise
state and constants. This elimination does not change the active cycles or the
nonzero spectrum. Isolated variables have no pairwise dynamics.

Old-Q damping is the fixed parameter `0 <= lambda < 1`:

\[
F_\lambda(q)=\lambda q+(1-\lambda)G(q).
\tag{2}
\]

Consequently G and F_lambda have exactly the same fixed points.

Assume q* is a fixed point at which every factor-to-variable update has a unique
minimizing sending label for each receiving label. This allows a tied decoded
belief; it excludes ties in the conditional factor minimizations. Finitely many
strict inequalities give an open neighborhood in which G is exactly affine:

\[
G(q^*+v)=q^*+Jv.
\tag{3}
\]

The local claims refer to the full state with distinct clones, not just an
invariant subspace in which equal clones have identical Qs.

## 2. A directed response cannot increase the sender's belief magnitude

Fix a factor f with table C_f. In this section its rows are receiving variable i
and its columns are sending variable j. Thus

\[
r_{f\to i}(q_{j,f})=
\min(C_{10},C_{11}+q_{j,f})
-\min(C_{00},C_{01}+q_{j,f}).
\tag{4}
\]

The nonzero derivative has sign

\[
\epsilon_f=\operatorname{sign}(C_{01}+C_{10}-C_{00}-C_{11});
\]

its magnitude is exactly one. The other possible derivative is zero. Positive
scaling preserves epsilon and moves the breakpoints only. An additive table
has epsilon zero and no nonzero response derivative.

At a fixed point, define the original variable's belief difference B_i by

\[
B_i=q^*_{i,f}+r_{f\to i}(q^*_{j,f}).
\tag{5}
\]

Equation (1) makes B_i independent of the incident factor used in (5), including
its clone index. Introduce the reparameterized 2-by-2 table

\[
H(x_i,x_j)=C_f(x_i,x_j)
+q^*_{i,f}\,x_i+q^*_{j,f}\,x_j.
\tag{6}
\]

The difference between its two row minima is B_i; the difference between its
two column minima is B_j. Adding q_i to a row or q_j to a column does not change
the conditional minimizers associated with the opposite message computation.

**Belief-magnitude lemma.** If the response from j through f to i has nonzero
derivative, then

\[
|B_j|\le |B_i|.
\tag{7}
\]

Equality holds exactly when the reverse response also has nonzero derivative.
If the common magnitude is positive, their signs obey

\[
\epsilon_f=\operatorname{sign}(B_i)\operatorname{sign}(B_j).
\tag{8}
\]

**Proof, with both orientations checked.** A nonzero j-to-i derivative means
the strict minima of the two rows occur in different columns. Permute the row
and column labels so that a global minimum is the first row's minimum H00 and
the other row's minimum is H11. Subtract the global minimum. The table becomes

\[
H=\begin{pmatrix}0&b\\c&a\end{pmatrix},
\qquad a\ge0,\quad b>0,\quad c>a.
\tag{9}
\]

The row-minimum gap has magnitude a. The first column minimum is 0 because
c>a>=0. The second column minimum is min(a,b). Hence the column-minimum gap
has magnitude min(a,b), proving (7). Strict conditional minima exclude a=b.
If b<a, both column minima are in the first row, so the reverse derivative is
zero and the inequality is strict. If b>a, the column minima are in different
rows, so the reverse derivative is nonzero and equality holds. This also covers
a=0, where b>a is automatic. When a>0 and b>a, the uniquely preferred row and
column correspond to the same member of the minimizing permutation. Restoring
the original labels gives (8): the permutation is preserving for epsilon +1
and reversing for epsilon -1. QED.

The word "sender" in this section refers to the input of the factor response.
For a Jacobian dependency `q_(j,g) -> q_(i,f)`, (7) orders the input variable's
belief magnitude below the output variable's, even though the actual numerical
message need not be ordered.

## 3. What any active cycle must look like

Order Q coordinates by the value |B_i| at the sending variable i. Every nonzero
entry of J goes from a smaller or equal magnitude to a larger or equal
magnitude, by (7). Thus J is block triangular across these magnitude levels.
Every directed cycle stays entirely within one level. Unequal levels can have
feed-forward dependencies but cannot form a directed cycle together.

At a positive level a>0, assign the coordinate sign `s_(i,f)=sign(B_i)`.
Equation (8) shows that every nonzero entry of that diagonal block becomes +1
after conjugation by the diagonal matrix of these signs. The resulting matrix
N_a is nonnegative with integer entries 0 or 1 in the full clone state. No
global balance assumption on the original interaction graph was used: balance
follows on each positive active level from fixed-point consistency itself.

If such a block contains a directed cycle, its spectral radius is at least one.
Perron-Frobenius supplies a real eigenvalue rho>=1. Block triangularity makes
rho an eigenvalue of the full J.

At level zero, every pair factor whose endpoints both have B=0 is active in
both orientations. To see this directly, the two row minima and two column
minima of H are all the same global value. Strict conditional minima imply that
the two minimizing cells occupy distinct rows and columns. They therefore form
a permutation and both derivatives are nonzero. In particular, this conclusion
holds for every positive clone of that original pair, whatever its weight.

Suppose such an original pair (i,j) has at least two positive clones f and g.
Their interaction signs coincide; call the sign epsilon. Define v by

\[
v_{i,f}=1,\quad v_{i,g}=-1,\quad
v_{j,f}=-\epsilon,\quad v_{j,g}=\epsilon,
\tag{10}
\]

with every other coordinate zero. The perturbed R responses sum to zero at
each endpoint. For an outgoing Q to f or g, excluding its recipient response
leaves exactly its corresponding component in (10). Every other outgoing Q
has zero change. Therefore

\[
Jv=v.
\tag{11}
\]

This is a full-state eigenvector, not only an eigenvector of the zero-level
diagonal block. It also leaves the aggregate belief differences unchanged.
Outgoing unary Qs, if retained, have zero change.

## 4. Replicated-binary local stability theorem

**Theorem.** Suppose every original pairwise factor is represented by at least
two positive clones, with fixed weights. At a fixed point with all conditional
factor minimizers strict, the following are equivalent:

1. The full active Q dependency graph is acyclic.
2. The fixed point is locally asymptotically stable without damping.
3. The fixed point is locally asymptotically stable with any specified fixed
   damping lambda in [0,1).

When these hold, undamped local convergence takes finitely many updates.
Original interaction signs may be frustrated; no bipartiteness or global sign
balance is required. The same theorem for sign-balanced interactions does not
require replication, as proved in `THEORY_CANDIDATES.md`.

**Proof.** If the active graph is acyclic, J is nilpotent. In a sufficiently
small neighborhood, the finite sequence of undamped iterates remains inside
the strict affine cell and reaches q* exactly. The damped matrix
`A_lambda=lambda I+(1-lambda)J` has only the eigenvalue lambda; its powers decay.
A sufficiently small invariant neighborhood therefore gives local asymptotic
stability for every lambda<1.

Otherwise an active cycle is contained in one magnitude level. At a positive
level, Section 3 supplies a real eigenvalue rho>=1 of J. If rho>1, A_lambda has
real eigenvalue `lambda+(1-lambda)rho>1`, hence the fixed point is unstable. If
rho=1, J has a nonzero fixed eigenvector v. Equation (3) shows that q*+t v is
itself fixed for sufficiently small positive or negative t, precluding local
asymptotic stability of the specified point.

If the cycle is at level zero, one of its pair factors joins two zero-belief
variables. Positive replication gives the eigenvector (10), and the same
nearby-fixed-point argument applies. These cases exhaust the possibilities.
QED.

"Locally asymptotically stable" includes both Lyapunov stability and convergence
to the specified point. Linear instability excludes that property; it alone
would not generally exclude convergence after a large excursion in a nonlinear
map. The theorem also does not assert that an entire fixed set cannot attract.

## 5. Damping can stabilize a fixed manifold without changing active rows

Consider a single binary pair with zero unary costs, represented by two equal
clones, each having table

\[
C_f=C_g=\begin{pmatrix}a&0\\0&a\end{pmatrix},\qquad a>0.
\]

For all incoming differences strictly inside (-a,a), each response equals -q
and its conditional minimizers are strict. In the full coordinate order
`q=(q_(i,f),q_(j,f),q_(i,g),q_(j,g))`,

\[
G(q)=(-q_{j,g},-q_{i,g},-q_{j,f},-q_{i,f})=Jq,
\qquad J^2=I.
\tag{12}
\]

For any `0<|b|<2a`, the point

\[
q^*=(b/2,-b/2,b/2,-b/2)
\tag{13}
\]

is fixed. Its decoded belief differences are B_i=b and B_j=-b, which are
strict and select an optimal anti-equality assignment. Thus this example does
not depend on a tied decoded solution.

Let `P_+=(I+J)/2` and `P_-=(I-J)/2`. Starting sufficiently near q*, all undamped
iterates remain in the same strict cell. For a perturbation e,

\[
G^t(q^*+e)=q^*+P_+e+(-1)^tP_-e.
\tag{14}
\]

A generic perturbation therefore produces a message two-cycle. Under damping,

\[
F_\lambda^t(q^*+e)=
q^*+P_+e+(2\lambda-1)^tP_-e.
\tag{15}
\]

For every `0<lambda<1`, the messages converge to the nearby fixed point
`q*+P_+e`. At lambda=1/2 convergence takes exactly one update. An infinity-norm
ball contained in the strict cell is invariant: J is a signed permutation and
F_lambda is a convex combination of I and J. Small enough perturbations also
preserve the strict decoded belief signs.

No active minimizer changes, and no attraction basin has to be crossed. Damping
suppresses the -1 modes transverse to a fixed manifold while preserving its
+1 modes. No individual point of that manifold is asymptotically stable, so
this does not contradict Section 4. It does disprove the broader assertion
that damping can help only by changing active regions or selecting a basin.

The clone redistribution direction in (10) preserves aggregate beliefs; other
directions within the fixed manifold can alter their nonzero magnitudes.
Removing ordinary per-message additive offsets does not remove this additional
message-level nonuniqueness.

### An open initial family with a strict improvement in original cost

The effect can improve original cost without any conditional minimizer change,
not merely improve a message residual. Take any initial full-clone state
q0=(u,v,w,z) whose four entries are in (0,a), and set

\[
S=u+v+w+z,\qquad D=u+w-v-z\ne0.
\]

This is an open set of full states, without a clone-synchronization assumption.
Undamped updates alternate between q0 and Jq0, whose entries are respectively
all positive and all negative. Their belief gaps both have the same sign, so
the decoded assignments alternate between 11 and 00. The original cost is
2a at every step. All iterates stay in the same strict interior active cell.

For 0<lambda<1, put m=2lambda-1. The exact trajectory and belief gaps are

\[
q^t=P_+q^0+m^tP_-q^0,
\qquad
B_i^t=(D-m^t S)/2,\quad B_j^t=(-D-m^t S)/2.
\]

After any t satisfying |m|^t S<|D|, the two belief gaps have opposite strict
signs and the original cost is zero forever. Such a finite t exists for every
0<lambda<1. At lambda=1/2 it is t=1. The whole interior cube is invariant under
the signed permutation J and its convex averaging, so no active cell is left.
The boundary case D=0 is deliberately excluded: its limiting decoded beliefs
tie. Also, zero initialization is outside this open family, so this is not a
claim that damping resolves a perfectly symmetric zero-start edge.

The implemented witness uses a=8 and q0=(1.25,.75,1,.5), giving S=3.5 and D=1.
The undamped cost stays 16; half damping reaches strict belief gaps (.5,-.5)
and cost zero in one pairwise Q update. Damping .2 and .9 reaches cost zero
at updates 3 and 6 respectively. The native-arithmetic PairwiseKernel checks
the full unsynchronized clone state, the exact modal formula, the costs, and
the continued strict interior inequalities in
`tests/test_damping_structural_theory.py`.

## 6. What this does and does not explain

There are at least two exact mechanisms in the binary setting:

- For convergence to an isolated, locally asymptotically stable strict fixed
  point, damping cannot create its local stability in the classes above. Any
  advantage must concern the trajectory reaching an already stable region.
- For convergence to a fixed manifold, damping can suppress oscillatory
  transverse modes while every conditional minimizer remains unchanged, as
  (15) proves. This may coexist with strict and stable decoded assignments.

Neither mechanism proves a preference for the globally best original cost.
Neither supplies a general threshold guaranteeing entry into a good region.
Conditional-minimizer ties, changing split weights, nonbinary domains and other
message schedules need separate arguments. Finite residual checks should also
be distinguished from convergence of messages, convergence to a specified
point, convergence to a fixed set, and stability of decoded assignments.
