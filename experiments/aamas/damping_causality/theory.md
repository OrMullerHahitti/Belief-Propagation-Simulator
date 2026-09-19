# Exact mechanism and scope of the damping experiments

## What the experiment establishes

There is a three-variable tree on which ordinary, undamped Min-sum converges,
but equal splitting creates an attracting two-cycle from the same zero-message
initialization. Damping changes the path through the active-minimizer regions and
enters a different, strictly attracting fixed-point region. Inside that region,
undamped updates converge faster: removing damping takes one update to the fixed
point. Thus damping is needed for the tested initialization and representation,
not for the existence or local stability of the successful fixed point.

The example proves an exact mechanism, rather than inferring it from a flat cost
curve. It does not prove that every split graph requires damping, that damping
always succeeds, or that it always improves cost. A small positive damping value
preserves the same bad cycle on this very example.

## Convention and exact split-factor response

Let a message difference be its label-1 value minus its label-0 value. Damping
means old-Q weight `lambda`, so `lambda=0` is undamped:

\[
q^{t+1}=\lambda q^t+(1-\lambda)G(q^t).
\]

Each original binary factor in the main example is

\[
C=\begin{pmatrix}16&0\\0&16\end{pmatrix},
\]

and each of its two equal clones has diagonal cost 8. For incoming difference
`q`, either oriented clone sends the exact factor-message difference

\[
r(q)=\min(0,8+q)-\min(8,q)=-\operatorname{clip}(q,-8,8).
\]

This identity includes the minimization itself. For `q<-8` or `q>8`, one sending
label is the strict active minimizer for both receiving labels: the factor
forwards one cost-table row. For `-8<q<8`, the active sending label depends on
the receiving label and the response has slope -1. The endpoints are ties.
These are precisely the paper's commitment and transition regions, not a
different approximate model.

Zero initialization and equal clones preserve clone synchronization. An outgoing
Q excludes its recipient clone but includes its sibling's previous R. Hence an
original edge contributes one return message to its own outgoing Q, and every
other original edge contributes two. Uniformly scaling an unsplit cost table
does not create these return dependencies.

The four-dimensional path recurrence is an exact reduction on that invariant
clone-synchronized subspace. Its displayed four-dimensional eigenvalues are
not automatically the spectrum under arbitrary clone perturbations. The two
main conclusions below also admit a direct full-state argument: at the
undamped B phase **both clones** of every outgoing message have strict constant
responses, so a sufficiently small perturbation, including unequal-clone
perturbations, is removed by the next factor response and following undamped Q
update. At the successful target, both clones have strict, constant responses
throughout an open neighborhood, so all Q differences obey the damped affine
contraction there. The argument uses strict active inequalities rather than
assuming clone symmetry persists under arbitrary perturbation.

All results concern message **differences**. Per-message additive normalization
preserves this algebra. Raw offsets need not converge. The exact reference
starts at `q^0=0` immediately after the native engine's first completed update;
that update has sent the fixed unary R messages. Reference update `k` therefore
corresponds to native completed update `k+1`.

## Three-variable path: exact cycle and exact convergence

Use the path `0--1--2`, the table above on both original edges, and unary tables

\[
\phi_0=(12,0),\qquad\phi_1=(13,0),\qquad\phi_2=(4,0).
\]

The original objective has unique optimum `101`, of cost 13; its next-best
assignment is `010`, of cost 16. Neither splitting nor damping changes this
objective.

Write the synchronized outgoing differences in order

\[
q=(a,b,c,d)=(q_{0\to01},q_{1\to01},q_{1\to12},q_{2\to12}),
\quad s(z)=\operatorname{clip}(z,-8,8).
\]

The entire nonlinear Q-difference update is the following four-dimensional map:

\[
G(a,b,c,d)=
\left(
-12-s(b),\;
-13-s(a)-2s(d),\;
-13-2s(a)-s(d),\;
-4-s(c)
\right).
\tag{1}
\]

The three decoded belief differences are

\[
B(q)=\left(-12-2s(b),\;-13-2s(a)-2s(d),\;-4-2s(c)\right).
\tag{2}
\]

A negative entry selects label 1, and a positive entry selects label 0.

### Undamped witness

From zero, equation (1) gives the exact finite trajectory

| Reference update | Q differences `(a,b,c,d)` |
|---|---|
| 0 | `(0,0,0,0)` |
| 1 | `(-12,-13,-13,-4)` |
| 2 | `(-4,3,7,4)` |
| 3 | `(-15,-17,-9,-11)` |
| 4 | `A=(-4,11,11,4)` |
| 5 | `B=(-20,-17,-9,-12)` |
| 6 | `A` |

Direct substitution proves `G(A)=B` and `G(B)=A`, so this is an infinite
two-cycle in exact arithmetic. At A the decoded beliefs are `(-28,-13,-20)`
and the assignment is `111`, costing 32. At B they are `(4,19,12)` and the
assignment is `000`, costing 61. All decoding gaps and all distances to clipping
boundaries are positive. Floating-point tie-breaking does not explain it.

The cycle itself is locally attracting. At B all four entries are strictly
below -8, so the Jacobian `DG(B)` is zero. At A only a,d are interior:

\[
J_A=DG(A)=
\begin{pmatrix}
0&0&0&0\\-1&0&0&-2\\-2&0&0&-1\\0&0&0&0
\end{pmatrix},\qquad J_A^2=0.
\]

Consequently `D(G composed with G)(A)=0`. Small perturbations that preserve the
two active patterns are removed after at most two undamped updates. It would
be incorrect to call this cycle an unstable eigenmode that damping merely
stabilizes. It is a stable nonlinear orbit in an undesirable basin.
For an arbitrary small consistent full-message perturbation near B, its first
factor response is constant and the following undamped Q/R pass returns to the
exact cycle. Extra unary-Q differences also settle after their constant input
is restored. This supplies full-message attraction, not just attraction inside
the four-dimensional invariant subspace.

At A the leaf messages are uncommitted; at B all messages are committed. This
main witness is therefore not an example satisfying the paper's full-commitment
hypothesis on both parities. Its force is that the same response/feedback
mechanism can be derived completely even when that hypothesis fails.
Nevertheless `111` is the strict synchronous best response to `000` and vice
versa. The paper's two-solution theorem assumes that weaker selection recursion,
so the theorem does apply to this selected-assignment cycle.

### A proof of damped convergence, including a finite entry certificate

For old-Q damping `lambda=1/2`, exact rational iteration reaches

\[
q^{12}=
\left(-\frac{8709}{512},\frac{4777}{512},
\frac{643}{64},-\frac{691}{64}\right).
\]

This point lies strictly inside the convex region

\[
\Omega=\{a<-8,\ b>8,\ c>8,\ d<-8\}.
\]

Throughout this entire region all factors forward fixed rows, and equation (1)
is the constant map

\[
G(q)=q^*=(-20,11,11,-12)\in\Omega.
\]

For every `0<=lambda<1`, `F_lambda(q)=lambda*q+(1-lambda)*q*` stays in Omega
by convexity and

\[
q^{12+k}-q^*=\lambda^k(q^{12}-q^*).
\tag{3}
\]

This proves convergence, rather than only finite-tail stability. Equation (2)
is constant throughout Omega, equal to `(-28,19,-20)`, so the assignment `101`
and its cost 13 are also stable, with strict decoding margins.
The factor R differences are already constant in Omega. The additional unary-Q
differences then average toward a constant input with rate lambda as well, so
the conclusion covers all message differences, not only the four reduced Qs.

The rational reference also verifies finite entry into this same invariant
region for the additional damping values used in the experiments:

| Old-Q damping | First reference update in Omega | Native completed update |
|---|---|---|
| .02 | 50 | 51 |
| .5 | 12 | 13 |
| .9 | 53 | 54 |
| .99 | 516 | 517 |

Each entry check is an exact rational inequality, and equation (3) proves the
infinite tail afterward. Thus .9 success and .99 slowness are also explained
on the original integer instance. This is a set of certified values, not an
assertion of monotonic behavior between them. In particular the separate .016
cycle below survives even though .016 exceeds the first pattern threshold.

The main causal result is basin entry: the fixed point and the region Omega
already exist when lambda is zero. Damping changes which clipping branches are
visited from zero and thereby reaches Omega. Once there, switching lambda to
zero reaches `q*` in one update; keeping lambda near one only delays message
settling. This explains the damping-on/damping-off intervention in the native
experiment. More generally, at a differentiable active pattern,

\[
DF_\lambda=\lambda I+(1-\lambda)DG.
\]

Here `DG(q*)=0`, so the successful fixed point is already locally fastest at
lambda zero. Fixed-point identities are independent of damping below one:
`F_lambda(q)=q` if and only if `G(q)=q`.

### Why some damping is insufficient on the same instance

Continuing the initial A/B clipping pattern under damping gives the exact
candidate pair

\[
A_\lambda=\left(
\frac{-4(5\lambda+1)}{1+\lambda},
\frac{31\lambda^2-6\lambda+11}{(1+\lambda)^2},
\frac{39\lambda^2+2\lambda+11}{(1+\lambda)^2},
\frac{4(1-3\lambda)}{1+\lambda}\right),
\]

\[
B_\lambda=\left(
\frac{-4(\lambda+5)}{1+\lambda},
\frac{11\lambda^2+42\lambda-17}{(1+\lambda)^2},
\frac{11\lambda^2+50\lambda-9}{(1+\lambda)^2},
\frac{4(\lambda-3)}{1+\lambda}\right).
\]

Both directions of the damped update hold exactly whenever their assumed
clipping patterns remain valid. The first inequality to fail is
`(B_lambda)_c<-8`, equivalent to

\[
19\lambda^2+66\lambda-1<0,
\quad\text{or}\quad
0\le\lambda<\lambda_*
=\frac{-33+2\sqrt{277}}{19}\approx0.01508599759.
\]

The other pattern inequalities are strict throughout this interval. Thus
lambda .01 preserves a strict two-cycle on the **same objective**. In the
stated pattern its two-step derivative is
`lambda^2 I+lambda(1-lambda)J_A`, whose eigenvalues are all `lambda^2`, so the
orbit is locally attracting. At the boundary one active minimizer ties; above
it this particular pair is no longer an orbit. Its disappearance alone is not
a proof of convergence above lambda*: another pattern could take over. The
lambda .5 convergence result instead has the explicit finite-entry and
invariance proof above.

The failure of that inference is visible on the **same path**. At lambda=.016,
or 2/125, there is a different exact pair

\[
A'=\left(-\frac{540}{127},\frac{7200583}{677418},
\frac{901}{84},\frac{40343}{21336}\right),
\]

\[
B'=\left(-\frac{2508}{127},-\frac{16473679}{1354836},
-\frac{1027}{168},-\frac{125645}{10668}\right).
\]

Direct substitution gives `F_.016(A')=B'` and `F_.016(B')=A'`. A' has the old
A pattern, but B' now has c strictly inside (-8,8), while a,b,d remain below
-8. The assignments are still `111` and `000`, with costs 32 and 61.

For this pattern `J_B` has the single nonzero entry `(d,c)=-1`. The two-step
Jacobian is `[lambda I+(1-lambda)J_B][lambda I+(1-lambda)J_A]`. Two eigenvalues
are lambda squared; the remaining pair has sum `T=3*lambda^2-2*lambda+1` and
product `D=lambda^4`. For every 0<lambda<1,
`1-T+D=lambda*(1-lambda)^2*(lambda+2)>0`, `1+T+D>0`, and `D<1`, which places
both roots strictly inside the unit circle. At .016 its spectral radius is
approximately .96876793235. The new pair is thus also locally attracting.
This displayed spectral calculation concerns the clone-synchronized recurrence.
It is not a transient mistaken for the original cycle. Its active inequalities
hold from the first threshold up to approximately .01716483570, where B's b
entry reaches -8; that boundary is the positive root of
`19*lambda^4+96*lambda^3+145*lambda^2+114*lambda-2=0`.

This establishes why a pattern-persistence threshold must not be advertised as
a convergence threshold, even on three variables. The exact pair and its
inequalities are checked using rational arithmetic in the focused tests.

## How this supports, and extends, the paper

The local manuscript's `sec5b_two_solutions.tex` identifies sibling return,
row forwarding, and synchronous local selection. Equations (1)--(2) isolate
these operations on an original **tree**, where unsplit Min-sum has no loop:
the loops responsible for the failure are introduced by the representation.
The native unsplit control, the sibling-return removal control, and the matched
damping intervention distinguish the feedback mechanism from cost scaling and
random instance difficulty.

The main path also gives an exact illustration of the paper's alternation cost
and bipartite rephasing. Its limit pair has `cost_2(111,000)=29`: every cross
edge has unequal endpoints and contributes zero, while the two unary totals
sum to 29. Rephasing the bipartition `{0,2}|{1}` gives `101` and `010`, costing
13 and 16; their sum is 29. Both are coordinatewise local minima. Damping enters
the lower-cost solution's basin in this witness, whereas the undamped raw
snapshots cost 32 and 61. The example explains how coherent information can
already be present in an oscillating run but decoded in incompatible phases.

For a direct fully committed example of the paper's selection rule, close the
path into a triangle and use diagonal cost 4 on every edge with unary gaps
`(-3,-3,-1)`. Undamped zero-initialized equal splitting reaches the uniform
assignments `111` and `000`, costing 12 and 19. Every outgoing message from a
given sender has alternating Q difference `(3,3,5)` or `(-9,-9,-7)` by sender,
strictly outside the transition interval `[-2,2]`. These two phases therefore
forward full selected rows, and each decoded assignment is the synchronous
best response to the other. A damped success on this triangle can be checked
empirically, but the proof-certified strict target basin above uses the path.

The additional experimental contribution is more specific than “damping
reduces oscillation”: **a locally attracting oscillation and a locally
attracting fixed point coexist; damping alters active-row transitions and
selects the fixed-point basin.** This supplies a concrete justification for
using damping during selection and reducing it after commitment is secure.
It does not yet provide an online rule guaranteed to detect secure commitment
on arbitrary instances.

## Three distinct damping calculations that must not be conflated

### Prescribed alternating input

For an externally prescribed drive alternating u,v, the exponential average
converges to the pair `(u+lambda*v)/(1+lambda)` and
`(v+lambda*u)/(1+lambda)`. Its persistent swing is the input swing times
`(1-lambda)/(1+lambda)`, equal to 1/19 at lambda .9. This is the paper's EMA
lemma. The output generally still alternates.

### An autonomous linear mode

If an undamped, locally fixed active map has eigenvalue mu, damping changes it
to `lambda+(1-lambda)*mu`. For mu=-1 the amplitude contracts by
`abs(2*lambda-1)`, equal to .8 at lambda .9. This is not the 1/19 forced-input
swing law. It applies only while the active pattern is unchanged. Positive or
complex feedback eigenvalues need not become stable under damping.

### A closed-loop committed alternation

In a specified committed pattern, the computed drive is alternating and the
EMA lemma determines candidate message differences. The candidate is an actual
orbit only if it retains all required active-minimizer inequalities. The
paper's committed persistence threshold, and the partly committed path
calculation above, are this extra self-consistency test. Crossing the threshold
forces a pattern change; it does not by itself force convergence.

## Counterexamples and qualifications

### A three-variable example above the committed threshold still oscillates

Take a regular degree-d graph with binary anti-equality cost 2a and identical
unary gap h. Exactly uniform zero initialization stays in a one-dimensional
invariant subspace. Its synchronized messages follow

\[
q^{t+1}=\lambda q^t+(1-\lambda)
\left[h-(2d-1)\operatorname{clip}(q^t,-a,a)\right].
\tag{4}
\]

A fully clipped two-cycle has values
`h +/- (2d-1)*a*(1-lambda)/(1+lambda)`. Strict persistence requires

\[
\lambda<\frac{2a(d-1)-|h|}{2ad+|h|}.
\tag{5}
\]

For a triangle, d=2, a=2, h=1, this threshold is 1/3. Nevertheless, at
lambda=2/5 equation (4) has the exact cycle

\[
q_-=-23/13,\qquad q_+=40/13.
\]

The negative phase is inside the transition interval and the positive phase
is outside it. The two-step derivative is `(-7/5)*(2/5)=-14/25`, so this
partly committed orbit is attracting within the uniform invariant subspace.
It is a direct counterexample to interpreting the loss of a fully committed
pattern as a convergence threshold.

### Message convergence need not imply assignment convergence

Inside the transition interval equation (4) has fixed point `q*=h/(2d)` and
slope `s_lambda=2d*lambda-(2d-1)`. It is strictly contractive there when
`lambda>(d-1)/d`; since the exterior slope is lambda, this also gives a global
contraction **of the scalar map on the uniform invariant subspace**.

For the same triangle at lambda=3/5 and q0=0, the entire trajectory remains
interior and has exact solution

\[
q^t=\frac14\left[1-(-3/5)^t\right],\qquad
B^t=1-4q^t=(-3/5)^t.
\]

Thus message differences converge geometrically, but the strictly nonzero
belief gap changes sign at every finite iteration. The decoded assignment
alternates forever in exact arithmetic, approaching a tied limiting belief.
Floating-point rounding may eventually hide this effect. A message residual
tolerance plus a stable-looking finite assignment tail is not a general
convergence theorem.

### No damping below one works uniformly over degrees

For a complete graph K12, degree d=11, use integer original diagonal cost 1000
(a=500) and unary gap h=1. Equation (5) gives threshold
`9999/11001 = 0.908917...`. At lambda .9 the exact committed alternating pair
is `1 +/- 10500/19`, so .9 is insufficient. These costs are just a positive
scale change of the local manuscript's cost-10, unary-gap-.01 example.

The threshold tends to one with degree. For every fixed lambda below one,
there is therefore a uniform committed two-cycle of sufficiently high degree.
Existence/local persistence and reachability from the stated initialization
are separate claims; native runs test reachability, while equation (5)
establishes the orbit and its active inequalities.

### The complete-graph scalar fixed point is transversely unstable

At the uniform interior fixed point of a degree-d regular graph, the directed
clone-synchronized Jacobian is

\[
(Jz)_{i\to j}=-z_{j\to i}-2\sum_{k\in N(i)\setminus j}z_{k\to i}.
\]

For an adjacency eigenvector v with eigenvalue eta, write
`z_(i->j)=alpha*v_i+beta*v_j`. On these coefficients the Jacobian acts as

\[
\begin{pmatrix}-2\eta&1-2d\\1&0\end{pmatrix},
\]

so the corresponding eigenvalues obey
`mu^2+2*eta*mu+(2d-1)=0`. For a complete graph with at least three variables,
nonuniform adjacency eigenvectors have eta=-1. Hence
`mu=1 +/- i*sqrt(2d-2)`. Under damping they become
`1 +/- i*(1-lambda)*sqrt(2d-2)`, of modulus strictly greater than one for
every lambda below one. Thus scalar convergence at lambda .95 in K12 is not
robust local stability under arbitrary nonuniform perturbations. It is an
exact symmetric-trajectory result, and its limiting decoded beliefs are tied.
The asymmetric path's strict basin certificate avoids both limitations.

## Reproducibility and evidence boundary

`theory_helpers.py` evaluates the displayed recurrences using Python rational
arithmetic. `tests/test_damping_causality_theory.py` checks the exact orbit,
finite entry certificate, strict margins, damping continuation, mixed-cycle
counterexample, and persistent decoding alternation. It also checks all Q and
belief differences against the native-arithmetic PairwiseKernel at damping
0, .5 and .9, with the one-update initialization offset accounted for.

The native-engine experiment runner independently checks the implementation
path, including unsplit controls and damping switches. These checks establish
the witness and its mechanisms; population frequency and benchmark relevance
require separately reported matched multi-instance experiments.
