# Exact obstructions to a universal damping explanation

These results are algebraic deductions for finite-domain pairwise Min-sum.
They do not assume that a small message residual proves convergence, and they
do not extrapolate one graph's basin-entry certificate to all graphs. They
identify the precise scope of an explanation based on reaching a fully
committed region.

## 1. An exact characterization of consistent committed fixed points

Write the original objective as

\[
E(x)=\sum_i u_i(x_i)+\sum_{\{i,j\}}C_{ij}(x_i,x_j).
\]

For the reverse orientation, use the transpose of the same table. Split edge
\(ij\) into two factors of weights \(w_{ij}\) and \(1-w_{ij}\), both strictly
positive. Unary factors may also be split; their total contribution remains
\(u_i\). Only Q is damped, with old-message weight \(0\le\lambda<1\); factor
R updates remain ordinary Min-sum. All statements concern message differences,
so additive message normalizations are immaterial.
Assume every variable has a pairwise neighbor; for a unary-only variable, add
the separate requirement that its chosen label strictly minimizes its unary.

Fix a candidate assignment \(x\), and define

\[
L_i(a;x)=u_i(a)+\sum_{j\in N(i)}C_{ij}(a,x_j),\qquad
\Delta_i(a;x)=L_i(a;x)-L_i(x_i;x),
\]

\[
D_{ij}(a,b;x)=C_{ij}(a,b)-C_{ij}(x_i,b)
             -C_{ij}(a,x_j)+C_{ij}(x_i,x_j).
\]

Here a **consistent, strictly committed fixed point at x** means that every
factor minimization over sender i selects the same label \(x_i\), uniquely,
for every receiving label, on every clone. This is stronger than merely having
a unique decoded assignment, and stronger than \(x\) being a strict
coordinatewise local minimum.

**Theorem.** Such a fixed point exists if and only if, for every directed
edge \(i\to j\), each clone weight \(v\in\{w_{ij},1-w_{ij}\}\),
every \(a\ne x_i\), and every receiving label b,

\[
\boxed{\Delta_i(a;x)+vD_{ij}(a,b;x)>0.}\tag{1}
\]

When it exists, its pairwise message differences are uniquely determined:

\[
R_{ij,v\to i}(a)=vC_{ij}(a,x_j)+\text{constant},
\]

\[
Q_{i\to ij,v}(a)=L_i(a;x)-vC_{ij}(a,x_j)+\text{constant}.\tag{2}
\]

Its belief differences equal those of \(L_i\), so its decoded assignment is x.
It is locally attracting for every \(0\le\lambda<1\).

**Proof.** Under consistent commitment, each factor forwards the selected row
of its weighted cost table, giving the first equation in (2). Summing all
incoming rows and excluding the recipient clone gives the second equation.
At a damped fixed point, \(Q=\lambda Q+(1-\lambda)Q^{\rm raw}\) implies
\(Q=Q^{\rm raw}\), because \(\lambda<1\). Substituting (2) into a factor
minimization and comparing candidate sender labels a and \(x_i\) gives exactly
(1), proving necessity. Conversely, if all inequalities hold, those constructed
messages forward the assumed rows and reproduce themselves, proving sufficiency.
Choosing \(b=x_j\) makes D zero, hence every \(\Delta_i(a;x)>0\); decoding is
strict. Strict inequalities persist in a neighborhood. There all R differences
are constant, and the subsequent Q differences average toward a constant target,
with factor \(\lambda\). The same reasoning covers unequal-clone perturbations
and the auxiliary unary-Q messages. No original-cost optimality assertion is
used in this proof.

The neighborhood is defined by linear inequalities in Q differences. On the
consistent Q/R manifold it is invariant, since the damped update is a convex
combination with the fixed point. For an arbitrary sufficiently small initial
Q/R perturbation, one factor pass restores the constant R differences and the
same conclusion then applies.

### Symmetric splitting maximizes this certificate

Let

\[
H_{ij}(a;x)=-\min_b D_{ij}(a,b;x)\ge0,\qquad
\beta_{ij}=\max(w_{ij},1-w_{ij}).
\]

Nonnegativity follows because \(b=x_j\) gives D=0. Taking the minimum over
receiving labels and both clones reduces (1) exactly to

\[
\Delta_i(a;x)>\beta_{ij}H_{ij}(a;x).\tag{3}
\]

Thus equal .5/.5 splitting maximizes every candidate's minimum commitment
margin among complementary splits. Unequal splitting cannot create a new
**consistently fully committed** assignment that was infeasible at .5/.5;
it can remove certificates or shrink their margins. This is a statement about
fixed-point feasibility, not basin sizes, finite-time performance, or all BP
fixed points. Asymmetric splitting can still steer trajectories more favorably.

## 2. A structural family with no consistently committed target

Consider a simple cycle of binary variables with original table

\[
C_{ij}=\begin{pmatrix}2a&0\\0&2a\end{pmatrix},\qquad a>0,
\]

equal splitting, and unary gaps \(h_i=u_i(1)-u_i(0)\) satisfying
\(|h_i|<2a\). For assignment x, let \(k_i\in\{0,1,2\}\) be the number of
incident edges with unequal endpoint labels, and let
\(\delta_i=u_i(1-x_i)-u_i(x_i)\), so \(|\delta_i|<2a\).
Flipping i has original local gap

\[
\Delta_i=4a(k_i-1)+\delta_i.
\]

For an unequal edge, \(H_{ij}=4a\), so (3) requires \(\Delta_i>2a\).
For an equal edge, \(H_{ij}=0\).

- If \(k_i=0\), the local gap is negative and even a strict best response fails.
- If \(k_i=1\), there is an unequal edge, but \(\Delta_i=\delta_i<2a\),
  so commitment is impossible.
- If \(k_i=2\), then \(\Delta_i=4a+\delta_i>2a\), and both constraints hold.

Consequently **a consistent strictly committed fixed point exists if and only
if the cycle has even length**, and its assignments are precisely the two
proper binary colorings. No amount of damping can produce a target of this
type on an odd cycle, because damping preserves the fixed-point equations.
By the preceding corollary, unequal complementary splitting cannot restore
such a target on these odd cycles either.

This does not exclude convergence to a partly committed fixed point. It does
prove that “damping eventually makes every factor forward a fixed selected
row” cannot be a universal explanation even on bounded-degree binary graphs.

The distinction from synchronous best-response closure is concrete. A triangle
with original diagonal cost 4 and all unary gaps -1 has strict coordinatewise
local minima: any assignment with two ones and one zero is one. Nevertheless,
the theorem proves there is no consistently strictly committed fixed point.
Strict local selection alone does not close the message self-consistency gap.

## 3. Damping cannot guarantee optimal original cost, even with strict convergence

Take the four-cycle with diagonal cost 4, off-diagonal cost zero, and

\[
u_0=(1,0),\qquad u_1=u_2=u_3=(0,0).
\]

At equal splitting, both alternating assignments have strict certificates:

| Assignment | Original cost | Minimum local gap | Minimum commitment gap |
|---|---:|---:|---:|
| 1010 | 0 | 8 | 4 |
| 0101 | 1 | 7 | 3 |

All costs are nonnegative, so cost zero is the global optimum. Yet the cost-one
assignment has an open attracting neighborhood for **every** damping value
below one. At the bad fixed point, the synchronized outgoing Q differences are
5 at sender 0, -6 at senders 1 and 3, and 6 at sender 2. Their absolute values
all exceed the clone threshold 2. Every factor therefore forwards a strict row,
and local contraction is exactly \(\lambda\). Damping cannot remove this
suboptimal attractor. Initialization or a split intervention must determine
which attractor is reached.

This counterexample is about the existence of an open suboptimal basin; it
does not claim zero initialization reaches that basin. Adding the same positive
constant \(\epsilon\) to the objective leaves all messages and dynamics
unchanged while giving the two costs \(\epsilon\) and \(1+\epsilon\). Hence
strict convergence alone supplies no universal multiplicative cost ratio.

### Losing the certificate does not prove that the bad assignment disappears

On the same four-cycle, increasing every first-clone weight to w invalidates
the bad full-commitment certificate when \(\max(w,1-w)\ge7/8\). The good
assignment remains certified for every weight strictly between zero and one.
It would nevertheless be wrong to infer that w=.95 forces improvement.

At w=19/20, the following **partly committed** Q fixed point still decodes
0101, with original cost 1. Edges are ordered 01,12,23,30; each pair of rows
contains first the weight-19/20 clone and then its weight-1/20 sibling; columns
are outgoing Q differences from the first and second listed endpoint:

\[
q=\frac15
\begin{pmatrix}
16&-21\\34&-36\\-18&20\\-36&37\\
20&-18\\37&-36\\-21&16\\-36&34
\end{pmatrix}.
\]

Each factor response is exactly \(-\operatorname{clip}(q,-4v,4v)\).
The resulting belief gaps are \((35,-37,38,-37)/5\), and subtracting the
recipient R from these beliefs reproduces every displayed Q. Every clipping
inequality is strict, but some responses are in the transition interval.
Therefore it is a fixed point for every damping value below one. Exact tests
check these equations without floating-point tolerances.

The split threshold is consequently a threshold for one **commitment pattern**,
not a guaranteed basin escape or cost-improvement threshold. This is the same
logical distinction that must be respected when a damping threshold destroys
one oscillatory pattern but another pattern remains available.

## Verification and scope

`code/obstructions.py` computes (1) using rational arithmetic for arbitrary
finite domains and ordered cost tables. The focused tests independently compare
the formula against direct factor minimization on asymmetric three-label tables;
enumerate every assignment on cycles of lengths 3--6; check the strict
best-response obstruction; verify both four-cycle attractors and their local
unequal-clone perturbation law using the native-arithmetic PairwiseKernel; and
check the partly committed .95 point with exact rational clipping.

The cycle parity assertion is proved for every cycle length; finite enumeration
is a check of the implementation, not the source of that general conclusion.
The suboptimal basin and split-certificate limits are exact counterexamples,
not evidence about their frequency in the paper's random benchmark families.

These results do not settle monotonicity of convergence in damping for arbitrary
instances. Nor do they characterize every partly committed recurrent region.
They provide a general target-feasibility test, an infinite class where that
target type is absent, and a proof that reaching a strict target does not by
itself imply globally optimal original cost.
