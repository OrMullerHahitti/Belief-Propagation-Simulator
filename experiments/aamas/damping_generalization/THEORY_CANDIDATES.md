# Structural reasons damping can change convergence

Status: derived 15 September 2026. The claims in Sections 1–4 are proved below;
they are not conjectured extrapolations from the three-variable run. The local
Jacobian and global order arguments received an independent mathematical audit.
Section 5 is a narrower potential-game connection with additional assumptions,
not a general convergence proof for the native algorithm.

## 1. Exact temporal-parity decomposition on bipartite graphs

Consider a fixed pairwise factor graph whose original variable graph is
bipartite, with sides U and V. Domains can have any finite size. Split factors
may have any fixed weights. Work with normalized Q-message vectors, after the
unary factor responses have become constant, and the synchronous schedule
Q -> R -> Q. Write u for all outgoing Qs from U and v for those from V.

An undamped outgoing Q from a variable in U depends only on incoming R messages
computed from Qs sent by variables in V, and conversely. Therefore the complete
message map has the form

\[
G(u,v)=(g(v),h(u)),\qquad
G^2(u,v)=(g(h(u)),h(g(v))).                 \tag{1}
\]

This identity uses neither full commitment nor binary costs. Splitting adds
dependencies inside g and h through sibling return; it does not change (1).

**Hybrid-cycle theorem.** Let p=(p_U,p_V) and r=(r_U,r_V) be two distinct fixed
points of G. Then

\[
A=(r_U,p_V),\qquad B=(p_U,r_V)
\]

satisfy G(A)=B and G(B)=A. They are distinct: agreement on either entire side
would, by the fixed-point equations, force agreement on the other side.

**Proof.** The fixed-point equations are g(p_V)=p_U, h(p_U)=p_V and the analogous
equations for r. Substitution into G(A) and G(B) proves the claim. Both sides
of G squared in (1) evolve independently. If p and r are locally asymptotically
stable fixed points of G, r_U is a locally asymptotically stable fixed point
of g composed with h, and p_V is one of h composed with g. Their product is
locally asymptotically stable under G squared. Thus the hybrid cycle is also
locally asymptotically stable under G. The same argument applies to the other
phase. No linearization is needed. QED.

This gives a structural source of stable oscillations: two temporal parities
can independently settle on components of two different stable message
solutions. Raw simultaneous decoding can then combine incompatible assignment
components, as it does in the path example. Distinct message fixed points need
not decode different assignments, so poor decoded costs are not automatic.
The paper's bipartite rephasing theorem concerns selected assignments under
its local-selection hypothesis. Equation (1) is an unconditional message-level
identity for the stated schedule and bipartite topology.

For old-Q damping lambda, with eta=1-lambda,

\[
F_\lambda(u,v)=(\lambda u+\eta g(v),
                 \lambda v+\eta h(u)).                       \tag{2}
\]

For 0<lambda<1, its square generally has both within-side and cross-side
dependence: the independent parity dynamics in (1) are coupled. In particular,
F_lambda(A)=lambda A+eta B, so the exact hybrid pair above ceases to be that
same orbit when lambda>0. This does **not** rule out a displaced attracting
cycle: the existing .01 and .016 path certificates prove such cycles exist.
Damping removes the exact decoupling, not every possible oscillation.

## 2. The path has two stable solutions, not only one

Use the existing exact map, in order q=(a,b,c,d),

\[
G(q)=(-12-s(b),-13-s(a)-2s(d),
      -13-2s(a)-s(d),-4-s(c)),\quad s(z)=\operatorname{clip}(z,-8,8).
\]

The bipartition groups u=(a,d), v=(b,c). Direct substitution gives three strict
fixed points:

| Fixed point | Belief differences | Decoded assignment | Cost | Local behavior |
|---|---|---|---:|---|
| p=(-20,11,11,-12) | (-28,19,-20) | 101 | 13 | attracting, constant local G |
| r=(-4,-17,-9,4) | (4,-13,12) | 010 | 16 | attracting, local Jacobian squared zero |
| z=(-9/2,-15/2,-7/2,-1/2) | (3,-3,3) | 010 | 16 | unstable for every damping below one |

The claim here is that these three points exist, not an exhaustive classification
of possible fixed points on active-region boundaries.

At r only a and d are unsaturated. Its local Jacobian is

\[
J_r=\begin{pmatrix}
0&0&0&0\\-1&0&0&-2\\-2&0&0&-1\\0&0&0&0
\end{pmatrix},\qquad J_r^2=0.
\]

An explicit open invariant neighborhood for every 0<=lambda<1 is

\[
|a+4|<1/4,\quad |d-4|<1/4,\quad
|b+17|<1,\quad |c+9|<1.
\]

Here G sets a=-4 and d=4, while its b and c errors are strictly smaller than
3/4. Convex averaging preserves the box. Undamped updates reach r in at most
two steps; damped local powers converge because the only eigenvalue is lambda.
Thus damping can also converge to the suboptimal solution. It does not select
the better basin by theorem.

The known undamped cycle is exactly the hybrid of p and r:

\[
A=(r_U,p_V)=(-4,11,11,4),\qquad
B=(p_U,r_V)=(-20,-17,-9,-12).
\]

This explains its values and its attraction structurally, rather than merely
checking that two saved vectors happen to map to one another.

At z every coordinate is inside the clipping interval. The Jacobian is

\[
J_z=\begin{pmatrix}
0&-1&0&0\\-1&0&0&-2\\-2&0&0&-1\\0&0&-1&0
\end{pmatrix},
\quad
\det(\mu I-J_z)=(\mu^2-3)(\mu^2+1).
\]

The eigenvalue sqrt(3) becomes lambda+(1-lambda)sqrt(3)>1 for every lambda<1.
For lambda=.9 it equals approximately 1.07320508076. Damping does not stabilize
this interior fixed point, even though it damps the modes associated with
the other eigenvalues. A spectral explanation based only on negative modes
would miss this obstruction.

## 3. A general limit on what local stabilization can explain

Now assume binary domains. For an oriented pairwise table C, the factor-message
difference as a function of incoming difference q is

\[
r_C(q)=\min(C_{01},C_{11}+q)-\min(C_{00},C_{10}+q).
\]

Outside ties its derivative is either 0 or

\[
\epsilon_{ij}=\operatorname{sign}(C_{01}+C_{10}-C_{00}-C_{11}). \tag{3}
\]

Positive scaling of C changes the two breakpoints, but not the nonzero slope's
sign or magnitude. Reversing the table orientation preserves epsilon.
Zero-interaction tables have derivative zero everywhere and can be deleted
from the dependency analysis.

Call the interaction signs *balanced* if vertex signs s_i in {-1,1} satisfy
epsilon_ij=s_i s_j for every nonzero interaction. This includes all attractive
binary interactions and all anti-equality interactions on bipartite graphs.
It does not include an all-anti-equality odd cycle.

Consider a fixed point q* at which all factor-message minimizing choices are
strict. In an open neighborhood, G is affine with Jacobian J. Index Qs by the
sending variable and the recipient factor, retaining distinct split clones.
An entry J_(i->f,k->g) is nonzero only if g joins k and i, g differs from f,
and the response through g is in its nonconstant branch. Its value is epsilon_ki.
Let D_(i->f,i->f)=s_i. Then

\[
N=DJD
\]

is a nonnegative integer matrix, with entries 0 or 1 in the full distinct-clone
state. Summing synchronized equal-clone coordinates can produce entries 2, as
in the path above. Outgoing Qs to unary factors add feed-forward coordinates
only, because unary responses do not depend on their incoming Qs.

**Local dichotomy theorem.** Under these conditions the following are equivalent:

1. The active Q dependency graph is acyclic.
2. q* is locally asymptotically stable for the undamped update.
3. q* is locally asymptotically stable for any specified lambda in [0,1).

When these hold, the undamped map converges locally in finitely many updates.
When they fail, no damping below one makes q* locally asymptotically stable.

**Proof.** If the graph is acyclic, N and J are nilpotent. Strict margins ensure
an open affine neighborhood. For sufficiently small perturbations all finitely
many undamped iterates before J becomes zero stay inside that neighborhood,
then reach q* exactly. With damping, the local matrix is
A_lambda=lambda I+(1-lambda)J. Its eigenvalues all equal lambda, and its powers
decay; a sufficiently small invariant neighborhood supplies local convergence.

If the graph has a directed cycle of length k, the product of its integer edge
weights is at least one. In particular (N^(mk))_ii>=1 for every positive m,
so the spectral radius rho(N)>=1. Perron-Frobenius gives a real eigenvalue rho
with a nonnegative eigenvector. Similarity gives the same eigenvalue of J, and
A_lambda has eigenvalue lambda+(1-lambda)rho. If rho>1 this exceeds one and
the fixed point is unstable. If rho=1, let w be the corresponding nonzero
eigenvector of J. For sufficiently small t, q*+tw stays inside the strict affine
cell and G(q*+tw)=q*+tw. There is an actual nearby segment of fixed points, so
q* is not locally asymptotically stable for any damping. QED.

Consequently, in this broad class damping cannot turn a fixed point lacking
local asymptotic stability into one having it. Every individually asymptotically
stable fixed point it can reach was already locally asymptotically stable
without damping. For convergence to such an isolated stable target, changed
active regions and basin selection explain the benefit rather than a newly
stable target. This is a structural result extending beyond the three-variable
certificate. It does not apply at minimizing ties, to unbalanced interaction
signs, or to domains larger than two.

There is a distinct neutral-manifold mechanism that this theorem does not
exclude. For one zero-unary anti-equality edge with two equal clones of
diagonal cost a, write q=(q_i1,q_j1,q_i2,q_j2). While all coordinates are in
(-a,a), the undamped map is

\[
G(q)=(-q_{j2},-q_{i2},-q_{j1},-q_{i1}),\qquad J^2=I.
\]

Every q*=(b/2,-b/2,b/2,-b/2), with 0<|b|<2a, is fixed and has strict belief
gaps (b,-b). Generic nearby undamped initial states alternate, whereas
lambda=1/2 applies the projection (I+J)/2 onto the fixed-point manifold and
converges after one update. Nearby states converge to different fixed points;
no individual point is locally asymptotically stable. Thus damping can improve
message convergence even with strict decoding without entering another active
region. Do not summarize the local theorem as 'damping only changes basins.'

The theorem concerns fixed split weights. A temporary split intervention changes
the map and its active thresholds; after weights settle, the theorem applies
to that final map. Q damping conventions other than the synchronous collapsed
Q->R->Q update require their own state and Jacobian analysis.

## 4. A global statement: uniqueness makes damping unnecessary

For the same sign-balanced binary class, set H(z)=D G(Dz). Equation (3) shows
that H is globally order-preserving in the componentwise order, including at
ties by continuity of the piecewise-linear responses.

Each factor response is globally bounded:

\[
\min_u(C_{u1}-C_{u0})\le r_C(q)
\le\max_u(C_{u1}-C_{u0}).
\]

Finite degrees and fixed unaries therefore give a finite box [L,U] containing
the entire image of H. For any initial z0, enlarge this box to
ell=min(L,z0), u=max(U,z0). Then ell<=H(ell) and H(u)<=u. Iterating
F_lambda in the gauged coordinates from these corners gives respectively a
nondecreasing and a nonincreasing bounded sequence. Order preservation keeps
every trajectory from z0 between them. Continuity makes their limits fixed
points of F_lambda and hence of H.

**Global uniqueness theorem.** If G has exactly one message-difference fixed
point, every initialization converges to it for every fixed lambda in [0,1),
including no damping. QED by squeezing between the two corner limits.

Equivalently, undamped nonconvergence in this class implies multiple message
fixed points. Merely observing a unique decoded assignment would not establish
the hypothesis: different message fixed points may decode the same assignment.
No existence or uniqueness of a global optimum is needed in this theorem.

For the path, the gauge is D=diag(1,-1,-1,1). A box containing H's whole image is

\[
L=(-20,-11,-11,-12),\qquad U=(-4,37,37,4).
\]

L is already Dp, and H(U)=Dr=(-4,17,9,4), which is fixed. Thus the two stable
solutions p and r are precisely the extremal fixed points for this order.
The interior unstable point lies between them. The observed cycle mixes
coordinates of those two extremal solutions, consistent with Section 1.

## 5. Conditional potential connection; do not overextend it

There is another useful but more restricted calculation. Suppose a split,
fully committed trajectory has an EMA-consistent representation

\[
Q_{i\to ij}(a)=\phi_i(a)+\sum_{k\ne j}C_{ik}(a,p_k)
                    +\tfrac12 C_{ij}(a,p_j)+\text{constant},
\]

where p_i are distributions over labels and C(a,p)=sum_b p(b)C(a,b).
If each outgoing Q has a strict active sender independent of the receiving
label, averaging the commitment inequalities over p_j proves that sender is
the common unique best response b_i to the full field
phi_i+sum_j C_ij(.,p_j). Q damping then preserves this representation with

\[
p_i^+=p_i+\eta(e_{b_i}-p_i),\qquad\eta=1-\lambda.
\]

For expected original cost
E(p)=sum_i phi_i dot p_i+sum_ij p_i^T C_ij p_j, define d_i=e_bi-p_i,
R=-sum_i d_i^T grad_i E>=0 and K=sum_ij d_i^T C_ij d_j. The exact quadratic
expansion is

\[
E(p+\eta d)-E(p)=-\eta R+\eta^2 K.              \tag{4}
\]

Thus eta<R/K suffices for descent when K>0; when K<=0 and R>0 every positive
eta gives descent. First-order improvement competes with a second-order
interaction from simultaneous moves. The continuous-time limit has
dE/dt=-R<=0. This is related to classical best-response dynamics in potential
games; see Swenson, Murray and Kar, [On Best-Response Dynamics in Potential
Games](https://arxiv.org/abs/1707.06465), SIAM J. Control Optim. 56(4), 2018.
Their generic continuous-time convergence result does not itself establish
convergence for finite-step native Min-sum.

An arbitrary native Q state need not have this EMA-consistent representation;
zero initialization need not satisfy it for arbitrary costs. Commitment can
also fail during the very transition that damping changes. Therefore (4) is
not asserted as a global Lyapunov function for the native dynamics. It is a
candidate diagnostic only after these assumptions have been independently
checked. The exact results in Sections 1–4 need no such representation.

## 6. A finite-label algebraic criterion

This observation applies to ordinary finite-domain Min-sum, including domains
larger than two, in a strict active-minimizer cell. Use a fixed reference label
for each message's differences. Each R difference is the difference between
two selected affine expressions in incoming Q differences; its coefficients
are integers. Q aggregation sums these expressions. Therefore the undamped
active-cell Jacobian J has integer entries. Positive cost-table split weights
change the constants and active inequalities, not the integer derivative
coefficients. Learned noninteger weights on message aggregation would invalidate
this premise.

**Integer-Jacobian theorem.** If an integer matrix J has spectral radius less
than one, J is nilpotent. Consequently every individually locally asymptotically
stable strict fixed point of undamped finite-domain Min-sum has finite-step
local convergence.

**Proof.** Factor the characteristic polynomial as x^k p(x), where p is monic
with integer coefficients and p(0) is nonzero, unless all eigenvalues vanish.
The modulus of p(0) is the product of the moduli of all nonzero eigenvalues.
If every eigenvalue has modulus less than one and p has positive degree, this
product is strictly between zero and one, contradicting that p(0) is a nonzero
integer. Thus every eigenvalue is zero; Cayley-Hamilton makes J nilpotent.
Strict-cell local convergence follows as in Section 3. QED.

For a general signed J, fixed damping can create local asymptotic stability
only if every eigenvalue mu satisfies Re(mu)<1. More precisely, with eta=1-lambda,
the exact condition is

\[
0<\eta<\frac{2(1-\operatorname{Re}\mu)}{|1-\mu|^2}
\quad\text{for every eigenvalue }\mu,
\qquad 0<\eta\le1.                             \tag{5}
\]

An eigenvalue mu=1 rules out strict stability; an eigenvalue with Re(mu)>=1
cannot be rescued by fixed damping. Formula (5) follows by expanding
|1+eta(mu-1)| squared. It concerns an isolated fixed point within its strict
active cell, not a global basin-entry guarantee or attraction to a fixed set.
The replicated-binary theorem in `BINARY_ACTIVE_STRUCTURE.md` supplies the
stronger structural obstruction without requiring global sign balance.

## Evidence and remaining scope

- The three path fixed-point identities and spectra were evaluated directly
  from the displayed clipping map; the strictly interior spectrum is
  {sqrt(3),-sqrt(3),i,-i}, and r has J squared zero.
- An 81-pattern enumeration found the three strict path fixed points above;
  enumeration was diagnostic, not a substitute for the displayed proofs or
  a claim to exclude boundary fixed points.
- Independent audit verified the response sign, full-clone gauge, the actual
  fixed-point segment when rho=1, finite-step local convergence for a DAG,
  and the global order-box squeezing argument.
- Thirteen focused tests in `tests/test_damping_structural_theory.py` passed,
  including exact rational fixed-point and hybrid identities, the suboptimal
  invariant box, integer Jacobians, and a full-clone native-arithmetic example
  where damping lowers original cost without changing any active region.
- `BEST_LABEL_GAUGE.md` strengthens the local theorem to all finite alphabets
  when decoded beliefs are unique: an exact change of reference makes the
  Jacobian nonnegative. The signed dependency graph in an arbitrary reference
  can therefore give a misleading picture of the active feedback structure.
- These results explain a general source of oscillation and constrain the
  mechanisms by which damping can help. They do not predict which basin a
  given damping value reaches, guarantee a lower original cost, or prove
  convergence for arbitrary non-bipartite, multi-label paper benchmarks.
