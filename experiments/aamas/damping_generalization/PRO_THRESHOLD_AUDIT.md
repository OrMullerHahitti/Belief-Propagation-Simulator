# Independent audit of the four-cycle threshold 31/32

The threshold was mentioned in an external consultation. The derivation below
was completed independently from the exact message equations before adopting
the claim. It is a threshold for **uniqueness of the fixed point and guaranteed
improvement from every initialization on this particular objective**. It is not
a general damping threshold or a threshold inferred from a finite run.

## Objective, gauge, and update convention

Use the four-cycle 01,12,23,30, original binary tables with diagonal 4 and
off-diagonal zero, and unary \(u_0=(1,0)\), all other unaries zero. Assignment
1010 has globally minimal cost zero; 0101 has cost one.

Let \(w\ge1/2\) be the heavier clone weight, with the same clone designated
heavy throughout any changing-weight schedule, and define

\[
t=4w,\qquad s=4(1-w),\qquad t+s=4.
\]

Multiply every Q and R difference at recipient/sender i by
\(\sigma_i=(1,-1,1,-1)_i\). In this gauge the interactions are attractive:
the response of a clone of threshold c is \(\operatorname{clip}(q,-c,c)\).
The unary gaps are \(h=(-1,0,0,0)\). Positive belief gaps at all vertices
decode the bad original assignment 0101; all-negative gaps decode 1010.

Q damping has any fixed old-message weight \(0\le\lambda<1\). It does not
change the fixed-point equations. Unary R responses are constant, and the
Q/R phase is the completed native step used by the other experiments.

## Sufficiency: above 31/32 the good fixed point is unique

At any fixed point, follow one orientation of the four strong-clone edges.
Its factor-message differences satisfy

\[
m_i=\operatorname{clip}(m_{i-1}+f_i,-t,t),\qquad
f_i=h_i+v_i,
\]

where \(v_i\) is the sum of the two incoming weak-clone R differences.
Since each weak R lies in \([-s,s]\),

\[
f_i\le h_i+2s,\qquad \sum_i f_i\le-1+8s.
\]

If \(w>31/32\), then \(s<1/8\), so this sum is strictly negative.
There must be a lower-saturated message \(m_i=-t\) on each oriented cycle.
Otherwise clipping can only lower the unclipped update, hence
\(m_i-m_{i-1}\le f_i\) for every i. Summing gives the contradiction
\(0\le\sum_i f_i<0\).

Starting from one lower-saturated message, every step can increase its value
by at most \(2s\). Indeed \(m\ge-t\) implies
\(\operatorname{clip}(m+f_i,-t,t)\le m+2s\) when \(f_i\le2s\).
Every message is at most three steps away, so on both orientations

\[
m_i\le-t+6s.
\]

Every belief and outgoing weak-clone Q therefore obey

\[
B_i\le-2t+14s<0,\qquad
Q_{i\to\mathrm{weak}}\le-2t+15s<-s.
\]

The inequalities hold throughout \(w>31/32\): the second requires only
\(t>8s\), which is weaker. Thus all weak R differences equal \(-s\).
Now every strong effective field is \(f_i=h_i-2s<0\). A lower-saturated
strong message propagates exactly \(-t\) around its cycle, so **all** strong
R differences equal \(-t\).

This determines every message difference uniquely:

\[
R_{\rm strong}=-t,\quad R_{\rm weak}=-s,
\]

\[
Q_{i\to\rm strong}=h_i-t-2s,\qquad
Q_{i\to\rm weak}=h_i-2t-s.
\tag{1}
\]

Both inequalities required for these negative clipping branches are strict.
Thus the displayed point exists and is the only fixed point. It decodes 1010.

The gauged message map is continuous, order-preserving, and has bounded image.
An enclosing lower corner iterates upward, an enclosing upper corner downward;
their limits are fixed points. Uniqueness makes these two limits equal and
squeezes every trajectory between them. This argument applies to
\(F_\lambda(q)=\lambda q+(1-\lambda)G(q)\) for every fixed
\(0\le\lambda<1\). Hence **every initialization converges to the good fixed
point, including undamped iteration**. See also the general uniqueness theorem
in `THEORY_CANDIDATES.md`.

## Necessity: a bad fixed point exists at and below 31/32

Assume instead \(1/2\le w\le31/32\). Set every weak R difference to \(+s\).
Then the effective strong fields are \(-1+2s\) at vertex zero and \(+2s\)
elsewhere. In either orientation, order the outgoing strong messages by their
distance after leaving vertex zero, and set

\[
m_0=\min(t,t-1+2s),\quad
m_1=\min(t,t-1+4s),\quad
m_2=\min(t,t-1+6s),\quad m_3=t.\tag{2}
\]

The recurrence closes because \(8s\ge1\): after the last positive field,
the message reaches the upper clip t. The first update is
\(\operatorname{clip}(t-1+2s,-t,t)=m_0\); all displayed messages are positive,
so lower clipping never intervenes. Both orientations use the same list in
opposite vertex order.

Every message in (2) is at least 2: \(t\ge2\), and
\(t-1+2s=7-4w\ge25/8\). Thus the outgoing weak Q satisfies

\[
Q_{i\to\rm weak}
=h_i+\text{two incoming strong R}+s\ge3+s>s.
\]

The assumed weak R=+s is self-consistent. Every belief is strictly positive,
so this is a bad cost-one fixed point for every damping below one.
Consequently no claim of convergence to the good solution from **every**
initialization can hold in this weight range.

The branch changes of this continuation occur at

\[
7/8,\qquad 15/16,\qquad 23/24,\qquad 31/32.
\]

The first is the loss of full commitment; the next two change the partly
committed pattern. Only the final value is the boundary established by the
global fixed-point argument above. At \(31/32\), subtracting a sufficiently
small positive constant from all four messages in (2) leaves their recurrence
unchanged and moves them inside the strong clipping interval. This gives a
segment of fixed points with strict bad decoding, not an isolated stable point.

Thus, for heavier weight \(w\in[1/2,1)\), **31/32 is the exact boundary:**
the fixed point is uniquely good iff \(w>31/32\). At or below it, bad fixed
points persist. This does not say that every initialized run below the
threshold fails; many initializations can still select the good solution.

## A finite high-weight pulse can then restore the original splitting

Let the baseline be .5/.5, whose clone response threshold is 2. Define the
open restore region in the same gauge:

\[
\Omega_{1/2}=\{q:\text{every directed clone Q difference is }<-2\}.
\]

For any fixed high weight \(w>31/32\), the unique target (1) lies strictly
inside this region: its strong Q values are at most \(-(4+s)<-4\), and
its weak Q values are below -4. Global convergence therefore guarantees entry
into \(\Omega_{1/2}\) after a **finite** number of updates.

Restore .5/.5 immediately after any completed high-weight Q/R update inside
this region, preserving every message. The native next Q update still uses
the old high-weight R values, so that first transition must be checked:

- Every old strong R is below -2, because its Q is below -2 and t>2.
- Every old weak R equals -s, because s<1/8 and its Q is below -2.
- An outgoing strong Q retains the other original edge's strong R and both
  weak R values; an outgoing weak Q retains both strong R values and one weak R.
  Since \(h_i\le0\), every raw outgoing Q remains below -2.

Convex damping with the old Q therefore preserves the region on the first
restored update. The subsequent baseline factor update sets every R difference
exactly to -2. Thereafter the undamped baseline Q target is the constant
\(h_i-6\), and the Q error decays as \(\lambda^k\). The baseline assignment
1010 and its cost zero remain strict and unchanged.

This proves a finite, state-certified split pulse followed by warm restoration,
for every starting state and every fixed damping below one, on this objective.
It does not prescribe one universal pulse duration. Damping is optional for
this particular theorem; the temporary representation makes the good fixed
point unique, while the restored representation has multiple attracting points.

## Checks

The rational constructor `biased_cycle_bad_fixed_point` and focused tests check
the entire continuation across all four branch thresholds, with correctly
oriented clones, original-gauge factor responses and strictly bad decoding.
The strict bounds used in the uniqueness argument are checked at exact rational
weights arbitrarily close to and above 31/32. Native-arithmetic tests start at
the baseline bad attractor, apply a .99 split, detect entry into the proved
restore region, preserve messages while restoring .5, and verify the good
assignment for damping 0, .5 and .9.

The mathematical proof establishes the all-initialization statement. The
finite numerical runs are implementation checks, not a substitute for that
proof and not a population-level benchmark result.

## Parameter-family corollary

The same proof applies to a bounded open family, not only the integer values
above. On the same four-cycle, let the original diagonal penalty be any C>0,
the off-diagonal cost be zero, and the only nonzero unary be \(u_0=(b,0)\),
where \(0<b<C/2\). Define

\[
t=Cw,\qquad s=C(1-w),\qquad
\boxed{w_*=1-\frac{b}{8C}}.
\]

For heavier clone weight \(w\in[1/2,1)\), the fixed point is uniquely good
if and only if \(w>w_*\). Above that threshold, every initialization converges
to it for every fixed Q damping below one. At or below it, a strictly
bad-decoding fixed point exists. These two assignments have original costs
zero and b, respectively.

For sufficiency, replace \(-1+8s\) by \(-b+8s\) in the oriented-cycle sum.
Above the threshold, \(s<b/8<C/16\), so the same inequalities
\(-2t+14s<0\) and \(-2t+15s<-s\) follow. All weak R must equal -s and all
strong R must equal -t, proving uniqueness. For necessity, replace the
continuation (2) by

\[
m_j=\min(t,t-b+2(j+1)s),\quad j=0,1,2,\qquad m_3=t.
\]

Closure uses \(8s\ge b\). Every strong message is at least C/2 because
\(t\ge C/2\) and \(t-b+2s=C+s-b>C/2\). Every weak Q is at least
\(-b+C+s>s\), validating the assumed weak R=+s and strict bad decoding.

The four continuation boundaries are now

\[
1-\frac{b}{2C},\quad 1-\frac{b}{4C},\quad
1-\frac{b}{6C},\quad 1-\frac{b}{8C}.
\]

For a finite state-certified pulse, replace the restore region by
\(\Omega_{1/2}=\{q:\text{every clone Q}<-C/2\}\). The high-weight good
target lies strictly inside it: strong Q is at most \(-(C+s)<-C\), weak Q
is at most \(-(2C-s)<-C\). The first restored Q update retains another
strong R below -C/2 and only additional nonpositive terms, so warm restoration
preserves the region. Baseline R then equals -C/2 and its Q target is
\(h_i-3C/2\), with subsequent error factor \(\lambda\).

Thus the construction provides a whole cost/bias/weight parameter family with
a proved pulse-and-restore mechanism. Its topology and unary structure remain
specified; it is not a theorem for arbitrary four-cycle cost tables or arbitrary
benchmark graphs. Rational checks cover several C,b choices and the exact
boundary in addition to the original fixture.

## Stronger necessity for the actual bad-baseline warm start

The fixed-point existence proof alone would not establish that the particular
baseline warm start fails below the threshold. A common positive subsolution
closes this remaining gap, including varying split and damping schedules.

For the parameter family above, in each strong orientation set outgoing Q
differences to

\[
(b/4,b/2,3b/4,b),
\]

starting with the message leaving biased vertex zero. Set every weak Q to b/8.
For every heavier weight \(1/2\le w\le w_*\), the strong threshold satisfies
\(t\ge C/2>b\), and the weak threshold satisfies \(s\ge b/8\). Thus every
factor response on this candidate equals its incoming Q. Each variable's
belief gap is exactly 5b/4. Subtracting the recipient R gives the same strong Q
profile, while every weak raw Q equals 9b/8, strictly above its candidate b/8.
Therefore \(G_w(q_{\rm sub})\ge q_{\rm sub}\) componentwise in the attractive
gauge for **every weight in this entire interval**.

The native two-stage update preserves the stronger joint bound
\(Q\ge q_{\rm sub}, R\ge r_{\rm sub}\): summing the old incoming R and
excluding its recipient gives raw Q at least the subsolution; convex damping
preserves the Q bound; each new clipped factor response is monotone and equals
\(r_{\rm sub}\) at the subsolution for all admissible weights. This argument
allows arbitrary changes of the damping coefficient within [0,1) and arbitrary
changes of w within \([1/2,w_*]\), including restoring .5.

At the bad baseline fixed point, Q is at least \(3C/2-b>C\), whereas every
subsolution Q is at most b<C/2. Baseline R=C/2 is also strictly above every
subsolution R. The initial native state therefore satisfies the bound, and
every subsequent belief gap remains at least 5b/4>0. **No such schedule can
escape the bad assignment.**

Combining this invariant lower bound with the high-weight uniqueness and warm
restoration results gives an exact initialized escape statement: from the bad
.5 baseline attractor, a pulse restricted to weights at or below w* cannot
improve the assignment, regardless of its duration or damping changes; holding
any weight above w* until the certified restore region is reached does improve
it, and permits permanent restoration of .5 without resetting messages.

For C=4 and b=1, this is precisely the boundary 31/32. Exceeding it is necessary
for this pulse family and sufficient when held long enough; merely exceeding it
for one arbitrarily short update is not claimed sufficient. The common
subsolution and the native Q/R ordering are checked in exact rational tests.
