# The splitting effect in min-sum: an exact reduction, the two ingredients, fast freezing, and the two-solution end state

Everything below is either (P) proved in this note, (Q) quoted from the paper's existing results
(`publish/sec4_effect_of_splitting.tex`, `publish/sec5b_two_solutions.tex`), or (M) measured by the
scripts in this directory (`exp0_checks.py` … `exp6_split_ratio.py`, `examples_tiny.py`; numbers in
`results/`, figures in `plots/`). Each claim is tagged.

---

## 0. Setting and notation

A pairwise problem: variables $X_1,\dots,X_n$ with domains of size $m$, unary costs $\phi_i$
(in the benchmarks: tie-break preferences in $[0,10^{-2})$), pairwise tables $C_{ij}(u,v)=C_{ji}(v,u)$
on the edges $E$ of the constraint graph $G$, and
$$\mathrm{cost}(x)=\sum_i \phi_i(x_i)+\sum_{ij\in E} C_{ij}(x_i,x_j).$$

Synchronous min-sum, iteration $t$, two phases (the PropFlow schedule):
$$Q^t_{i\to f}=\phi_i+\sum_{g\ni i,\,g\ne f} R^{t-1}_{g\to i}\qquad\text{(variable to factor: the cavity)}$$
$$R^t_{f\to j}(v)=\min_u\big[C_f(u,v)+Q^t_{i\to f}(u)\big]\qquad(f=\{i,j\})$$
$$b^t_i=\phi_i+\sum_{g\ni i}R^t_{g\to i},\qquad \hat x^t_i=\arg\min_v b^t_i(v).$$
Q-damping with factor $\lambda$: the message actually sent is $\hat Q^t=\lambda\hat Q^{t-1}+(1-\lambda)Q^t$.
Messages are defined up to additive constants; subtracting a constant from a message ("normalisation")
changes no argmin anywhere.

Splitting (SCFG): every factor $f$ is replaced by two copies $f'$, $f''$ on the same variables with
tables $pC_f$ and $(1-p)C_f$. The symmetric split has $p=\tfrac12$. Nothing else changes.

---

## 1. What the split is, exactly (P)

**Theorem 1 (exact reduction).** Take the symmetric split with zero initial messages, on any pairwise
graph, with or without Q-damping. For every directed edge $i\to j$ define the *edge messages*
$$\tilde Q_{i\to j}:=2\,Q_{i\to f'_{ij}},\qquad \tilde R_{j\to i}:=R_{f'_{ij}\to i}+R_{f''_{ij}\to i}.$$
Then at every iteration
1. $b_i=\phi_i+\sum_j \tilde R_{j\to i}$ (the split graph's belief, hence the same decoded assignment);
2. $\tilde R^t_{j\to i}(v)=\min_u\big[C_{ji}(u,v)+\tilde Q^t_{j\to i}(u)\big]$ — the **ordinary factor rule of the
   unsplit graph with the full table**;
3. $\tilde Q^t_{i\to j}=2\,b^{t-1}_i-\tilde R^{t-1}_{j\to i}=\mathrm{cav}^t_{i\to j}+b^{t-1}_i$, where
   $\mathrm{cav}_{i\to j}:=\phi_i+\sum_{k\ne j}\tilde R_{k\to i}$ is the ordinary min-sum message.
With damping, $\tilde Q$ is damped with the same $\lambda$.

*Proof.* By the paper's clone-synchronisation lemma (Q, `lem:clones`) the two copies of an edge send
identical messages, and so does $X_i$ toward them: at $t=0$ all messages are zero; if
$R_{f'\to i}=R_{f''\to i}=:r_{j\to i}$ for all edges at $t-1$, then
$Q_{i\to f'}=\phi_i+\sum_{g\ne f'}R_{g\to i}=b_i-r_{j\to i}=Q_{i\to f''}$, and then
$R_{f'\to i}(v)=\min_u[\tfrac12 C(u,v)+q_{j\to i}(u)]=R_{f''\to i}(v)$. Now
$\tilde R_{j\to i}=2r_{j\to i}$ and $\tilde Q_{i\to j}=2q_{i\to j}=2(b_i-r_{j\to i})=2b_i-\tilde R_{j\to i}$,
which is (3); and $\tilde R_{j\to i}(v)=2\min_u[\tfrac12C(u,v)+q(u)]=\min_u[C(u,v)+2q(u)]$, which is (2).
Damping is linear, so it commutes with the factor 2. $\square$

Numerically (M, `exp0_checks.py` (a)): the split engine and the unsplit engine with rule
$\tilde Q=\mathrm{cav}+b$ agree on all beliefs to $2\cdot10^{-8}$ over 400 iterations on random dense,
random sparse, coloring and bipartite instances, for $\lambda\in\{0,0.5,0.9\}$, with zero assignment
mismatches.

**Reading of Theorem 1.** The split adds no information and no nodes. It is the original min-sum with
one change in the variable rule:
$$\tilde Q_{i\to j}=\underbrace{2\,\mathrm{cav}_{i\to j}}_{\text{doubling}}+\underbrace{\tilde R_{j\to i}}_{\text{echo}} .$$
*Doubling:* the factor receives the outside evidence at twice its scale relative to its own table.
*Echo:* the recipient's own previous message is sent back to it (ordinary min-sum is non-backtracking;
this is the paper's "length-two feedback cycle through the sibling copy").

**Remark (general $p$).** Without symmetry there is no single edge message, but the structure is the
same: $Q_{i\to f'}=b_i-R_{f'\to i}=\mathrm{cav}_{i\to j}+R_{f''\to i}$ (each copy receives the sibling's
echo), and the copy with weight $p$ computes $p\,\min_u[C(u,v)+Q_{j\to f'}(u)/p]$: it weighs the incoming
evidence by $1/p$ against its table. As $p\to1$ the weak copy's table vanishes, its message tends to a
constant, the echo disappears and the weight tends to 1: the unsplit algorithm. `exp6` measures this
continuum.

---

## 2. Commitment: when a factor forwards one row (P, terminology from the paper)

**Definition.** The message $Q$ into the factor on edge $ij$ (from $X_j$) is *committed* at $u_0$ if
$u_0=\arg\min_u[C_{ji}(u,v)+Q(u)]$ for **every** receiver value $v$. Then
$R(v)=C_{ji}(u_0,v)+Q(u_0)$: the factor forwards row $u_0$ of its table plus a constant. (The paper's
"stable active minimizer".)

**Lemma 2 (commitment criterion).** Let $g(u):=Q(u)-Q(u_0)\ge0$ be the sender's margins and
$\omega(u_0,u):=\max_v[C(u_0,v)-C(u,v)]$ the largest amount by which row $u_0$ loses to row $u$ at any
column. The message is committed at $u_0$ iff $g(u)\ge\omega(u_0,u)$ for all $u\ne u_0$.
(Binary domains: $\omega=\tau_U$ of the paper's Definition `def:thresholds`.) *Proof:* rewrite the definition. $\square$

**Lemma 3 (a committed factor is locally constant).** While committed, $\partial R(v)/\partial Q(u)=[u=u_0]$
for all $v$: the message *differences* $R(v)-R(v')$ do not depend on $Q$ at all. Combined with the
affine variable rule, the composite map on message differences has zero derivative through every
committed message. *Proof:* $R(v)=C(u_0,v)+Q(u_0)$. $\square$

**What doubling does to commitment.** With $\tilde Q=2\,\mathrm{cav}+\tilde R_{\text{own}}$ the margins are
$\tilde g(u)=2g_{\mathrm{cav}}(u)+[\tilde R_{\text{own}}(u)-\tilde R_{\text{own}}(u_0)]$. Ignoring the echo term,
commitment needs $g_{\mathrm{cav}}\ge\tfrac12\omega$: **half the margin plain min-sum needs** for the
same table. This is the precise sense in which the split makes every local decision easier.

**Self-consistency of a committed configuration (the field).** If every neighbour $k$ of $i$ sends a
message committed at $x_k$, then $\tilde R_{k\to i}(u)=C_{ki}(x_k,u)+\text{const}$ and
$$b_i(u)=F_i(u;x)+\text{const},\qquad F_i(u;x):=\phi_i(u)+\sum_{k\sim i}C_{ik}(u,x_k)$$
(the paper's `lem:decode`), so the decoded value is the best response $\hat x_i=\arg\min_u F_i(u;x)$.
The decision carried by the arc $i\to j$ is $\arg\min_u[2F_i(u;x)-C_{ij}(u,x_j)]$, which is not obviously
$\hat x_i$; the paper's `lem:sibling` (Q) shows that **if** that arc is committed (with unique argmins),
its decision equals $\hat x_i$, because the commitment condition evaluated at the receiver value $v=x_j$
recombines the two half-costs into the full field. So committed arcs carry node decisions, and the node
decision is the decoded assignment.

---

## 3. The echo: a two-step feedback loop on every edge, and lock-in (Q + P)

By Theorem 1, $\tilde Q_{i\to j}$ contains $\tilde R_{j\to i}$, which was computed from $\tilde Q_{j\to i}$,
which contains $\tilde R_{i\to j}$, computed from $\tilde Q_{i\to j}$. Every edge is a cycle of length two
in message space, and one round trip applies the edge's table twice.

The paper analyses exactly this loop for one binary edge with outside evidence held fixed or moving
by at most $B$ per step (Q):
* `thm:stability`: outside the transition interval $(\tau_L,\tau_U)$ the outgoing difference
  $\Delta R$ is locked to a difference of two table entries — this is Lemma 3 in the binary case;
* `thm:absorbing`: the round-trip constants $c_L,c_U$; a regime whose constant lies inside it is absorbing;
* `lem:drift`: inside the transition interval every round trip moves $\Delta Q$ by at least $2|d|$,
  $d=C'(b,b)-C'(a,a)$;
* `thm:convergence`: finite-time absorption from any start;
* `thm:dyncvg`: with $|\delta\phi^t|\le B<\min(2|d|,-(\tau_U+\tau_L))$ the loop reaches its absorbing
  regime within $\lceil(\tau_U-\tau_L+2B)/(2|d|-B)\rceil$ round trips and never leaves it (tight, `rem:tight`);
* `prop:damped`: inside a regime, damped differences settle geometrically at rate $\lambda$;
* `prop:asym`: the asymmetric-split constants.

In words: the loop **pumps the margin of the edge's preferred row by a fixed amount every round trip**
until it crosses the commitment threshold; after that the message difference is a table constant and
is immune to bounded changes of the outside evidence. Worked instance (M, `examples_tiny.py`, part A):
one edge, table $[[0,3],[4,1]]$, unaries $(0,0.6)$ and $(0.5,0)$. Plain min-sum: $Q_{1\to2}=(0,0.6)$ from
iteration 1 on and nothing moves again (a tree is solved in one step). Split: $Q_{1\to2}$ goes
$(0,0)\to(0,2.2)\to(0,2.2)\to(0,4.2)$ — the cavity margin $0.6$ is doubled to $1.2$ and the echo of the
returned row adds $1$, then $2$, then $3$ — and the arc commits at iteration 1 (margin $2.2\ge\omega=2$).
With $\lambda=0.5$ the same margins are approached geometrically ($1.1, 1.65, 2.2, 2.75,\dots$) and the
arc commits at iteration 3.

On the full graph the echo term is, once the neighbour is committed, the edge's own row:
$\tilde R_{j\to i}(u)-\tilde R_{j\to i}(u_0)=C_{ji}(x_j,u)-C_{ji}(x_j,u_0)$, i.e. the edge's own evidence for
$u_0$ is added to the doubled cavity margin — the paper's "same cost term applied twice".

---

## 4. Why the freeze is fast, and what damping does (P + M)

Two phases.

**Phase 1 — decisions commit.** Each edge's loop drives its margin outward at a *local* rate
($\ge 2|d|$ per round trip, `lem:drift`), doubling halves the distance to the threshold (Lemma 2), and a
committed regime is absorbing against bounded perturbation (`thm:dyncvg`). None of this depends on $n$.
Measured (M, `exp1_speed.py`, 50 seeds, float semantics; a run counts as frozen at the iteration after
its last assignment change, and only if at least 100 unchanged iterations follow, so a run that was
caught between two changes at iteration 1996 is *not* frozen):

| random dense | commit $\ge95\%$ reached (median) | frozen within 2000 | median freeze | period 1 / 2 / other |
|---|---|---|---|---|
| MS | never | 0% | – | 0 / 0 / 50 |
| DMS ($\lambda=0.9$) | never | 80% | 630 | 40 / 0 / 10 |
| MS + split | iteration 20 | 0% (period 2) | – | 0 / 50 / 0 |
| DMS + split | iteration 81 | 100% | 62 | 50 / 0 / 0 |
| DMS($\lambda=0.5$) + split | iteration 28 | 94% | 19 | 47 / 0 / 3 |

On random sparse: DMS + split 98% frozen (median 71) vs DMS 38% (median 599); on graph coloring 80%
(median 130) vs 16% (median 380); on scale-free 98% (44) vs 44% (360); on meeting scheduling 86% (107)
vs 6% (447). Full tables in `results/exp1_summary.md`, figures
`plots/exp1_<bench>_aggregate.pdf` (fraction of runs frozen, mean committed fraction, mean cost vs
iteration) and `plots/exp1_<bench>_example.pdf` (seed 0).

**Phase 2 — local message stability.** Fix a reference-label gauge and an active-minimizer pattern.
The undamped Q-to-Q map is affine, with Jacobian $J$; old-Q damping changes its Jacobian to
$$J_\lambda=\lambda I+(1-\lambda)J.$$
An eigenvalue $\mu$ therefore becomes $\lambda+(1-\lambda)\mu$. A fixed point strictly inside this
active region is locally attracting when $\rho(J_\lambda)<1$. A change of minimizing rows changes
$J$, so one region's spectrum does not prove convergence of the complete trajectory. Likewise,
constant assignments alone do not establish message convergence.

If every arc retains the same committed row, Lemma 3 gives $J=0$ and
$\hat Q^{t+1}-Q^*=\lambda(\hat Q^t-Q^*)$ while that pattern is retained. This yields geometric
decay at rate $\lambda$; full commitment that alternates between rows is a different case. With
partial commitment (77–95% of arcs in the saved benchmarks), there is no general rate bound between
$\lambda$ and 1. The remaining dependencies can support oscillating or growing modes.
Measured (median over seeds of the per-iteration decay of the
largest Q change; `plots/exp1_<bench>_message_change.pdf` shows seed 0 against $0.9^t$): at $\lambda=0.9$,
0.945 in the first 50 iterations after the freeze and 0.922 in the next 100 on dense, 0.960 then 0.924 on
sparse, 0.947 on coloring; at $\lambda=0.5$, 0.58 (dense) and 0.61 (sparse). These are finite-window
measurements, not general convergence rates. Damping can affect both entry into an active region
and subsequent decay; the observed assignment-freeze time is not independent of $\lambda$.

**Comparison with plain min-sum.** Unsplit min-sum omits the direct sibling return, but messages can
still return through other graph cycles. Local response derivatives with entries in $\{0,\pm1\}$
do not imply a non-expansive network map: the variable updates sum multiple responses. Its local
stability must also be assessed through the full $J_\lambda$ and the active-region inequalities.
The table above reports slow or absent assignment freezing on these particular dense runs; plain
min-sum can converge on other inputs, including the single-edge example in §3.

**What damping does and does not do.** Commitment removes local message dependencies (Lemma 3),
while damping changes the remaining feedback modes through $J_\lambda$. It can stabilize some modes
and alter which minimizing rows are selected, but it need not stabilize every mode or trajectory.
For a prescribed alternating computed sequence, its asymptotic sent swing is multiplied
by $\kappa(\lambda)=(1-\lambda)/(1+\lambda)$ ($1/19$ at $\lambda=0.9$; Q, `lem:ema`). A specified committed
binary alternation has a persistence threshold $\lambda^*$ (Q, `thm:lambdastar`), which approaches 1
with degree in the complete-graph counterfamily
(Q, `prop:kn`). Destroying one pattern does not establish convergence to a fixed point. In the saved
experiments, a smaller $\lambda$ freezes faster (median 19 at
$\lambda=0.5$ vs 62 at $0.9$ on dense) but less reliably: 3/50 dense runs at $\lambda=0.5$ never settle
(they wander with long or no period; none of them is period 2), against 0/50 at $\lambda=0.9$, and on meeting
scheduling 14/50 runs stay in period 2 at $\lambda=0.5$. `exp5` sweeps $\lambda$ with and without the split
(20 seeds): with the split the alternation survives at $\lambda=0$ in every run, at $\lambda=0.2$ in 1/20
dense, 3/20 sparse and 16/20 coloring runs, at $0.4$ in 1/20 coloring runs, and in none at
$\lambda\ge0.5$; the freeze is fastest at $\lambda=0.2$–$0.5$ (median 17–23 on dense and sparse, 32–42 on
coloring) and slows to 229–623 at $\lambda=0.98$; once the alternation is gone the final cost is flat in
$\lambda$ (dense 100005–100307, slightly better at low $\lambda$). Without the split, freezing on dense is
best at $\lambda=0.6$–$0.7$ (95–100%, median 212–261) and worse on both sides (0% at $\lambda\le0.2$, 45%
at $0.95$); on sparse and coloring DMS freezes in at most 55% of the runs at any $\lambda$, with medians
in the hundreds. Figure: `plots/exp5_damping.pdf`.

**Ingredient dissection** (M, `exp2_ingredients.py`; four variable rules on the *unsplit* graph, 50 seeds,
2000 iterations: plain $Q=\mathrm{cav}$; doubling only $Q=2\,\mathrm{cav}$; echo only $Q=b=\mathrm{cav}+R_{\text{own}}$;
both $Q=2\,\mathrm{cav}+R_{\text{own}}$, which is the split). Frozen within 2000 / median freeze at $\lambda=0.9$:

| rule | random dense | random sparse | graph coloring |
|---|---|---|---|
| plain (DMS) | 80% / 630 | 38% / 599 | 16% / 380 |
| doubling only | 60% / 916 | 2% / 1227 | 22% / 412 |
| echo only | 100% / 208 | 100% / 224 | 100% / 162 |
| both (= split) | 100% / 62 | 98% / 71 | 80% / 130 |

Undamped, no rule freezes; echo only and the split end in period 2 (dense 50/50 for both), plain and
doubling only wander (period "other" in 50/50). Reading: **the echo is what locks** — echo alone freezes
every run; doubling alone locks nothing and even slows DMS down, although it raises the committed fraction
(0.71 vs 0.47 undamped on dense): a committed arc without the echo has nothing that keeps it committed.
Doubling on top of the echo shortens the freeze about three-fold (208 → 62, 224 → 71), as Lemma 2
predicts. On graph coloring the doubling costs a little reliability (echo only freezes 100%, the split
80%), which is the same trade-off as lowering $\lambda$: a sharper commitment criterion locks a few runs
into a state that keeps flickering.

---

## 5. The feedback loop at large scale: synchronous best response on the double cover (Q + P)

Under full commitment the paper's `cor:rule` (Q) gives, simultaneously for all $i$,
$$\hat x^{t+1}_i=\arg\min_v F_i(v;\hat x^t)$$
— **synchronous best response** (SBR). The paper's `thm:tworoutes` (Q) then shows that
$$\mathrm{cost}_2(x,y)=\sum_{ij\in E}\big[C_{ij}(x_i,y_j)+C_{ij}(y_i,x_j)\big]+\sum_i[\phi_i(x_i)+\phi_i(y_i)]$$
is non-increasing along $(\hat x^t,\hat x^{t+1})$, strictly unless $\hat x^{t+2}=\hat x^t$, so the decoded
sequence ends with period 1 or 2, and (`cor:localmin`) the limit pair is a layer-wise local minimum of
$\mathrm{cost}_2$; period 1 means $x=y$, a local minimum (1-opt / Nash point) of the original problem.

**Reformulation (P).** $\mathrm{cost}_2(x,y)$ is the cost of the assignment "$x$ on layer 1, $y$ on layer 2"
of the bipartite double cover $G\times K_2$ (every edge $ij$ of $G$ becomes the two cross-layer edges
$(i,1)(j,2)$ and $(j,1)(i,2)$). For fixed $y$, $\mathrm{cost}_2(\cdot,y)$ is separable over $i$ with
coordinate terms $\phi_i(x_i)+\sum_j C_{ij}(x_i,y_j)=F_i(x_i;y)$, so SBR is **exact alternating
minimisation of $\mathrm{cost}_2$ over the two layers**. That is the whole large-scale content of the
feedback loop: once the edges have locked, the algorithm is coordinate descent on a potential function,
and it stops at the first layer-wise minimum it meets.

Measured (M, `exp3_two_solutions.py`, undamped split, random $n=50$, 8 densities × 20 seeds): 159 of
160 runs end in period 2, and in every one of them each layer is exactly the synchronous best response
of the other (all 50 variables, at every density) — the limit object of `thm:tworoutes`. In the 8 runs
where commitment reached 99% (median at iteration 25) the decoded assignment after each step equals the
synchronous best response of the previous one in 99.97% of the steps, and
$\mathrm{cost}_2(\hat x^t,\hat x^{t+1})$ increased once in about 7800 steps. Before full commitment the
decoded dynamics is only approximately SBR; its end state is the same kind of object, a pair of mutual
best responses.

---

## 6. Why dense graphs end in two bad solutions (P + Q + M)

**(a) Which edges a layer leaves unoptimised.** Classify each edge by how many of its endpoints
alternate ($x_k\ne y_k$). Class 0: $C_{ij}(x_i,x_j)$ appears twice in $\mathrm{cost}_2$ — fully optimised.
Class 1 (say $x_j=y_j$): $C_{ij}(x_i,y_j)=C_{ij}(x_i,x_j)$ — optimised. **Class 2: $\mathrm{cost}_2$ contains only
the cross terms $C_{ij}(x_i,y_j)$ and $C_{ij}(y_i,x_j)$; the layer's own cost $C_{ij}(x_i,x_j)$ never enters
the potential, so it is whatever the cross optimisation left — a typical table entry.** With flip
fraction $f$ the class-2 share is about $f^2$. Measured (random $n=50$, undamped split, period-2 runs):
the mean cost of a class-2 edge in a layer is 151–155 at every density — the mean of a $U[100,200)$
table entry is 149.5 — against 107–132 for class-0 and 114–140 for class-1 edges; after greedy 1-opt
repair from the layer the same edges cost 111–139. The flip fraction rises from 0.45 at density 0.05 to
0.79–0.83 at density $\ge0.4$ and the class-2 share from 0.38 to 0.66–0.70. Each layer has 22–41
improving single-variable moves (of 50 variables) and greedy 1-opt from it lowers its cost by 9.8% on
average; the damped split's fixed point has no improving single move in any run.

**(b) The smallest example is an odd cycle** (M, `examples_tiny.py`, part B): a triangle of "be different"
constraints (10 if equal, 0 otherwise). Every single assignment violates one edge (cost $\ge10$), but the
pair $x=(0,0,0)$, $y=(1,1,1)$ has $\mathrm{cost}_2=0$: the double cover of a triangle is a 6-cycle, which is
bipartite, so the alternation satisfies every edge while each layer costs 30, the worst possible.
Undamped split min-sum goes to exactly this: `010 110 010 111 000 111 000 …` with 100% of the arcs
committed. Plain min-sum on the same triangle never commits and cycles through cost-10 assignments.
The displayed damped trace ($\lambda=0.9$) contains only 40 updates and ends on `101`, with pairwise
cost 10; this is a short trajectory, not convergence evidence. A native 20,000-update replay of the
same fixture in the [damping-causality investigation](../other/damping_causality/README.md)
still has assignment changes in its final tail. Damping lowers the cost here without establishing
convergence. The costs 0, 10 and 30 in this illustration omit the tiny unary contributions.

**(c) Bipartite versus not.** On a bipartite graph the two layers can be re-phased (`cor:bipartite`, Q):
$x'$ = $x$ on side $A$, $y$ on side $B$ and $y'$ the opposite satisfy
$\mathrm{cost}(x')+\mathrm{cost}(y')=\mathrm{cost}_2(x,y)\le\mathrm{cost}_2(x,x)=2\,\mathrm{cost}(x)$, so the better
re-phasing is at least as good as either layer and both are coordinate-wise local optima. The period-2
state then *contains* two good solutions. On a non-bipartite graph — a dense random graph has triangles
everywhere — no re-phasing is a single assignment, the odd cycles carry the alternation's frustration,
and the two layers are what they are: each a best response to the other, neither a local optimum of its
own cost. Measured (bipartite $25+25$, same densities): the undamped split ends in period 2 in 158 of 160
runs with the same class-2 signature (class-2 edges cost 152 against 115 for class 0; layers 12% above
their greedy 1-opt repair). Re-phasing the two layers gives assignments with **zero** improving single
moves at every density, and the better of the two costs less than the damped split's single solution
in 111 of 158 runs (mean ratio 0.998). On the random instances no re-phasing exists and the layers stay
10% above a local optimum.

**(d) Why period 2 at all, and what density changes.** SBR is a parallel update; parallel best response
overshoots because every variable answers the *old* neighbourhood while the whole neighbourhood moves
(the Goles–Olivos / Poljak–Sůra period-2 phenomenon). Undamped split runs end in period 2 in 50/50 dense
and 50/50 sparse seeds (`exp1`), so density does not decide *whether* the alternation appears; it decides
(i) how much of the graph alternates and how many edges fall in class 2 (0.38 of the edges at density
0.05, 0.66–0.70 at density $\ge0.8$; period 2 itself occurs in 159/160 runs across all eight densities), (ii) whether
re-phasing can recover solutions (only when bipartite), and (iii) how much damping is needed to kill the
alternation — and this depends on the tables more than on the density. For "be different" tables the
persistence threshold grows with the degree (`prop:kn`: $(10(n-2)-0.01)/(10(n-1)+0.01)$ on $K_n$, $0.909$
at $n=12$, so $\lambda=0.9$ fails there and $0.95$ converges), and on graph coloring (the same tables,
degree ≈ 5) `exp5` finds period 2 in 16/20 runs at $\lambda=0.2$ and 1/20 at $0.4$. For random
$U[100,200)$ tables $\lambda=0.4$ suffices at every density: 1/20 dense and 3/20 sparse runs alternate at
$\lambda=0.2$, none at $\lambda\ge0.4$, and the AAAI choice $\lambda=0.9$ ends in period 1 in 50/50 dense
seeds (`exp1`).

**(e) The damped split on dense random: one solution, but a worse one than DMS's** (M, `exp1`: final cost
99758 vs 99575). Three measured facts (M, `exp4_quality.py`, 50 seeds) and one guarantee explain it.

*Which local optimum, not whether.* The damped split's fixed point has no improving single-variable
move, no improving two-variable (edge) move and no improving three-variable path move among ≈800
sampled induced paths, on dense and sparse alike (sparse: 4.5 of 1792 paths). DMS's frozen points
(40/50 dense, 19/50 sparse) are exactly as locally optimal: 0 / 0 / 0 improving moves in every frozen run,
while its unfrozen runs still have 10.7 single, 128 edge and 120 path moves on dense. So nothing up to
three variables separates the two frozen points. Paired on the seeds where DMS froze, the split's point is
worse by $538\pm49$ on dense (0.5%) and by $144\pm19$ on sparse (1.0%); on the seeds where DMS did not
freeze it is better by $1236\pm651$ and $768\pm126$. DMS wins the dense mean because it freezes in 80% of
the dense runs and only 38% of the sparse ones.

*The guarantee gap.* A fixed point of max-product/min-sum is optimal within the single-loops-and-trees
(SLT) neighbourhood of its factor graph (Weiss–Freeman 2001): assignments that differ on a set whose
induced factor subgraph has at most one cycle per component. On the split graph every edge $ij$ is a
4-cycle $i\!-\!f'\!-\!j\!-\!f''\!-\!i$, so a connected set of $k$ variables induces at least $k-1$ independent
cycles: **the split graph's SLT neighbourhood is single-variable moves plus single-edge (two-variable)
moves, and nothing larger.** On the unsplit graph it contains every tree-shaped move and every move with
one cycle. The measured 0.5% gap therefore sits in moves of four or more variables, which we did not
enumerate.

*Lock-in time.* Splitting DMS's state at iteration $K$ (transfer mode) locks onto a committed point near
DMS's *message* state within about 50 iterations, and the later the split, the better the point: on dense
99543 ($K=50$) → 99453 → 99433 → 99394 → 99336 → 99296 ($K=1500$), against 99758 for the split from the
start, 99575 for DMS and 99283 for DMS's final point after greedy 1-opt. At every $K$ the split-at-$K$
endpoint beats greedy 1-opt repair of DMS's *decoded* state at $K$ (99704 at $K=50$ → 99337 at $K=1500$):
the split reads DMS's messages, not its argmins, and DMS's messages keep improving for hundreds of
iterations after its decoded assignment looks settled. On sparse every $K$ gives 14549–14587, below DMS
(15055), DMS + greedy (14640) and the split from the start (14633); lock-in time hardly matters there.
Figure: `plots/exp4_split_at_k.pdf`.

On sparse, coloring, scale-free and meeting instances DMS freezes in far fewer runs and its decoded
snapshot is poor, so DMS + split wins outright (`results/exp1_summary.md`).

**(f) Asymmetric split** (M, `exp6_split_ratio.py`, DMS 0.9, 20 seeds, $p$ from 0.5 to 0.99 and the
unsplit graph). The freeze time stays at 53–67 (dense) / 64–74 (sparse) for $p\le0.8$ and rises to 122,
176, 420 (dense) and 90, 109, 262 (sparse) at $p=0.9, 0.95, 0.99$, reaching 468 (dense, 90% frozen) and
321 (sparse, only 40% frozen) unsplit; the committed fraction falls with $p$ (0.95 → 0.83 dense, 0.76 →
0.15 sparse). The final cost improves with $p$ up to 0.95 — dense 100300 ($p=0.5$) → 99798 ($p=0.95$) →
100004 unsplit; sparse 14518 → 14388 → 14889 — so a weak second copy ($p=0.9$–$0.95$) freezes 95–100% of
the runs within 90–180 iterations at a better point than either the symmetric split or the unsplit
algorithm. This is the continuum of the §1 remark: the
echo carries the weak copy's weight $1-p$ and the doubling factor $1/p\to1$; less echo means a slower,
later lock-in, which by (e) is a better one.

---

## 7. Implementation facts that matter for reproducing the AAAI numbers (M, `exp0_checks.py`)

1. **Integer truncation.** PropFlow's `compute_R` casts every incoming Q message to the cost table's dtype
   (`src/propflow/bp/computators.py:200`). The random_dense and random_sparse tables are `int64`, so in the
   recorded runs every unsplit factor saw $\mathrm{trunc}(Q)$; split copies ($\tfrac12 C$) are float tables,
   so split runs — and split-at-$K$ runs after the split — did not truncate. graph_coloring, scale_free and
   meeting_scheduling have float tables. Effect on DMS over 50 seeds (final cost, 2000 iterations):
   truncated $99921\pm2663$ vs float $99575\pm2709$ on dense (paired difference $+346$, s.e. $281$);
   $14921\pm1489$ vs $15055\pm1567$ on sparse ($-134$, s.e. $119$); best-so-far $99239$ vs $99172$ and
   $14506$ vs $14505$. Not a measurable bias. The mechanism experiments use float semantics.
2. **Exact reproduction.** With truncation and PropFlow's normalisation schedule (subtract the per-message
   minimum every `graph_diameter` iterations) the vectorised engine reproduces the recorded cost curves
   iteration for iteration over all 2000 iterations for MS, MS + split, DMS + split and DMS split-at-$K$
   on random dense/sparse and graph coloring (`results/exp0_checks.txt`). Damped **and** truncated runs
   (DMS before any split) cannot be reproduced bit for bit: the truncated value of
   $0.9\,\hat Q_{\text{old}}+0.1\,Q$ depends on the last bit of a float sum, and two summation orders
   diverge after 3–30 iterations. Those runs are compared statistically (item 1).
3. **Message level.** Between normalisations the level of undamped messages grows like
   $(\deg-1)^t$; an undamped PropFlow run with no normalisation reaches $10^{17}$ in 23 iterations on a
   degree-5 instance and its decoded assignment is float noise. Every PropFlow cross-check therefore
   normalises each step (a per-message constant, no effect in exact arithmetic).
4. **Provenance.** In `data_cuda/meeting_scheduling_raw_costs.csv` the DMS and split rows have integer
   costs (no tie-break unaries) while the MS rows carry the unaries; only the MS rows match the current
   `problems.py` builder. The other four benchmarks are consistent.

---

## 8. Summary: what is proved, what is measured

| Claim | Status |
|---|---|
| Symmetric split $\equiv$ unsplit min-sum with $\tilde Q=2\,\mathrm{cav}+\tilde R_{\text{own}}$ (damping included) | P (Theorem 1), M (exp0 a) |
| Commitment criterion; doubling halves the required margin | P (Lemma 2) |
| Fixed committed rows give $J=0$ and decay at rate $\lambda$ while retained; otherwise assess $J_\lambda=\lambda I+(1-\lambda)J$ and branch inequalities | P (Lemma 3, §4); exp1 decay estimates are finite-window measurements |
| The echo pumps each edge's margin by $\ge2|d|$ per round trip and locks it, robust to bounded perturbation | Q (`lem:drift`, `thm:dyncvg`; one binary edge) |
| The saved dense runs freeze faster with splitting and damping; timing and reliability depend on $\lambda$ and the instance | M (exp1, exp5); not a general convergence theorem |
| Under full commitment the decoded dynamics is synchronous best response = alternating minimisation of $\mathrm{cost}_2$ on the double cover; period 1 or 2 | Q (`cor:rule`, `thm:tworoutes`), P (reformulation), M (exp3) |
| Period-2 layers are unoptimised exactly on class-2 edges; bipartite graphs re-phase, non-bipartite do not | P (§6a), Q (`cor:bipartite`), M (exp3) |
| Split fixed points are guaranteed optimal only for 1-/2-variable moves (SLT of the split graph), DMS's for every tree-shaped move on $G$; measured: neither has improving 1-, 2- or 3-path moves, and where DMS freezes its point is better by 0.5% | P (§6e) + Weiss–Freeman, M (exp4) |
| Split-at-$K$ improves with $K$ (99543 → 99296 on dense) and beats greedy repair of DMS's decoded state at every $K$: the split locks the nearest committed point to DMS's message state | M (exp4) |
| Damping threshold for the alternation grows with degree for "be different" tables; $\lambda\ge0.4$ suffices for random tables at every density; with the split, $\lambda=0.2$–$0.5$ freezes fastest | Q (`thm:lambdastar`, `prop:kn`), M (exp5) |
| Asymmetric split interpolates to the unsplit algorithm as $p\to1$; $p=0.9$–$0.95$ is faster than DMS and better than both | P (§1 remark), M (exp6) |
| AAAI unsplit runs on integer tables truncated Q; no measurable bias | M (exp0 d) |

Open, as before: a convergence theorem for the full graph with domains $>2$ (the local results are for one
binary edge with bounded outside perturbation; commitment on benchmarks is partial), and the damped
analogue of `thm:convergence`.
