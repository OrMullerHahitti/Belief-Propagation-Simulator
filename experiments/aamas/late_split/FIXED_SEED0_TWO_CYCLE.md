# Why the fixed-time seed-0 run repeats two assignments

The two assignments form a self-consistent feedback cycle on this instance.
The unchanged native replay selects B after A and A after B. Independently,
the full message equations on the same saved cost tables admit an exact
two-cycle of relative messages that selects those same assignments. All 20
values were checked for all 50 agents; the winners have strict positive
margins. This goes beyond observing two values or assuming that all factor
messages reduce to hard neighbor choices.

The scope is the domain-20, random-dense seed-0 fixed-time run: 1,000 DMS
updates with old-Q damping 0.9, then equal factor splitting with equal R
transfer and 1,000 undamped updates. Iterations below count completed native
updates. Artifacts are in
[`fixed_seed0_cycle/`](../runs/late_split_domain20_20260921/fixed_seed0_cycle/).

## 1. Verify the saved cycle

Every saved input/checkpoint/trace hash passed. Recomputing all saved costs
on the original ordered tables gave zero error. Restoring the archived
checkpoint and replaying the original continuation reproduced every one of
the 1,000 assignments and costs exactly.

The earliest period-two suffix begins at completed update **1013** and
continues through **2000**, covering **988** updates. Let:

| Assignment | Completed-update parity | Original objective cost |
|---|---|---:|
| A | Even | 103496.246257804 |
| B | Odd | 103827.26190184215 |

The assignments differ at 37 agents; the other 13 retain their values.
Thus the joint assignment has period two, rather than a fixed assignment.

Evidence: `step1_saved_cycle.json`, `replay.log`. The checkpoint loader is
the original archived `core.py`; all native PropFlow source hashes match
the original run manifest. The current checkout's unrelated checkpoint
capture extension was not used to redefine the replay.

## 2. Inspect what the factors compute

The graph has 726 original pairwise factors and 50 unary factors. Both
clones' Q and R messages remained exactly equal at all 1,000 replay updates
(1,502,000 paired checks). Therefore one representative of each clone pair
suffices for the mathematical analysis.

We recorded every internal minimizer for all 20 receiver values, for both
directions of every pairwise factor, during updates 1001–1040 and 1901–2000.
The tail contains 1,452 representative directed messages at each update.

| Transition producing the next assignment | Messages with one strict neighbor minimizer for all 20 receiver values |
|---|---:|
| A → B | 1369 / 1452 = 94.28% |
| B → A | 1316 / 1452 = 90.63% |

For these messages, the relative outgoing score profile is one fixed
cost-table slice. Changing the selected incoming score's magnitude only
adds a common offset until a minimization boundary is crossed.

The remaining 5.72% or 9.37% can minimize using different neighbor values
for different candidate entries. They still depend on message magnitudes.
It would be incorrect to replace all native factor calculations with hard
neighbor choices.

Nevertheless, **at the receiver value that actually wins, every factor
minimizes at the neighbor's preceding selected value**: 145,200 of 145,200
checks in the final 100 updates. Every fully common minimizer also equals
that preceding neighbor value. Computing each agent's ordinary best
response against the preceding assignment agrees with the native winner
in all 5,000 agent-update checks. In particular, BR(A)=B and BR(B)=A.

This agreement concerns the winners. Full belief vectors and best-response
cost vectors need not be equal. For example, some rival entries receive
lower estimates through a different hypothetical neighbor choice.

The relative native messages are not exactly period two: their maximum
lag-two difference in the tail is 0.00818001. There are also 23 changes in
internal minimizer entries at lag two, all on four directed arcs. Those
changes do not change the winning assignments. The smallest observed
belief winning margin is 6.009765625.

Evidence: `factor_trace.npz`, `arcs.json`, `unary_factors.json`,
`step2_factor_observation.json`, `step3_closure.json`.

## 3. Establish a full message mechanism that sustains A → B → A

Best-response agreement alone would not prove a closed BP cycle: the full
vectors, including the nonwinning entries, feed the next update. We
therefore solved and checked the full, undamped message equations.

Every saved IEEE cost-table entry is an exact binary rational. Multiplying
by 2^73 makes the original unary tables and half pairwise tables integers.
The algebraic calculation uses arbitrary-precision integers, so no
tolerance or floating-point convergence test is involved in its closure.
No costs, assignments, graph edges or update parameters are changed.

We found two relative message states, M_A and M_B, with:

\[
F(M_A)=M_B,\qquad F(M_B)=M_A,
\]

where F is the original Min-sum Q-then-R update, with equal clones and no
damping. Adding or subtracting a common constant from every entry of one
message changes no selected value in exact arithmetic. The states are
therefore represented by their differences relative to the phase's
selected receiver value. This is a cycle of relative messages, not a
claim that unnormalized additive offsets repeat.

| Exact message transition | Selected assignment | Smallest belief margin over any rival value, across all agents |
|---|---|---:|
| M_A → M_B | B | 14.018349310307961 |
| M_B → M_A | A | 6.009793880184027 |

The positive margins show that all other candidates lose at their
respective phase. The distinct assignments A and B show that this closed
orbit's minimal period is two, not one. There is no third state in this
exact orbit: after its second update the relative message state itself
has returned.

The serialized orbit was then checked independently through the native
`MinSumComputator.compute_Q` and `compute_R` functions using integer arrays.
Both transitions, all original factors including unaries, both clones,
all 20 values, and all 50 decoded winners passed exactly. An additional
independent mathematical review reconstructed the saved tables and axes
and verified the enclosure argument below.

### Why some remaining message variation is compatible with the cycle

We also constructed two componentwise enclosures [L_A,U_A] and [L_B,U_B],
with the exact orbit as their lower endpoints. They preserve the same
decoded A/B assignments throughout. The proof uses the actual message
formula, not a two-value restriction.

Write r_(f→i)(v)=R_(f→i)(v)−R_(f→i)(a_i) for a state in phase a. With
synchronized equal clones, and unary preference phi_i, the relative Q
sent from i to one clone of f is

\[
q_{i\to f}(v)=\phi_i(v)-\phi_i(a_i)
+2\sum_{g\ni i}r_{g\to i}(v)-r_{f\to i}(v).
\]

The sum here ranges over representative pairwise factors; unary terms are
already in phi_i. The reverse factor's coefficient is therefore 1 and
every other coefficient is 2. All coefficients are nonnegative.

At the lower endpoint, for every outgoing direction and next-phase winner
b_i, the unique minimizing neighbor is a_j, with positive margin. This
remains true throughout the enclosure because q(a_j)=0 and increasing
other input coordinates cannot make another neighbor cheaper. Consequently
the normalized outgoing update throughout the enclosure is

\[
r'_{f\to i}(v)=\min_w\{\tfrac12C_f(v,w)+q_{j\to f}(w)\}
-\tfrac12C_f(b_i,a_j),
\]

a coordinatewise monotone function of the incoming relative messages.
The exact checks establish

\[
F(L_A)=L_B,\quad F(U_A)\le U_B,
\qquad F(L_B)=L_A,\quad F(U_B)\le U_A.
\]

Monotonicity sandwiches the image of each enclosure inside the other.
The positive belief margins at the lower endpoints ensure that every
state in them decodes to the same A/B pair. Moreover, three alternating
updates send the upper endpoints to the lower orbit exactly, so squeezing
shows the same finite capture for every exact state inside these
enclosures. This is a sufficient mechanism sustaining the two-cycle.

These enclosures are thin: only 1,149 of 58,080 pairwise-message coordinates
have positive width. This is not a proof of attraction from an arbitrary
open neighborhood or arbitrary perturbations.

### Relation to the recorded native run

Throughout the final 100 updates, after aligning each message to its
phase's selected value, the native messages differ from the exact orbit
by at most 0.008135 per entry. Comparing entire relative belief vectors
gives a maximum difference of 0.040805, while the exact smallest winning
margin is 6.009794. Every native winner matches the corresponding exact
winner. Thus the small variations observed in the replay are far too
small, in the measured belief vectors, to overturn the phase's selection.

Native states do slightly cross the exact enclosure boundaries because
of their numerical differences. Accordingly, the certificate proves a
matching exact-arithmetic cycle and its enclosure conditions; the
unchanged native replay proves its observed assignment repetition through
update 2000. It does not prove perpetual repetition under IEEE arithmetic.

![Recorded x1 belief preferences across A to B to A](../runs/late_split_domain20_20260921/fixed_seed0_cycle/two_cycle_explanation.png)

The figure shows actual recorded beliefs for x1. “Best of other 18” is the
minimum over those candidates, so none of them is hidden below the displayed
winner. These scores are not global objective costs.

## Reproduction and evidence

From the repository root, in this order:

```sh
PYTHONPATH=. uv run --no-sync python experiments/aamas/runs/late_split_domain20_20260921/fixed_seed0_cycle/replay.py
uv run --no-sync python experiments/aamas/runs/late_split_domain20_20260921/fixed_seed0_cycle/analyze.py
uv run --no-sync python experiments/aamas/runs/late_split_domain20_20260921/fixed_seed0_cycle/verify_certificate.py
uv run --no-sync python experiments/aamas/runs/late_split_domain20_20260921/fixed_seed0_cycle/plot_explanation.py
```

- Saved-run identity and earliest suffix: `step1_saved_cycle.json`.
- Exact native replay and passive factor observations:
  `step2_factor_observation.json`, `factor_trace.npz`, `replay.log`.
- All-agent response checks and exact closure: `step3_closure.json`,
  `response_fields.npz`, `exact_cycle.json.gz`.
- Independent native-Q/R certificate verification:
  `certificate_verification.json` and its log.
- Figure inputs, belief-error comparison and hashes:
  `presentation_data.json`, `plot_explanation.py`, and PNG/PDF outputs.

Only the fixed-time seed-0 case is analyzed here. The work follows the
requested three steps and does not extend to a comparison with the
irregular run.
