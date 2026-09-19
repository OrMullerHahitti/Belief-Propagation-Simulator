# Damping changes which stable region split Min-sum reaches

The cause is a feedback loop introduced by factor duplication, together with
synchronous updates and changes of the active minimizing rows. On the exact
three-variable example below, the undamped algorithm enters an attracting
two-cycle. Damping changes the trajectory into a different region where every
factor response is fixed. It can then be removed. This is an exact, native-tested
example of basin selection, not a universal convergence theorem.

The complete derivation, rational certificates, and qualifications are in
[theory.md](theory.md). All traces and source copies are under
`results/damping_causality_20260915/`. [PROTOCOL.md](PROTOCOL.md) records the
requirements; [LITERATURE.md](LITERATURE.md) distinguishes this evidence from
prior work. No neural network, DABP, online learner, search branch, or objective
change is involved.

## The smallest main example

Use three binary variables on a path `x1 -- x2 -- x3`. Both original edge tables
are `[[16,0],[0,16]]`: agreeing costs 16, disagreeing costs zero. Unary costs are
`x1:[12,0]`, `x2:[13,0]`, `x3:[4,0]`. Enumerating all eight assignments gives the
unique optimum `101`, cost **13**; the next best is `010`, cost 16.

Split each edge into two copies of half its table, leaving the objective exactly
unchanged. All methods use the same native zero initialization, Q-then-R phases,
and cycle normalization. Lambda is the coefficient on the **old Q message**.

| Method, same original objective | Observed behavior | Original cost |
|---|---|---:|
| Unsplit MS, lambda 0 | fixed `101` | 13 |
| Equal split MS, lambda 0 | `111` / `000` forever, exact proof | 32 / 61 |
| Equal split DMS, lambda .01 | attracting two-cycle | 32 / 61 |
| Equal split DMS, lambda .016 | a different attracting two-cycle | 32 / 61 |
| Equal split DMS, lambda .02 | fixed `101`, exact convergence proof | 13 |
| Equal split DMS, lambda .5 | fixed `101`, exact convergence proof | 13 |
| Equal split DMS, lambda .9 | fixed `101`, exact convergence proof | 13 |

![Individual native belief and cost trajectories](../../../results/damping_causality_20260915/figures/path_damping.png)

The two panels show the middle variable's belief difference and original cost.
These are individual per-update values without smoothing, early stopping, or
best-so-far selection. The full traces contain 2,000 updates. The dominant
unaries make the arithmetic transparent; a second path with unary preferences
between .001 and .004 reproduces the same qualitative result.

## What creates the oscillation

An outgoing variable message excludes the factor receiving it. After duplication
it still receives the other copy's message. Information can therefore return
immediately through the sibling copy. Every other original edge contributes two
messages as well. Multiplying costs by one half without adding this connectivity
does not create the same dependencies.

In full-original-table message units, equal splitting gives the cavity field

`2 * (unary + other-edge messages) + sibling return`.

We separated the external gain `g` and sibling gain `s`, preserving the same
objective and initialization. Only `(1,0)` and `(2,1)` are ordinary unsplit and
equal-split Min-sum; the other operators are diagnostic interventions.

| External gain g | Sibling gain s | Outcome on the path |
|---:|---:|---|
| 1 | 0 | fixed `101`, cost 13 |
| 2 | 0 | fixed `101`, cost 13 |
| 1 | 1 | `000` / `111` cycle |
| 2 | 1 | `000` / `111` cycle |
| 2 | -1 | fixed `100`, cost 33 |

Removing sibling return from the **same state after update 32** stops the
oscillation. Removing it for just one update and restoring it also reaches
`101`. Reversing a binary sibling's label preference preserves the magnitude of
its message difference at the intervention, but changes the outcome. Its
subsequent evolving amplitudes are not matched. Thus the role of feedback is
more specific than a change in numerical scale, but we do not infer a general
necessity theorem from these five operators.

For this selected path, sibling return is sufficient to create the oscillation;
external amplification alone is not. The original path is a tree, so this is
also a clean illustration that splitting can introduce a convergence problem.

## What damping actually changes

The per-clone factor response is exactly `r(q) = -clip(q,-8,8)`. The unclipped
part reverses incoming preferences. Outside this interval, the same sending
label minimizes both receiving-label calculations, and the returned difference
is independent of the incoming message magnitude.

With no damping, the four synchronized directed-Q differences reach
`A=(-4,11,11,4)` and `B=(-20,-17,-9,-12)`. Exact substitution maps A to B and B to
A. Both decoded assignments and inner minimizers have strict margins. The orbit
is itself attracting; it is not floating-point noise or an unstable fixed point.

With lambda .5, exact rational arithmetic proves that after 13 completed native
updates the state lies in `a<-8,b>8,c>8,d<-8`. Throughout this convex region the
undamped Q map is the constant `q*=(-20,11,11,-12)`, so subsequent Q errors are
multiplied by lambda. All decoded beliefs are fixed at `(-28,19,-20)`, selecting
`101`. Unary Q differences also converge; raw additive message offsets are not
the convergence criterion. See [theory.md](theory.md) for the finite entry point
and invariance proof.

Damping mixes successive messages while keeping the synchronous schedule. It
changes which minimizing-row boundaries are crossed and which region is reached.
The good fixed region already exists and is stable at lambda zero. Consequently,
after entering it, removing damping reaches its Q fixed point in one update.

The native intervention starts with 32 identical undamped updates, introduces
lambda .5, and removes it after update 256. Q/R, beliefs, assignments and costs
match the untreated run exactly through the intervention point. The treated run
reaches cost 13 and stays there after damping is removed, with all-message
undamped-map defect below 1e-10.

![Damping can be removed after entry into the fixed region](../../../results/damping_causality_20260915/figures/same_state_intervention.png)

## Why merely adding damping is not enough

The initial two-cycle has an exact continuation for
`0 <= lambda < (-33 + 2*sqrt(277))/19`, approximately **.015086**. At that boundary,
one active-minimizer condition fails. This is a threshold for that specific
pattern, not a convergence threshold. At **.016**, a different exact two-cycle
takes over on the same path. The .02 run reaches the fixed point, while the .5
case has a finite-entry proof. Exact rational checks also establish entry for
.02 and .9, after 51 and 54 completed updates, respectively. Their infinite
tails then follow from the same invariant-region argument; these checks do not
establish a complete interval of successful damping values.

The often-used formula `J_lambda=lambda*I+(1-lambda)*J` applies within a fixed
active region. It explains some attenuation of alternating modes, but it cannot
replace the nonlinear branch analysis here. In this example the good fixed
point is already locally stable without damping. Also, suppressing one symmetric
mode on a complete graph does not stabilize all of its nonuniform modes.

## Countercontrols and limits

- **Damping is not always necessary.** A single equally split edge with table
  `[[0,3],[4,1]]` and unaries `[[0,.6],[.5,0]]` converges without damping to `00`,
  cost .5, with strict belief margins and zero message defect.
- **High damping is not sufficient.** The old anti-equality triangle with
  diagonal costs 10 and seed-3 tiny unary preferences still oscillates at
  lambda .9 after **20,000 updates**. It changes assignments 36 times within
  its last 300 updates; the maximum all-message step difference there is 1.1367.
  The former 40-update illustration was therefore not convergence evidence;
  its explanatory comments and the related formal notes have been corrected.
- **Convergent messages need not imply a stable decoder.** An exact symmetric
  triangle recurrence converges to zero belief gaps while those gaps alternate
  sign forever. This separate rational counterexample is proved in `theory.md`.
- **Convergence and cost quality remain distinct.** The sign-reversed feedback
  control converges at cost 33, worse than the exact optimum 13. No universal
  optimum or cost-improvement claim follows from message stability.
- **The examples establish mechanisms, not prevalence.** Four deliberately
  selected tiny fixtures do not establish how often each mechanism governs the
  sparse, dense, coloring, scheduling, or scale-free benchmark families.

![A damped triangle still moves after 20,000 updates](../../../results/damping_causality_20260915/figures/damping_counterexample.png)

## Paragraph for the experiment section

> To isolate the role of damping after splitting, we construct a three-variable
> binary path for which ordinary Min-sum converges to the unique optimum.
> Symmetric factor duplication preserves the objective but introduces sibling
> feedback and produces an attracting period-two orbit, alternating between
> assignments of costs 32 and 61. Exact active-minimizer calculations identify
> this orbit and prove that damping with old-message coefficient .5 enters a
> different invariant region whose fixed assignment has optimal cost 13.
> Introducing damping into an already oscillating native run produces the same
> outcome; removing it after stabilization preserves the solution. External-gain
> and sibling-return interventions identify the return feedback as the source of
> oscillation in this example. These results show that damping can change basin
> selection in split Min-sum, beyond reducing visible oscillation amplitude.
> Insufficient damping can preserve the original cycle or replace it with a
> different cycle, so destroying one oscillatory pattern does not establish
> general convergence.

## Reproduction and verification

From the repository's existing uv environment:

```sh
uv run --no-sync python -m experiments.aamas.damping_causality.code.native_experiments --out results/damping_causality_replay
uv run --no-sync python -m experiments.aamas.damping_causality.code.run_feedback --output results/damping_feedback_replay
uv run --no-sync python -m experiments.aamas.damping_causality.code.plot_evidence
uv run --no-sync python -m pytest tests/test_damping_causality_theory.py tests/test_damping_causality_native.py tests/test_damping_causality_feedback.py -q
```

The native evidence contains 18 runs and 72,000 completed updates, with every
Q/R difference, belief, assignment, original cost and undamped-map defect saved.
A separate replay reproduced every native NPZ array and CSV byte exactly.
Source/configuration copies and explicit capture chronology are recorded in
`native/source_verification.json`. The feedback study contains 26 prescribed
runs and direct equal-clone arithmetic comparisons.

The final independent audit and verification outcomes are recorded under
`results/damping_causality_20260915/INDEPENDENT_AUDIT.md` and `VALIDATION.md`.
No paper source, package API, version, commit, push, or deployment was changed.
