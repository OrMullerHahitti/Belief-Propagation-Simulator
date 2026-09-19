# A small laboratory for learning splitting and damping

> **September 15 correction:** Extending the same saved policies and inputs
> from 128 to 2,000 updates removes the apparent advantage over fixed .5/.5
> splitting with .9 damping. Mean final costs were 10.9623 for that baseline
> versus 11.2567 for the settled controller on bowties, and 12.0577 versus
> 12.8244 on K4. The controller also had fewer stable final tails. These runs
> do not demonstrate an improvement over the usual paper baseline. See
> `results/aaai_derived_control_20260915/experiment_audit.md` for the matched
> extension, family breakdown, and limitations. The original 128-step results
> below remain a record of the earlier experiment.

**Recommendation:** use five variables arranged as two triangles sharing one
variable, and a shared controller with **145 trainable parameters**. The useful
lead from the completed experiments is to separate early split interventions
from a later settling phase. We have **not demonstrated that online learning
beats a strong fixed baseline**.

The experiment code, protocol, sources, and commands are in [README.md](README.md).
The final evidence is in
[the verified run directory](../../../results/adaptive_split_control_20260914_final/).
Earlier directories are retained intermediate runs; use the `final` directory
for the numbers below.

## The graph and controller

```text
    x1 ─── x2
      \   /
       x0
      /   \
    x3 ─── x4
```

There are six original pairwise cost functions. Splitting each into two gives
twelve factors, with five variable nodes. Three labels per variable give only
243 assignments, so we can measure the exact optimality gap. Binary instances
have 32 assignments. Exact solutions are used for evaluation, never as online
observations or training rewards.

One split edge alone isolates sibling feedback. A triangle isolates one loop.
The bow-tie introduces two interacting loops and a shared variable while
remaining easy to inspect. A complete four-variable graph, K4, tests whether
the controller transfers to a different arrangement of the same six edges.
K4 is smaller in variable count but less clean for attributing interactions
between the loops.

For original table C_e, use the actual decomposition

\[
C_{e,0}^{t}=w_e^t C_e,\qquad C_{e,1}^{t}=(1-w_e^t)C_e.
\]

Their sum remains C_e at every step. The two factor-variable connections of a
clone share its cost coefficient. Arbitrary independent multipliers on
received messages would be a different algorithm, not this decomposition.
When a weight changes, existing messages retain their values and clone
identities. We do not reset or rescale them at the intervention.

The implemented network is `16 inputs → 8 tanh units → 1 action score`, shared
over candidate edge/split/damping actions: 128 + 8 + 8 + 1 = **145 parameters**.
It chooses an edge/action by its score, a form of hard attention. It does not
replace the BP aggregation rule. Observations include one- and two-step R
residuals, clone-message magnitude, belief margin, disagreement with original
best responses, and the current/candidate parameters.

The current action set changes at most one edge to .05, .5, or .95 and chooses
global Q damping from 0, .5, or .9, every eight iterations. Damping always means
the weight on the **old** Q message:

\[
Q^{t+1}=\lambda_t Q^t+(1-\lambda_t)Q_{\mathrm{computed}}^{t+1}.
\]

This is deliberately a small action class. It does not search all real-valued
weights or independent damping rates for every directed message.

## What the splitting research tells us to measure

The project's existing equal-clone reduction contains a sibling return term.
Without unary factors or damping, the combined directed message obeys

\[
M_{j\to i}^{t+1}(x)=\min_y\{C_{ij}(x,y)+2B_j^t(y)-M_{i\to j}^t(y)\}.
\]

The new tests reproduce this identity. The earlier frozen-instance sibling
feedback result motivates the experiments, but does not prove the same
behavior across all small or dense graphs.

Two deductions are particularly useful for designing the learner.

**1. Split changes can be invisible to combined beliefs.** Start with equal
clones. Compare two runs with the same initial state and damping sequence,
changing only cost-split weights. Suppose that through a finite interval both
clones in the perturbed run retain the reference run's active minimizing
sender label, for every outgoing message and receiving label. Then the
combined beliefs remain equal throughout that interval, modulo message gauges.

To see why, for a common selected label y*, the sum of the two factor outputs
is, before subtracting a label-independent constant,

\[
wC(x,y^*)+Q_0(y^*)+(1-w)C(x,y^*)+Q_1(y^*)
=C(x,y^*)+Q_0(y^*)+Q_1(y^*).
\]

The corresponding summed Q update uses `2B − (R_0+R_1)` and the same damping
coefficient, so induction preserves both message sums. Individual clone
messages can move while their combined contribution stays unchanged.

For a finite strict-active trajectory, sufficiently small weight changes
therefore give zero local sensitivity of the combined beliefs to those
changes. This statement assumes common clone damping and unchanged active
choices; it is not a blanket claim about asymmetric states, arbitrary
attention weights, or DABP's entire model.

We checked 432 controlled perturbations: 48 graphs × three intervention times
× three perturbation sizes. With a 1e-6 split change, none of the 144 cases
changed active choices or combined beliefs. With a .01 change, some beliefs
changed but none of those 144 runs changed final cost. With a .45 change,
final cost changed in 50/144 cases; a change was not necessarily beneficial.
The largest belief discrepancy before any active-choice change was about
6.5e-12. These are numerical checks of the stated algebra and tested cases,
not a prevalence theorem.

**2. Splitting and damping act differently inside an active region.** For a
fixed cost decomposition, let the undamped Q-state map have Jacobian J_0.
Then

\[
J_\lambda=\lambda I+(1-\lambda)J_0.
\]

Cost weights affect affine constants and which minimizing choices are active;
while those choices stay fixed, they do not change this Jacobian. Damping does.
The analytic Jacobian was checked against finite differences. For a real
eigenvalue μ, damping maps it to λ+(1−λ)μ. A mode μ=−1 is canceled at λ=.5;
a real mode μ>1 cannot be made contractive with 0≤λ<1 by this mechanism.
These are local statements. A switching trajectory crosses regions, and its
stability cannot be inferred from one Jacobian or from assignment repetition.

## The two learning tracks we actually ran

**Across instances:** 48 training graphs, 24 validation graphs, and 11,232
eight-step counterfactual action outcomes. Train three network initializations
and select one using validation data. Also select the fixed baseline from
20 split/damping configurations on the same validation instances. No test
optimum or test action outcome enters fitting.

**Within one problem:** execute a chosen action for eight steps, observe the
actual original-cost change and stability penalty, and update the scorer using
only executed actions. Keep a separate best assignment, but evaluate terminal
cost independently. Each test starts a fresh online learner; learning across
changing problems or changing cost tables is not implemented in this study.

The first comparison used 128 held-out graphs. Continuous learned control
performed poorly. We then selected an early intervention/settling boundary
on validation graphs and evaluated on 256 additional graphs: 128 bow-ties
and 128 K4 graphs. The selection allowed zero online interventions, so a
controller was not required to win by construction.

Validation selected two actions, at iterations 0 and 8, followed by **fixed
split weights and λ=.7 from iteration 16 onward**. It separately selected a
non-learning equal-split schedule with four undamped iterations followed by
λ=.5. All methods run 128 BP iterations. The online learner therefore gets
only two real feedback blocks in the settled variant; its negative result
must not be generalized to all possible online learners.

## Results and what improved

For the 128 confirmation bow-ties, the gap is
`100 × (final cost − optimum) / sum of original table ranges`.
It is not a percentage relative to the optimum. Stability means the last
16 iterations, with both Q and R checked for message stability.

| Method | Mean final gap | Stable assignments | Stable messages |
|---|---:|---:|---:|
| Tuned fixed .5/.5 split, λ=.5 | 5.114 | 80/128 | 77/128 |
| Continuous online controller | 7.874 | 66/128 | 2/128 |
| Early online interventions, then settle | 3.335 | 79/128 | 74/128 |
| Same early controller, frozen neural weights | 3.335 | 79/128 | 74/128 |
| Four undamped iterations, then λ=.5 | 4.065 | 80/128 | 78/128 |

Stopping interventions improved the continuous controller's cost and message
stability. However, the online and frozen settled controllers had **identical
final costs on all 128 bow-ties**. The apparent 1.779-point mean advantage over
the tuned fixed baseline has a paired 95% bootstrap interval of
[-4.059, 0.488], which includes zero. This is a promising sample mean, not an
established baseline improvement.

Transfer also limits the claim: on K4, tuned fixed scored 2.345 versus 3.260
for the settled online controller. Online learning improved only one K4 case
over its otherwise matched frozen control, tying the other 127.

The small graph is useful precisely because it separates three questions:
did the weights change, did the active choices/beliefs change, and did final
cost and stability improve? They are different outcomes.

## Recommended next research target

Keep this small graph and controller size. Replace generic weight movement
as a target with **active-choice-changing interventions**, and learn when to
stop them. A proposed split should be characterized by its conditional
minimizer margins and which clone choices it would change; retain damping as
the separate stabilization control. This next feature/action redesign has
not been implemented or shown to outperform the tested methods.

Retain fixed, frozen-learning, random-intervention, and deterministic-schedule
controls in every comparison. The current evidence supports pursuing this
more specific question; it does not justify a larger neural model or claiming
that learned splitting has already won.

## Verification and limitations

- The kernel matches the actual PropFlow pipeline with explicitly shared
  per-message gauges and tie handling, including live interventions. A broader
  audit caught and fixed a near-tie decoding difference of 3.47e-18; all final
  training and evaluation were rerun. This numerical correction reused the
  confirmation seeds and did not retune parameters on their outcomes.
- Independently reconstructed **442,368 saved costs**, with zero discrepancy;
  replayed 12 policies from saved inputs and verified source hashes.
- **50 focused tests passed**, including 16 new research checks. Formatting
  and lint checks pass for the new files.
- The wider repository has three independently reproduced Figure 5/8 test
  failures. `make ci` also stops at formatting issues in seven existing files;
  unrestricted pytest collection hits duplicate submission-directory modules.
  Those unrelated files were preserved.
- The two families use random three-label costs and frustrated binary costs;
  family and domain effects are not independently identified. Graphs are tiny,
  and stable finite tails are not convergence proofs. Split and unsplit runs
  share an iteration budget but differ in factor-update work. Training costs
  are separate from deployment timings. No large-instance gain or DABP rerun
  is claimed.

Literature: [DABP](https://arxiv.org/abs/2209.12000) motivates the online
comparison; [the AAMAS damping analysis](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2281.pdf)
motivates separating transients and stabilization. [Residual BP](https://arxiv.org/abs/1206.6837)
and [contextual bandits](https://arxiv.org/abs/1003.0146) informed observations
and chosen-action feedback. Their guarantees are not transferred to this
changing-split process. Additional sources and their scope are in README.md.
