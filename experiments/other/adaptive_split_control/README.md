# Adaptive splitting and damping: small controlled laboratory

Research started 2026-09-14. This is an experimental controller study, not a
replacement for the paper experiments or the public PropFlow engines.

## Question and protocol, recorded before running

Can an inexpensive controller use the project's sibling-feedback and
period-two findings to improve **final original-objective cost with stable
convergence**, beyond a well-tuned fixed splitting/damping configuration?

Use a five-variable bow-tie (triangles 0–1–2–0 and 0–3–4–0). Its six pairwise
factors become twelve factors after splitting. A triangle and a single edge
serve as mechanism controls; a four-variable complete graph tests transfer.
Random cost tables have domain three; frustrated anti-coordination tables
have domain two. Small continuous perturbations avoid accidental exact ties.
Exact enumeration (243 or fewer assignments) is an evaluation instrument only.

For each original factor C_e, maintain two tables w_e C_e and (1-w_e) C_e.
Changing w_e keeps their sum equal to C_e. This is actual cost splitting,
not arbitrary attention on received messages. Clone messages keep their
identity and history when w changes; no message reset or rescaling is hidden
at the intervention boundary. Swapping .05 and .95 mid-run is a distinct
intervention even though they are equivalent from symmetric initialization.
No claim is made that a zero-cost clone is equivalent to removing a factor.

Use native-style synchronous Q-then-R updates and Q damping:
Q_new = lambda Q_old + (1-lambda) Q_computed. Larger lambda retains more old
information. Gauge every message against label zero. Decode the first label
within 1e-12 times the sum of original table ranges of the minimum belief.
The same explicit gauge and tie rule is installed in the native PropFlow
reference through its existing hooks and computator API. A dedicated parity
test compares Q, R, assignments, and original costs through that real runtime.
This is a specified research convention, not a claim about unmodified default
floating-point tie behavior. No core implementation was changed.

### Two tracks

1. **Across instances / theory:** measure a fixed parameter surface and
   same-state interventions. Test the equal-clone reduction and the distinction
   between period-two assignments and period-two messages. Train a small shared
   action scorer on counterfactual eight-step continuations of training states.
2. **Online:** choose one edge/split and global damping every eight iterations.
   Update the same small scorer from the observed continuation only. No exact
   optimum, future result, counterfactual rollout, or test-instance retraining
   outside the trajectory is available to this controller.

The scorer is a 16-input, 8-hidden-unit tanh MLP with one scalar output (145
parameters), shared over candidate actions. Choosing the highest scoring
edge/action is hard attention, not DABP's neural message aggregation. Inputs
include one/two-step message residuals, sibling contribution, belief margin,
best-response disagreement, current split, and candidate damping/split.

Reward is the decrease in actual original cost divided by the sum of table
ranges, minus .02 times the block's assignment-flip fraction and .005 times
its final normalized message residual (clipped at one). This is a short-horizon
surrogate, not a theorem or a direct final-horizon optimizer. Preserve an
incumbent assignment, but report terminal cost separately from best-seen cost.

Compare fixed unsplit and split configurations, validation-selected fixed
configuration, random switching, a period-two-triggered rule, a fixed
explore-then-damp schedule, a frozen offline network, offline-plus-online,
and online-from-scratch. Add a frozen network with exactly the same exploration
as the online model to isolate weight updates from random exploration.

Training seeds 0–23 per family; validation 100–111; test 1000–1031. Use 128
iterations, blocks of eight. Choose fixed settings on validation only; keep
test data out of model fitting. Report paired outcomes and finite-tail
stability, not asymptotic convergence. Track online wall time and update count.
Follow-up changes use fresh confirmation seeds and preserve the first result.

Numerical audit: a wider parity check found a native/reference decoding
difference at step 2 of unsplit, undamped bowtie/frustrated seed 2007:
the differing belief gap was -3.47e-18, and maximum Q/R discrepancy 6.94e-18.
The explicit tie convention above removes that arithmetic-order dependence.
All final results rerun both training and evaluation with this convention.
The earlier raw-argmin pilots are retained as intermediate artifacts.
Message stability checks both Q and R, rather than R alone.

### Follow-up protocol (after the pilot, before confirmation)

The pilot's learned policies had poor message stability. Test an early
intervention phase followed by fixed parameters and no further learning.
Select onset from {0,8,16,32,64} and settling damping from {.3,.5,.7,.9} on
validation seeds only. Onset zero is allowed: it can reveal that skipping
the controller is best. Compare online updates with an otherwise identical
frozen model, including the same exploration probability and settling phase.
Also tune an undamped-then-damped equal-split schedule with switch time
{0,4,8,16,32,64}; its zero case includes immediate damping. Confirm on fresh
seeds 2000–2063 per family and topology. No confirmation result selects a policy.

## Literature and connection to existing research

- [DABP, NeurIPS 2022](https://arxiv.org/abs/2209.12000): online learning of
  damping and incoming-message weights using a smoothed-cost loss. It motivates
  the comparison, not a presumption of improvement in this project.
- [Damping analysis, AAMAS 2025](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2281.pdf):
  analyzes chains, cycles, and lemniscates and the role of transient inconsistent
  contributions. This motivates measuring entry separately from stabilization.
- [Residual BP](https://arxiv.org/abs/1206.6837): residuals guide asynchronous
  scheduling. Here they are controller observations; its convergence results
  do not transfer automatically to changing split factors.
- [Adaptive damping for GAMP](https://arxiv.org/abs/1412.2005): precedent for
  adapting damping to observed behavior in a different message-passing model.
- [Contextual bandits](https://arxiv.org/abs/1003.0146): an inexpensive
  chosen-action learning pattern. BP actions change future states, so standard
  contextual-bandit regret guarantees do not automatically apply here.
- [TRW-S](https://proceedings.mlr.press/r5/kolmogorov05a.html): a distinct
  convergent message-passing direction worth a later benchmark; no TRW-S
  guarantee is attributed to this splitting controller.

The immediate project basis is
`results/splitting_investigation_20260909/MATHEMATICAL_NOTES.md` and
`local_theory_track/PROTOCOL.md`: equal-clone reduction, sibling return,
held-state interventions, and the limitation of an instance-level two-cycle
result. This study uses no unary factors, avoiding the previously identified
unary-startup mismatch; parity is checked explicitly rather than assumed.

## Reproduce

```bash
uv run --no-sync python -m pytest tests/test_adaptive_split_control.py -q
uv run --no-sync python -m experiments.other.adaptive_split_control.code.run \
  --output results/adaptive_split_control_replay
uv run --no-sync python -m experiments.other.adaptive_split_control.code.confirm \
  --pilot results/adaptive_split_control_replay \
  --output results/adaptive_split_control_replay/confirmation
uv run --no-sync python -m experiments.other.adaptive_split_control.code.symmetry_probe \
  --output results/adaptive_split_control_replay/theory
uv run --no-sync python -m experiments.other.adaptive_split_control.code.verify \
  --pilot results/adaptive_split_control_replay
```

Saved inputs, model weights, row-level measurements, intervention traces,
source hashes, runtime settings, and the generated results report are in the
chosen output directory. These generated outputs are local research artifacts.

Choose a new directory for each replay; completed runs are protected against
overwriting. The completed investigation is summarized in [REPORT.md](REPORT.md).
