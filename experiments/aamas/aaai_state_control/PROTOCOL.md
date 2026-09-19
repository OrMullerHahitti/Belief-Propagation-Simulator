# State-based splitting and damping study

Authorized September 15, 2026: complete the mechanism, adaptive-trigger, and
offline/online-learning plan. This study has separate sources and outputs from
the completed temporary-split confirmation. No existing manuscript, runtime
default, or completed result is changed.

## Questions and fixed references

1. Does changing the minimizing rows mediate the pulse's immediate effect?
2. Is damping needed throughout, during the pulse, or only while settling?
3. Do timing, duration, and amplitude effects support a state-based trigger?
4. Can a small learned selector improve on the fixed pulse, and do actual
   within-run parameter updates help beyond a frozen matched selector?

References are equal splitting with old-Q damping .9, and the completed
schedule: equal splitting, .95/.05 pairwise splitting before update 64, restore
equal splitting before update 256, damping .9 throughout. Every method retains
messages and the original objective. Unary factors remain equally split.

## Stages and data separation

- Development: seeds 18000--18007, four-variable domain 10 random graphs,
  five-variable binary frustrated bowties, and actual 50-variable sparse/dense
  paper graphs. No small-graph optimum enters a policy observation or target.
- Mechanism: log minimizing-row commitment/churn, belief/assignment changes,
  original cost and damping-independent fixed-point defect. At a fixed
  pre-intervention Q, compare weights immediately inside and outside effective
  row boundaries. An immediate row change is not by itself a long-run benefit.
- Ablations: hold start 64/duration 192/weight.95 except for one tested axis.
  Starts 16/32/64/128/256; durations 32/64/192/384; weights.51/.65/.8/.95.
  Compare damping .9 throughout, zero throughout, zero only during the pulse,
  and zero until restoration followed by .9. Include matching damping-only
  controls so a damping effect cannot be attributed to splitting.
- Offline training: seeds 18100--18111, tiny/domain 10 and sparse inputs.
  Obtain counterfactual training outcomes using actual 256-step continuations
  from both reference and exploratory states; labels use observed original
  costs and stability, never exact-optimum labels. Charge all training work.
- Validation: seeds 18200--18207, tiny/domain 10 and sparse inputs. Select a
  deterministic state rule and a small linear action scorer using only these
  data. Compare a frozen scorer, frozen scorer with exploration, offline-trained
  online scorer with matched exploration, and online learning from scratch.
- Reserved confirmation: seeds 19000--19015, tiny/domain 10 and sparse inputs.
  Freeze the exact methods and parameters before constructing these inputs.
  Run every selected comparison for 10000 updates, with primary readout 2000.
  All methods use the same live update budget. Record policy overhead separately.

## Learning and control constraints

The action set includes hold, a global split pulse, a split pulse with zero
damping during the intervention, damping-only intervention, and an edge subset
chosen using predicted minimizing-row changes. The pulse duration is 192 and
decision blocks are 256 updates. At least five decision opportunities precede
the final settling period, so feedback can affect later actions within a run.

Theory-derived features describe minimizing-row commitment, churn, predicted
response to asymmetry, margins, original-cost progress, assignment switching,
and undamped fixed-point defect. A small linear predictor scores observed
cost progress and stability. Frozen and online arms start from identical
parameters; matched exploration uses identical random draws. Online updates
use only the chosen action's subsequently observed block, without simulated
counterfactual feedback or future labels.

## Outcomes and claim boundaries

Primary: original terminal cost including unary preferences. Strict stability:
constant final 100 assignments, all pairwise/unary Q/R gauge changes below
1e-7 of summed original factor ranges, and undamped Q-map defect below the
same threshold. Report paired stability regressions as well as total counts.
Use paired seed bootstrap intervals separately by family. Any method selected
as an improvement must lower validation mean cost without reducing either
assignment or strict-message stability counts relative to the fixed pulse.
Report failures even if no learned or adaptive candidate passes this gate.

Small graph exact optima are evaluation diagnostics only. Preserve all inputs,
trajectories, action events, model snapshots, imported-source copies and hashes.
Verify original costs independently and replay selected policies through the
real native engine. An empirical schedule improvement is not a proof of the
paper's proposed mechanism, convergence theorem, or online-learning benefit.

## Confirmation freeze, after validation and before constructing seed 19000

Run nine methods on each of the 32 reserved inputs, all through 10000 updates:
`baseline`, `pulse`, `state_plateau`, `frozen`, `frozen_explore`, `online`,
`scratch_frozen_explore`, `scratch_online`, `pulse_no_damping_during`.

The candidate satisfying the validation mean-cost/stability gate in both
families is `scratch_online`; its advantage over the pulse occurs on only one
of eight inputs per family and both marginal bootstrap intervals include zero.
This is a candidate for confirmation, not a demonstrated improvement.
`state_plateau` and `pulse_no_damping_during` are diagnostic comparisons: they
did not pass the gate in both families. No parameters are selected using the
reserved inputs. Starts 32 and 64 were compared on validation; start 32 did not
improve on 64, so do not expand its confirmation testing.

Five actions and nine features with two linear heads give 90 prediction weights.
Use the unchanged model from `training/offline_model.npz`, ridge .1, instability
penalty .05, exploration .2, six choices at 32/288/544/800/1056/1312. The scratch
control uses identical initial zero outcome estimates and identical exploration
draws; only `scratch_online` updates its sufficient statistics after feedback.
The scratch model resets for each independent input. It learns within a solve,
not across the sequence of confirmation inputs.

Validation provenance correction: the original evaluation observed rows every 8
updates only until 512, causing later learned choices to use a longer row-churn
window than training. `validation_v2` repeats all eight core comparisons after
correcting every learned observation to an 8-update cadence through 1568. Its
terminal costs and stability counts match the earlier validation. Use v2 for
learning claims; retain the earlier files as superseded evidence. A focused
test checks observation cadence before all six choices. The damping and timing
validation schedules are unaffected. An added native replay test checks a
preserved-message damping switch through the actual engine.
