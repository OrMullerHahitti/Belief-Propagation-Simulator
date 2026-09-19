# Splitting, damping, and small offline/online control

The full plan is complete. **Scratch online gives a small sparse cost improvement
over the fixed pulse, but fails the combined cost/stability goal at the primary
2000-update endpoint.** Offline training followed by online updates adds no
terminal-cost benefit over its matched frozen exploration control. On tiny K4,
scratch learning ties its frozen control on all 16 fresh inputs. The immediate
splitting mechanism is supported; a general learning advantage is not.

## Fresh confirmation: main result

All nine frozen methods ran on 16 fresh inputs per family for 10000 updates,
with the prespecified primary readout at 2000. Costs are means over the same
inputs; stability counts require the full strict message criterion.

| Sparse method | Final cost at 2000 and 10000 | Stable at 2000 /16 | Stable at 10000 /16 |
|---|---:|---:|---:|
| Ordinary .5 split / .9 damping | 14101.193 | 16 | 16 |
| Fixed pulse | 14016.696 | 15 | 15 |
| State-plateau trigger | 14019.882 | 15 | 15 |
| Offline frozen | 14101.193 | 16 | 16 |
| Offline frozen + exploration | 14042.131 | 16 | 16 |
| Offline-trained online | 14042.131 | 16 | 16 |
| Scratch frozen + exploration | 14042.131 | 16 | 16 |
| Scratch online | **14008.191** | **14** | **15** |
| Fixed pulse, damping off during pulse | 14005.879 | 16 | 16 |

Scratch online versus the fixed pulse reduces sparse final cost by 8.504
(0.061%), with 7 wins, 9 ties and no losses. At the primary endpoint its paired 95%
bootstrap interval is [-17.444, -2.251]. Versus the matched scratch frozen control,
the reduction is 33.940, interval [-54.943, -15.687], 10 wins and 6 ties. However,
scratch online loses strict stability on seed 19013 relative to the pulse at 2000.
It recovers by 10000. Seed 19011 remains unstable for both pulse and scratch online;
both seeds are stable under the ordinary baseline and the frozen controls.
All sparse final costs are unchanged between 2000 and 10000.

The damping-switch schedule has the lowest observed sparse mean and 16/16
stability, but its advantage over the ordinary fixed pulse is uncertain:
mean −10.817, 10000-update interval [-47.440, 29.429], 8 wins, 4 ties, 4 losses.
It had failed the independent validation gate, and is a diagnostic finding
requiring a new prespecified confirmation, not a selected proven winner.

The fixed pulse still improves sparse cost over ordinary .5/.9 by 84.497
(0.599%), with 15 wins, 1 tie and no losses. Its 10000-update interval is
[-118.617, -53.247]. It has 15/16 strict stability versus the baseline's 16/16.

| K4 method | Final cost at 2000 | Stable at 2000 /16 | Final cost at 10000 | Stable at 10000 /16 |
|---|---:|---:|---:|---:|
| Ordinary .5 split / .9 damping | 667.334 | 15 | 656.959 | 15 |
| Fixed pulse | 654.833 | 14 | 660.771 | 14 |
| State-plateau trigger | 654.833 | 14 | 665.208 | 14 |
| Offline frozen | 656.959 | 15 | 656.959 | 15 |
| Offline frozen + exploration | 654.457 | 15 | 654.457 | 15 |
| Offline-trained online | 654.457 | 15 | 654.457 | 15 |
| Scratch frozen + exploration | 654.457 | 15 | 654.457 | 15 |
| Scratch online | 654.457 | 15 | 654.457 | 15 |
| Fixed pulse, damping off during pulse | 660.582 | 15 | 654.457 | 15 |

K4 illustrates why the long endpoint is necessary: the baseline changes on
seed 19000, while the fixed/state pulses change on 19000 and 19001. The same strict
stability counts do not imply unchanged endpoints on unstable inputs. Scratch
online ties its scratch frozen control in 16/16 final costs at both horizons,
despite making 56/96 different action choices. At 10000, scratch online reaches
the exact optimum on 15/16 inputs; that is not evidence for learning because the
frozen control reaches the same costs. Exact optima are computed only afterward
for evaluation.

Offline online updates were actually executed 96 times per family. They changed
one of 96 K4 action choices and zero sparse choices relative to the matched frozen
exploration model, with no final-cost differences. Scratch updates changed 56
K4 and 42 sparse choices, confirming that the online code used feedback even
where it did not improve the endpoint.

**Decision:** do not promote a learned controller as a joint cost/stability
improvement. Keep the established pulse as the main sparse reference, report
its stability limitation, and retain scratch online and the damping switch as
qualified follow-up candidates. A new learning claim needs fixed action-sequence
and feature-ablation controls; a damping-switch claim needs independent
confirmation because validation and confirmation disagreed.

## Scope and implementation

This study completes the authorized mechanism study, state-based trigger,
damping ablations, offline training and actual within-solve online learning.
The anchor is the AAAI splitting investigation and its complete active-row
inequalities. There is no DABP code or architecture in these policies.
See [research rationale](RESEARCH.md) for exact source versions, algebra, and
the supporting residual-scheduling and linear-bandit literature.

The small test systems are K4 (four variables, six pairwise factors, domain 10)
and a frustrated binary bowtie (five variables). The paper-scale systems use
the original 50-variable/domain 10 sparse and dense input generators. All inputs
include the original unary preferences. The pulse changes complementary factor
table weights; it never changes the original objective or clears Q/R messages.

The learned selector has 90 prediction coefficients: five actions, nine
observations and two linear outcome heads. It also retains ridge sufficient
statistics. There are no hidden layers. Four observations explicitly come from
splitting theory: common-row commitment, row churn, prospective row changes,
and row margins. An edge-local action ranks current predicted row changes and
changes at most one quarter of pairwise factors. That deterministic ranking
supplies attention to individual edges; it is not learned neural attention.

Offline training uses 24 inputs and 720 counterfactual 256-update blocks, costing
185,088 solver updates including warmup. Targets are observed original-cost
progress and observed instability. Online execution chooses at 32/288/544/800/
1056/1312, receives feedback after each 256-update block, and updates only the
chosen action. It performs no counterfactual rollouts or oracle queries online.
The learned state resets between independent input graphs.

The fixed comparator remains .5/.5 splitting with old-Q damping .9. The fixed
pulse changes pairwise weights to .95/.05 before zero-based update 64 and restores
.5/.5 before 256, retaining damping .9 throughout. Unary factors remain equally
split throughout every experiment.

## Mechanism: what the paper argument explains

We tested 24 effective boundaries across eight development inputs, using up to
three edges per input. Each pair branches from identical Q/R after 64 updates:
change one edge to just inside or just outside its next active-row boundary for
one update, restore equal splitting, and continue through 2000.

- Inside: 0/24 changed a minimizing row; maximum immediate normalized combined
  belief difference was 1.294e-14.
- Outside: 24/24 changed a minimizing row; immediate belief differences became
  nonzero, with maximum 3.571e-6 of the original factor-range scale.
- All 48 interventions tied the corresponding baseline's final original cost.

This verifies immediate cancellation and branch sensitivity in the tested
states. Crossing alone was insufficient for improved final cost. It does not
establish the complete causal explanation of a192-update pulse.

The amplitude sweep reinforces this limitation. Weights .51 and .65 produced
the same final costs as the baseline on all eight sparse and all eight dense
development inputs. Yet both weights changed at least one predicted minimizing
row on every one of those inputs. Their failure cannot be explained simply as
"no boundary was crossed." Stronger or better-directed changes are needed to
alter the eventual solution in these examples.

Cancellation also has a precise condition: when complementary clones use the
same active sender choices, their weight dependence cancels. Once their active
choices differ, beliefs can vary continuously with weight inside a fixed active
region. Our crossing score is therefore useful but incomplete; it is not a
universal measure of whether reweighting can matter.

## Damping and timing

On development inputs, removing damping throughout gave 0/8 strictly stable
sparse runs and 0/8 dense runs, both with and without the pulse. Mean final costs
were substantially worse. K4 retained 3/8 stable runs without a pulse and 4/8 with
one; the frustrated bowtie retained 0/8 in either case. Damping is not a theorem-
level necessity for every graph, but it is essential to these successful fixed
pulse comparisons.

Temporarily removing damping during the pulse looked promising on development
sparse inputs, but reduced stability from 8/8 to 7/8. Independent validation did
not confirm a combined cost/stability improvement:

| Validation method | K4 mean final cost | K4 stable /8 | Sparse mean final cost | Sparse stable /8 |
|---|---:|---:|---:|---:|
| Ordinary .5 split / .9 damping | 663.892 | 8 | 14485.258 | 8 |
| Fixed pulse, damping .9 throughout | 661.642 | 8 | 14418.383 | 8 |
| Pulse, damping off only during pulse | 661.517 | 8 | 14434.005 | 7 |
| Damping off only during pulse window, no split change | 662.016 | 8 | 14485.258 | 8 |
| Pulse, damping enabled only after restoration | 661.891 | 8 | 14477.873 | 8 |

Step 64 has no special theoretical status. On sparse development inputs,
starting at 32 improved mean cost from 14817.370 to 14783.113 with 8/8 stable in
both cases. Starting at 16 reached 14777.738 but only 5/8 strict stability. The
separate validation reversed the start 32 advantage: K4 cost 662.767 versus
661.642 for start 64, and sparse 14418.758 versus 14418.383. All were stable 8/8.
We retained 64 without tuning on confirmation.

## Validation and selection

Seeds 18000--18007 were development;18100--18111 were training;18200--18207
were validation. The 32 confirmation inputs use 19000--19015 independently in
K4/domain 10 and sparse families. The final method list and settings were frozen
before constructing any confirmation input.

The initial evaluation had inconsistent row-churn windows after update 512.
This was corrected before confirmation, a focused regression check was added,
and all eight core comparisons were rerun as `validation_v2`. Outcomes matched
the earlier validation, which is retained but superseded for learning claims.

Scratch online passed the prespecified validation mean-cost/stability gate,
but its advantage over the pulse was only one win and seven ties in each
family: mean change−0.125 in K4 and−2.373 in sparse. Both marginal 95% bootstrap
intervals included zero. The state-plateau and damping-switch methods did not
pass the gate in both families; they were retained as diagnostic comparisons.

The offline-trained online model matched its frozen exploration control on
validation. Scratch online improved sparse mean cost over its matched scratch
frozen control by 59.746, with 5 wins,2 ties,1 loss and stability 8/8 versus 7/8.
This motivated confirmation, rather than a claim that learning was already
better than the fixed pulse.

## Verification and reproducibility

The evidence root is `results/aaai_state_control_20260915/`. Each stage records
its command arguments, frozen source copies, saved original objectives,
cost/assignment/message diagnostics, action events, model states and SHA-256
hashes. `confirmation/PROTOCOL.md` contains the executed method freeze.

Native replays cover K4 and sparse scratch-online trajectories at 2000 and 10000
updates:24,000 assignments in total. Every assignment and original cost matched.
All sampled pairwise and unary Q/R messages matched exactly; samples include
interventions, every 128 updates and the final update. Maximum native-versus-fast
reported cost discrepancy was 7.276e-12. These cases include global and local
split pulses and damping switches; they were selected for intervention coverage.

All 16 focused tests pass. The broader test suite returned 383 passed,2 skipped,
and 3 failures in untouched Figure 5/8 reproduction tests. `make ci` stopped at
seven existing formatting failures outside the new experiment. New experiment
code and its focused test pass Black and flake 8. See the saved `pytest.log` and
`make_ci.log`; these unrelated failures were not modified.

All 2,880,000 confirmation costs were independently reconstructed from saved
assignments and original factor/unary tables; maximum error was 3.638e-12.
The completed source/evidence audit checked 4650 saved artifact hashes with no
mismatches. [Primary-endpoint figure](../../../results/aaai_state_control_20260915/analysis/confirmation_2000.png)
and [long-endpoint figure](../../../results/aaai_state_control_20260915/analysis/confirmation.png)
show paired cost intervals and strict stability. The first development sparse
input is used for the mechanism trace without ranking cases by outcome.

The separate instrumented policy profile reproduces saved validation
trajectories and isolates disjoint policy-entry calls. On one sparse input,
the first measured repetition attributed 0.259s to offline-online policy calls
and 0.268s to scratch-online policy calls over 2000 updates, including state
observations. Total instrumented runs were 1.380s and 1.457s. These timings include
profiler overhead and research diagnostics; they are not deployment latency
claims. Raw repeated measurements are in `policy_profile/profile.json`.

## Claim limits

Strict stability means 100 constant decoded assignments, all pairwise/unary
gauged Q/R changes below 1e-7 times summed original factor ranges, and undamped
Q-map defect below the same threshold. It is finite-window evidence, not a
proof of convergence. Final cost is the actual last assignment; best-encountered
cost is saved only as a diagnostic. Paired seed-bootstrap intervals are marginal,
not adjusted for the full exploratory family of comparisons.

Neither a benefit over a frozen model nor a benefit from a state rule alone
establishes that learning or all four splitting-derived features are necessary.
That would require feature ablations and a matched fixed intervention sequence.
Dense and frustrated-bowtie results are development evidence only. This study
does not claim general graph-family superiority or global optimality.
