# State-based splitting and damping research

**Completed result:** scratch online lowers sparse final cost slightly versus
the fixed pulse, but has worse strict stability at the primary 2,000-update
endpoint. Offline-trained online updates add no final-cost benefit over their
frozen exploration control, and tiny K4 shows no scratch-learning benefit.
The report separates these outcomes from the verified immediate splitting
mechanism and the promising but unconfirmed damping-switch result.

This isolated experiment implements the authorized mechanism study, state-based
control, and offline/online learning plan. Its anchor is the existing AAAI
splitting investigation, especially the complete minimizing-row inequalities.
There is no DABP implementation, attention-message multiplier, differentiable
BP training, or exact optimum inside the controller.

See `PROTOCOL.md` for the frozen comparisons and `REPORT.md` for results.
Evidence is under `results/aaai_state_control_20260915/` from the repository root.
The completed earlier pulse study and the manuscript are unchanged.

## Methods

- Fixed reference: pairwise .5/.5 splitting, old-Q damping .9.
- Fixed pulse: .95/.05 from zero-based update 64 through 255, then .5/.5 again.
  Messages are retained. Damping remains .9; unary factors remain equally split.
- Mechanism: one-edge, one-update changes immediately inside/outside the next
  effective minimizing-row boundary, followed by restoration and 2000 updates.
- Ablations: start, duration, amplitude, damping throughout/during/after;
  matching damping-only interventions isolate the split contribution.
- State rules: commitment, row changes, prospective response to asymmetry, or
  an observed cost/assignment plateau trigger a single pulse. A restoration
  variant also checks commitment. Rules observe every 8 updates.
- Learning: five actions, nine features, two linear outcome heads per action,
  giving 90 prediction coefficients. Fit ridge statistics offline on observed
  cost progress and instability from 720 actual continuation blocks. During an
  online run, update only the chosen action after its 256-update block.
  Six decisions permit feedback to affect subsequent actions; there is no
  counterfactual simulation at deployment. Each input starts a fresh learner.

Actions are hold, global pulse, global pulse with temporarily zero damping,
damping-only change, and a pulse on up to one quarter of edges ranked by
predicted minimizing-row changes. Local ranking evaluates current messages;
it is a deterministic theory-based attention rule, not a neural attention net.
Frozen/exploring/online and scratch-frozen/scratch-online comparisons share
initial models and random draws as applicable. All final costs use the original
objective including unary preferences. No best-encountered assignment is returned.

Tiny graphs are complete K4 with four variables/domain 10, and frustrated
binary bowties with five variables. The larger graphs use the original AAAI
50-variable/domain 10 sparse and dense generators. Every generated objective,
trajectory, intervention, learned decision, and model is saved.

## Reproduction

Run from the repository root with its existing uv environment. Every output
directory must be new; completed evidence is never overwritten.

```sh
uv run --no-sync python -m experiments.aamas.aaai_state_control.study ablate \
  --out results/state_reproduction/development \
  --families k4_d10 bowtie_frustrated random_sparse random_dense \
  --seed-start 18000 --seeds 8

uv run --no-sync python -m experiments.aamas.aaai_state_control.mechanism \
  --out results/state_reproduction/mechanism --seeds 2

uv run --no-sync python -m experiments.aamas.aaai_state_control.study train \
  --out results/state_reproduction/training \
  --families k4_d10 random_sparse --seed-start 18100 --seeds 12

uv run --no-sync python -m experiments.aamas.aaai_state_control.study evaluate \
  --out results/state_reproduction/validation \
  --families k4_d10 random_sparse --seed-start 18200 --seeds 8 \
  --model results/state_reproduction/training/offline_model.npz

uv run --no-sync python -m experiments.aamas.aaai_state_control.study evaluate \
  --out results/state_reproduction/confirmation \
  --families k4_d10 random_sparse --seed-start 19000 --seeds 16 \
  --horizon 10000 --confirmation \
  --methods baseline pulse state_plateau frozen frozen_explore online \
    scratch_frozen_explore scratch_online pulse_no_damping_during \
  --model results/state_reproduction/training/offline_model.npz

uv run --no-sync python -m experiments.aamas.aaai_state_control.native_replay \
  --stage results/aaai_state_control_20260915/validation_v2 \
  --out results/state_reproduction/native_replay \
  --case k4_d10 18201 scratch_online \
  --case random_sparse 18204 scratch_online

uv run --no-sync python -m pytest tests/test_aaai_state_control.py -q

uv run --no-sync python -m experiments.aamas.aaai_state_control.analyze \
  --root results/aaai_state_control_20260915 \
  --out results/state_reproduction/analysis
```

`study.py` freezes sources, protocol and native-source hashes before generating
inputs. The numerical kernel and original benchmark builders are copied from
the completed prior confirmation's saved sources. Source snapshots and trained
models, plus SHA-256 manifests, record the precise execution version. The
original validation learning-feature cadence was corrected before confirmation;
`validation_v2` is the authoritative validation for learned methods.

`native_replay.py` checks all saved assignments and original costs through
PropFlow's actual split/damping engine, uses `latest_snapshot()`, and additionally
compares all pairwise/unary messages at interventions and every 128 updates.
Full-horizon execution deliberately continues after the native early-stop signal
so that later interventions and restoration are executed.

Finite-tail stability requires 100 unchanged assignments and small changes in
all gauged Q/R messages, plus a small undamped Q fixed-point defect. It is an
empirical diagnostic, not a convergence or optimality proof. See the report for
paired cost uncertainty and stability regressions.
