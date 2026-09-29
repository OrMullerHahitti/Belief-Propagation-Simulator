# Late splitting with permanent damping release

The user approved a local pilot on **dense seeds 0, 1, 2** on 2026-09-19.
Run the fixed-time cases first, then the best-checkpoint cases. The subsequent
50-instance studies require final approval after reviewing the pilot/handoff.

- Native unsplit DMS: 1,000 updates, damping 0.9.
- Fixed-time branch: split at zero-based update 1000, equal 0.5/0.5 R transfer,
  then 1,000 undamped updates.
- Best-checkpoint branch: select the earliest minimum-cost checkpoint within
  that same first 1,000 updates, restore the full state after its cycle events,
  split before the next update, then 1,000 undamped updates. Retain the original
  iteration index for normalization; do not restart the normalization clock.
- Check the final 100 assignments for a fixed assignment or exact period two.
  Other tails are reported without claiming a two-cycle or running its merge.
- MGM on the two-value menus from both branches (10,000-round cap), then B&B
  on tables conditioned on those menus, warm-started from MGM, 300 seconds per
  instance. B&B completion proves only the optimum within the menus.

The new engine composes `MidRunSplitEngine` and the existing Q-damping hook.
Normalization remains the native graph-diameter cycle schedule. The split
transfer clears old histories as the existing engine specifies; damping is
disabled before the first split update. No core PropFlow code is changed.

## Local pilot

From the repository root:

```sh
uv run --no-sync python -m experiments.aamas.late_split.run \
  --out experiments/aamas/runs/late_split_pilot_20260919 \
  --benchmarks random_dense --seeds 0 1 2 --workers 3 --phase both
```

The output directory must be new. `--resume` skips completed case/phase results
only when their evidence hashes still match. Source and configuration changes
require a new run directory. `--phase fixed` and `--phase best` permit separate
review; the latter requires the completed fixed-time evidence.

Each case stores the exact original graph/tables, hashes, prior baseline
slices, prefix costs/assignments, fixed/best portable message checkpoints,
post-split costs/assignments, and MGM/B&B assignments and completion status.
The runner verifies every recorded original cost independently. It checks the
unsplit prefix against the retained corrected DMS curve; before the best-point
continuation, it replays the checkpoint to the prefix end and checks costs,
assignments, final mailboxes, retained Q history and convergence-monitor state.
Those replay updates are validation overhead, not algorithm updates.

Snapshots contain only costs/assignments. Checkpoints contain inbox/outbox
messages, retained damping history, monitor state and the next native update
index. They are numeric JSON compressed with gzip, not executable pickle files.
The input NPZ preserves table dtypes and factor-axis/insertion order.

The full search window counts toward the best-checkpoint method's work even
when it restores an earlier state. Report terminal cost and best encountered
cost separately; compare the merges with the prefix incumbent as well as with
the two branches. Existing pulse and fixed-0.95 baselines are reused unchanged.

## Validation

```sh
uv run --no-sync pytest tests/test_aamas_late_split.py \
  tests/test_aaai_experiments.py tests/test_aaai_split_pulse.py -q
```

Tests cover checkpoint serialization/replay, normalization phase, actual native
DMS and undamped continuation equivalence, original-input identity, independent
original-cost scoring, and B&B versus exhaustive menu search on a small graph.

## Domain-20 pilot

On 2026-09-21 the user requested the same three-seed experiment with domain 20,
and separate cost-over-iteration line plots with split labels only on the x-axis.
Keep 50 agents, density 0.6, integer pairwise costs in [100, 200), and the same
tiny unary preferences. The generator uses the historical RNG convention and
matches the old domain-10 input exactly when passed domain 10. Changing the
domain generates new tables, so the old domain-10 costs are not reused.

```sh
uv run --no-sync python -m experiments.aamas.late_split.run \
  --out experiments/aamas/runs/late_split_domain20_20260921 \
  --benchmarks random_dense --domain-size 20 --seeds 0 1 2 --workers 3 --phase both
uv run --no-sync python -m experiments.aamas.late_split.plot_comparison \
  experiments/aamas/runs/late_split_domain20_20260921
```

For each seed the runner saves a new domain-20 input and executes fresh DMS and
standard damped 0.5-splitting controls for 2,000 updates. Both controls preserve
full assignment traces and independently verified original costs. The unsplit
prefix must match this new DMS control exactly in assignments and costs. The
fixed-time and best-checkpoint continuations, tail check and MGM/B&B limits
remain unchanged. All fixed-time cases finish before the best-checkpoint phase.

Each plot shows standard damped splitting and both late-split trajectories.
The blue/orange portions preceding their marked split points are unsplit DMS.
The best-state curve uses its native update indices and ends after 1,000
post-split updates. The full 1,000-update checkpoint-selection window still
counts as computational work; these figures compare trajectories, not runtime.
MGM/B&B results are recorded separately in `COMPARISON.md` and are not inserted
into the BP curves. PNG and vector PDF files go in the run's `plots/` directory.

## All observed tail values and complete BP-to-MGM plots

The follow-up MGM extension uses every value each variable visits in the final
100 post-split updates. It starts MGM independently from every distinct complete
assignment in that window and retains the best result. Fixed tails have one
start; two-cycles have two; seed 0's best-checkpoint tail has seven starts and
two variables with three candidate values. These are observed-tail menus for
that nonperiodic case, not a claim that its complete future cycle is known.

```sh
uv run --no-sync python -m experiments.aamas.late_split.run_tail_mgm \
  experiments/aamas/runs/late_split_domain20_20260921
uv run --no-sync python -m experiments.aamas.late_split.plot_pipeline \
  experiments/aamas/runs/late_split_domain20_20260921 \
  --mgm-dir mgm_tail_values --stem pipeline_tail_mgm
```

The extra plots preserve the original curves and append the winning MGM history
after 1,000 post-split updates. Split and MGM-start events are marked on the
x-axis, and circles mark final MGM results. The axis counts native BP updates
followed by MGM rounds, not wall time or total work across all starts. Restoring
a selected tail assignment is shown as a dotted initialization segment at the
MGM boundary, separate from improving MGM rounds. The full checkpoint-selection
window still counts as search work even though its discarded suffix is not drawn.
The original no-MGM plots and earlier two-value results remain available.

## Approved 50-seed domain-20 expansion

The September 21 population run uses seeds 0–49 of the same dense domain-20
problem, reusing the three pilot cases and computing 47 new instances. It
keeps both late-split methods and all-tail-values MGM, with no new B&B runs.
See [POPULATION_DOMAIN20.md](POPULATION_DOMAIN20.md) for the exact protocol,
resume commands, checkpoint coverage, saved files and mean-curve interpretation.

The output directory is `experiments/aamas/runs/late_split_domain20_50seeds_20260921/`.
`population.py` resumes complete seed/phase results and post-split checkpoints;
`plot_population.py` validates the retained evidence and produces separate
fixed-time and best-checkpoint means plus MGM close-ups. Each mean includes
all 50 seeds and counts the entire 1,000-update observation window before
splitting. The raw native-index traces remain available for individual-seed
investigation and MGM-only reruns.

The completed run's mean final costs are 98,216.71 for the standard baseline,
98,192.04 for fixed-time splitting plus MGM, and 97,516.03 for best-checkpoint
splitting plus MGM. The latter beats the standard baseline on 48/50 seeds, but
does not beat the damping-only DMS best-seen mean of 97,453.77. Full figures,
per-seed comparisons and validation are saved in the run directory; see its
`RESULTS.md`. All 100 BP continuations and all 100 MGM results pass validation.
