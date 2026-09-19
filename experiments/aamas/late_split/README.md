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
