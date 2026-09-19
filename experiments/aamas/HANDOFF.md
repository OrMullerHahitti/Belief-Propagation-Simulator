# AAMAS late-split handoff — awaiting final approval

## Pilot finding

Dense seeds **0, 1, 2** completed both approved variants: split after 1,000
DMS updates, then split from the earliest best checkpoint in that window.
All six final 100-assignment tails were fixed. Both branch assignments were
identical, so MGM and B&B were no-ops. Final original costs were respectively
**97,895.285399**, **97,632.225847**, and **98,796.285596** for both variants,
equal to each prefix's best encountered cost. The best checkpoint indices
were 541, 410 and 429 (zero based).

The pilot therefore does **not** demonstrate the proposed two-cycle/merge
improvement mechanism. It verifies execution and shows incumbent retention
on these inputs. Seed 1's result is better than DMS's terminal assignment,
but it is not better than DMS's earlier incumbent; damped late splitting
already reaches the same endpoint on these seeds. This is not population
evidence for superiority or convergence.

See `runs/late_split_pilot_20260919/SUMMARY.md`, `summary.csv`, the two figures
in its `plots/`, and the per-case NPZ/checkpoint/result files. The run completed
with exit code zero. The local tmux session exited normally.

## Work awaiting approval

The proposed full run is **50 seeds (0–49) in each of five binary families**:
`random_dense`, `random_sparse`, `scale_free`, `graph_coloring`, and
`meeting_scheduling`. This means 250 original inputs and 500 continuations.
It has not been launched.

1. Run all fixed-time cases: 1,000 unsplit DMS updates at damping 0.9; split
   all factors equally using native R transfer; 1,000 undamped updates.
2. Classify each final 100-assignment tail. For a two-cycle, extract its last
   two assignments, run MGM from both, then menu-conditioned B&B warm-started
   from the better MGM result (300-second cap). Fixed tails get a no-op merge;
   other tails remain recorded and are not silently dropped or called cycles.
3. Only after every fixed-time case completes, run every best-checkpoint case:
   restore the earliest minimum-original-cost state from that same 1,000-update
   search window, then 1,000 undamped split updates and the same merge process.
4. Return the complete run directory, including unsuccessful/capped cases,
   manifest, frozen source, exact input tables, checkpoints and assignments.
   Analysis and final plots follow review; do not tune parameters on these seeds.

The prior pulse and fixed-0.95 results are reused, not rerun. Ternary studies
are preserved in the main workspace but are not part of this proposed run.
The new comparisons exclude the old elementwise-random `DMS_split_0.4_0.6`
line: on structured benchmarks its clones can be asymmetric, so symmetry of
the original tables alone does not establish immunity to the old axis bug.
Those rows remain preserved and require a separate provenance audit before use.

## Execution package

`runs/late_split_handoff_20260919.tar.gz` is a focused executable package.
It contains the native library source, experiment dependencies, locked Python
environment specification, focused tests, corrected binary baseline CSVs and
metadata, and the completed pilot evidence. It contains no virtual environment,
credentials, unrelated working files, or prerequisite links outside the package.
The rest of the migrated research collection remains in the main repository.

The package uses the pilot's exact runner/core bytes. Its `FILE_SHA256.json`
records every packaged file. The archive has an adjacent `.sha256` file.
Extract it into a new directory, then work inside `late_split_handoff_20260919/`.
The pilot used Python 3.13.9. Set up and check the package before any full run:

```sh
uv sync --locked --extra dev --python 3.13
uv run --no-sync python VERIFY_PACKAGE.py
uv run --no-sync pytest tests/test_aamas_late_split.py -q
uv run --no-sync python -m experiments.aamas.late_split.run --help
```

After final user approval, from that directory on macOS/Linux:

```sh
tmux new-session -s propflow-aamas-full
bash RUN_AFTER_APPROVAL.sh
```

Reattach with `tmux attach -t propflow-aamas-full`. The script writes a named
log and exits on errors. It uses four worker processes and runs fixed-time then
best-checkpoint phases. To resume the *identical* run after interruption, append
`--resume` to its command; completed case/phase artifacts must pass hash checks.
Changing source, parameters or input selection requires a new output directory.

Do not run the package's full-run script as a smoke test: it is the 50-instance
job and is gated by the requested final approval.

## Verification and compute budget

- 51 focused tests pass, including native damping-release equivalence and
  portable checkpoint replay. The same corrected DMS prefix matches the saved
  CSV at its writer's four-decimal precision. Original costs and restored
  trajectories are checked at full precision.
- The isolated handoff directory passed all 11 new tests with imports verified
  to resolve to its own copied native source. This used the current machine's
  installed dependencies; a fresh dependency installation on the other host
  has not been tested here.
- Every pilot cost was independently reconstructed from original tables and
  stored variable-axis order. Best checkpoints replay to the prefix endpoint
  with identical assignments and complete final dynamic state.
- The original pilot attempt stopped because its CSV comparison demanded
  precision absent from the old file. Its source/logs remain preserved locally;
  no continuation from that stopped attempt is included as a result.
- `make ci` in the main repository still stops on seven pre-existing formatting
  files; the new late-split code passes scoped Black/flake8 checks.
- Three concurrent fixed-time cases took about 54 seconds, followed by about
  42 seconds for the three best-checkpoint cases, including validation replay.
  All menus were singletons, so this gives no useful estimate of hard B&B time.
  At most 500 searches can each consume their 300-second allowance: **41.7
  worker-hours** for capped B&B alone (roughly 10.4 hours at four workers if all
  calls hit the cap), plus native BP, replay and I/O. Actual costs depend on the
  graph family and host. Timeouts must remain explicit feasible incumbents.

The next decision is whether to run this unchanged 50-per-family protocol
despite the pilot's absence of two-cycles. No larger job is queued or scheduled.
