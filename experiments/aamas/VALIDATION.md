# Migration validation — 2026-09-19

- Existing focused research suite: **193 passed**, across the 16 affected
  research/snapshot test files. This includes native pulse replay and original
  objective checks; it is not a new benchmark execution.
- Historical benchmark CLI `--help` loads successfully with the new pulse import.
- The benchmark integration tests also pass: **28 passed** in
  `tests/test_aaai_experiments.py` (221 focused/integration tests in total).
- The relocated splitting laboratory loads the real dense seed-0 input with
  50 variables and domain 10; repository-root lookups and local evidence links
  resolve. No long experiment entry point was invoked.
- All 119 migrated non-cache files are present. Existing numerical artifacts
  retain their bytes; 16 source/document files received path changes.
- All 56 baseline files match the capture hashes. All 33 pre-existing modified
  tracked files, including ternary data and generated plots, remain byte-identical
  to their state before this migration.
- `make ci` was attempted and stops at its formatting gate: the seven unchanged
  files are `tests/test_splitting.py`, `tests/test_bp_engine.py`,
  `tests/test_engines.py`, `tests/conftest.py`,
  `src/propflow/integrations/dabp/build.py`, `src/propflow/snapshots/analyzer.py`,
  and `src/propflow/snapshots/visualizer.py`. Later CI gates did not run through
  this command. These files were not changed by the migration.
- A separate formatting check on the 26 Python files with migration edits
  passes 25 and flags the moved `splitting_explanation/lab.py`. Checking its
  original `HEAD` contents confirms that formatting issue predates the move;
  only its repository-root lookup changed here.

Detailed migration logs and the before/after preservation audit are in
`results/aamas_migration_20260919/`. These checks preceded pilot approval.

After approval, the late-split implementation passed 51 focused native,
checkpoint, merge and benchmark tests. All six native pilot continuations
completed with independently verified original costs and exact checkpoint
replay. The baseline comparison matches the old CSV's four-decimal format.
`make ci` still stops on the same seven untouched formatting files.
These findings were recorded before the requested commit/push. No 50-instance
run was performed.
