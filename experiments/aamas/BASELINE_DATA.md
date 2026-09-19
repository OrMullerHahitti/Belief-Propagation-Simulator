# Reusing the previous data

Captured on 2026-09-19 from branch `aamas`, starting at `068f4d5`.
The `compute_R` axis-order correction is present in the source and was committed
as `9c2f4c8`. Data provenance must still be checked separately from source state.

| Data | Current evidence | Reuse status |
| --- | --- | --- |
| `previous_submission/data/{random_sparse,random_dense,scale_free}_*` | `20d0bb1` and metadata explicitly record replacement of all affected PropFlow lines after `9c2f4c8`. DABP rows were retained. | Use the corrected tables. |
| Unsplit and proportional-split lines in `previous_submission/data/{graph_coloring,meeting_scheduling}_*` | Their symmetric binary tables are unchanged by swapping axes; proportional splitting preserves that symmetry. The September 15 meeting summary records unchanged results. | Retain these lines; do not describe them as newly rerun. |
| Structured `DMS_split_0.4_0.6` lines | Elementwise random splitting can make each clone asymmetric even when the original table is symmetric. The metadata does not separately establish a post-fix rerun of these rows. | Preserved but not certified by the symmetry argument; excluded from the new runner's retained comparisons pending a separate audit. |
| Pulse and fixed-0.95 rows in all five binary families | Code `2377254`, data `bcd7276`, 50 seeds per family, same 2,000-update harness. | Retain alongside the corrected comparisons. |
| `previous_submission/ternary_data/{random_sparse_ternary,random_dense_ternary,meeting_scheduling_ternary}_*` | Existing local modifications; `logs/rerun_ternary_axis_fix_20260915.log` records completion of all three reruns and regenerated analyses/plots. Metadata has 50 seeds and 2,000 updates but no immutable code hash. | Preserve this local dataset and log together; current hashes identify the exact files. This migration did not rerun them. |
| `previous_submission/data_cuda/`, `backups/`, `fix/before*` | Historical comparison and backup material; not the authoritative corrected dataset. | Keep out of new baseline comparisons. |
| `experiments/aij/` and earlier DABP studies | Outside this migration; no fresh corrected-run audit here. | Retain as historical material; do not silently pool with the benchmark results. |

The eight current final-cost tables each contain 50 distinct seed rows per
algorithm with no duplicate algorithm/seed keys. This is a structural check,
not a fresh numerical reproduction; an `Optimal` row can still contain NaN
when its capped search did not finish. The raw histories include ordinary
0–1999 trajectories and shorter merge outputs, as in the prior harness.

The older `experiments/FINDINGS.md` note saying the reruns are still in progress
predates both the committed binary rerun and the local ternary completion.
The newer evidence above takes precedence for this inventory.

`provenance/baseline_files.json` hashes the current CSVs and metadata. In
particular, a clean checkout of the current commit will not reproduce the
working-tree ternary files. Preserve those files before handoff or cleanup.

Seeds alone are insufficient to prove identical inputs after a builder change.
New runs must reuse saved original tables when available; otherwise use the
unchanged builders and verify input fingerprints and a baseline replay before
accepting a comparison. Evaluate every assignment on the original tables with
their stored variable-axis order, including the unary preferences.

The inherited splitting laboratory uses a separate floating-point kernel. Its
saved reports are carried forward as historical evidence, not a newly verified
match to every native trajectory. Its `exp0_checks.py` now targets `aaai/data/`
instead of `data_cuda/`; that comparison has not been rerun during migration.
