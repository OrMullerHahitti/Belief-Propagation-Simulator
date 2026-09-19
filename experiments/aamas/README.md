# AAMAS experiments

This directory collects the recent work for the next submission. Inclusion is
an inventory decision, not a decision to include a study in the paper. The new
[late-split experiment](late_split/README.md) is implemented and its approved
three-seed local pilot is complete. The exact order and parameters are in
[PROTOCOL.md](PROTOCOL.md). The 50-instance studies await final approval.

## What moved here

The six September 15 studies moved from `experiments/other/`; the September 13
splitting laboratory moved from `experiments/splitting_explanation/`. Imports,
test imports, current command examples, and repository-root lookup were updated.
Saved numerical outputs and frozen source copies retain their original bytes.

| Directory | Introducing commits | What it contains and what the existing evidence says |
| --- | --- | --- |
| [adaptive_split_control](adaptive_split_control/README.md) | `ff1cc5d` | Small-graph learned splitting/damping controller. The apparent short-horizon advantage disappeared at 2,000 updates; no established advantage over the fixed baseline. |
| [aaai_derived_control](aaai_derived_control/README.md) | `6fd6ec6` | Fixed split pulse, native engine, fast kernel, and boundary calculations. A separate sparse confirmation improved terminal cost, with finite-tail stability limitations. |
| [undamped_split_nodes](undamped_split_nodes/) | `354f92e` | Saved-input replay, beliefs, alternating assignments, commitment transitions, and an interactive explorer. Descriptive evidence about selected inputs. |
| [aaai_state_control](aaai_state_control/REPORT.md) | `40aefeb` | State-triggered and learned interventions. No general joint cost/stability advantage; includes a distinct temporary damping-release diagnostic. |
| [damping_causality](damping_causality/README.md) | `6100bb0` | Native same-state interventions and exact small examples explaining how damping changes the reached solution. |
| [damping_generalization](damping_generalization/README.md) | `fc09cf3` | Conditional stability and split-escape certificates, counterexamples, and numerical limitations when damping is released. |
| [splitting_explanation](splitting_explanation/README.md) | `34605a6`, `27e719a`, `f49eeef`; corrections in `6100bb0` | Six mechanism studies plus correctness checks, Hebrew/formal explanations, existing plots and results. Historical claims keep their original qualifications. |

Snapshot diagnostics from `2e87774` remain in `src/propflow/snapshots/`, their
proper library layer. Existing tests remain in `tests/`. Earlier DABP studies
remain under `experiments/dabp_*`; they have not been selected for migration.

## Previous submission and pulse results

[previous_submission](previous_submission/) is a relative link to
`experiments/aaai/`, which remains the source of the benchmark builders, merge
helpers, previous data and plots. No benchmark CSV or existing plot was moved
or overwritten. Use [BASELINE_DATA.md](BASELINE_DATA.md) to choose reusable data.
New AAMAS code and outputs belong here, not in that historical directory.

The benchmark pulse and fixed asymmetric split were introduced by `2377254`;
their five-family, 50-seed results were appended by `bcd7276`. The pulse uses
old-Q damping 0.9 throughout, split 0.5/0.5 at updates 0–63, 0.95/0.05 at
64–255, and 0.5/0.5 from 256 onward. It keeps messages at both changes. This
differs from the proposed late split, which starts unsplit and releases damping
permanently when splitting. Fixed 0.95/0.05 results are retained as a comparison.

The historical benchmark runner now imports the migrated
`aaai_derived_control.code.pulse.SplitPulseEngine`. Its default output still
points at the old submission: any approved new run must explicitly write to
`experiments/aamas/runs/<run-name>/`. The new runner is `late_split/run.py`.

## Saved evidence and portability

[evidence](evidence/) contains relative links to the existing local research
results, including frozen inputs, source copies, trajectories and reports.
These targets live under the repository's ignored `results/` directory;
**a Git checkout alone does not contain them**. Historical scripts retain their
original evidence paths so that frozen-input studies remain reproducible.
When moving to another computer, include the required evidence targets, the
working-tree ternary data, and the migrated splitting laboratory's local NPZs.
Copying only this directory will not resolve the links. A run package will be
prepared for review before the 50-instance run; see [HANDOFF.md](HANDOFF.md).

`provenance/migration.json` records original paths, destination paths, original
hashes, tracked status, and the starting commit. `provenance/baseline_files.json`
records hashes of all 56 current binary/ternary baseline files, including local
changes. These are capture records, not claims that every result was rerun.

Migration checks and the existing CI limitation are in
[VALIDATION.md](VALIDATION.md).

## Next steps

1. Select which inherited studies are relevant; all seven are currently kept.
2. Review the completed three-seed pilot: all six final assignment tails are
   fixed, their costs equal the prefix incumbent, and both merges are no-ops.
3. Review [HANDOFF.md](HANDOFF.md) and give final approval before a 50-instance
   run. No such run has been started.
4. After the larger results return, verify their inputs and protocol, summarize
   outcomes, and make plots. The pilot does not establish general behavior.
