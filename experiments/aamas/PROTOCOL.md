# Late split with damping released — approved pilot protocol

**Approved on 2026-09-19:** implement and run both versions locally on dense
seeds **0, 1, 2**, then prepare a handoff for final approval before 50-instance
execution. The sequence and checkpoint rule below were approved. The original
five-seed pilot proposal is reduced to three seeds by the user's latest reply.
Keep the existing pulse results and protocol unchanged.

## Proposed execution order

1. Validate the implementation on a tiny native case, including the split
   boundary, original cost, and checkpoint replay. This follows approval.
2. Fixed-time pilot: `random_dense`, seeds 0–2, existing 50-variable/domain-10
   instances. Run 1,000 unsplit DMS updates with old-Q damping 0.9. Before
   zero-based update 1000, split every factor 0.5/0.5, set damping to zero,
   and run another 1,000 updates (last index 1999). Use the existing `transfer`
   convention: redistribute each old R message equally to its two clones.
   Record the actual normalization schedule; do not silently substitute a
   different floating-point kernel or normalization cadence.
3. Inspect the final 100 assignments of each split continuation. A period-two
   tail requires exact assignment repetition at lag two throughout that window,
   with two distinct parity assignments. A fixed assignment, longer cycle or
   unsettled tail must be reported as such. Assignment period does not by
   itself establish a period-two message orbit.
4. Extract the assignments at indices 1998 and 1999. Run the existing MGM-1
   merge on each variable's menu `{x_even[v], x_odd[v]}`, once from each branch,
   to no improving move (existing 10,000-round cap); keep the better feasible
   result. Then run B&B on the same menus, warm-started from that MGM result,
   with a 300-second cap per instance. A finished search is exact only within
   the menus. A timeout gives a feasible incumbent, not a proved optimum.
   If no two-cycle is observed, retain the run in the results but do not call
   its last two assignments oscillation branches; the proposed cycle-merge
   analysis is not applicable. A fixed assignment gives a no-op merge.
5. Best-checkpoint pilot on those same inputs: examine only the first 1,000
   unsplit DMS updates, select the earliest minimum-original-cost checkpoint,
   restore its full dynamic state (messages, damping history and update phase,
   not just the best assignment), split and release damping as above, then
   run 1,000 post-split updates. Repeat the same tail check, MGM, then B&B.
   Count the full 1,000-update search prefix in computational cost even when
   the restored checkpoint is earlier. Separate search work, checkpoint index,
   and post-split time when plotting this branch.
6. Review the pilots before expanding. Proposed full order: fixed-time version
   on all five binary benchmarks, seeds 0–49; then best-checkpoint version on
   those same instances. Extension to the three ternary families is a separate
   scope decision, with their previous data already retained.

The best-checkpoint version is a retrospective/anytime restart after a fixed
observation window. It is not an online rule that knows a future minimum.
If the intended method is instead an online trigger or a restart from only
the best assignment, revise this proposal before implementation.

## Comparisons and recorded evidence

Reuse the corrected original data for DMS, immediate undamped 0.5 split and
its old merges, immediate damped 0.5 split, late split with damping retained
at K=1000, fixed 0.95 split and the existing pulse. Compare terminal original
cost and best encountered cost separately. A claim that the two late branches
are better must compare each branch's cost, not only the better branch or
their average. MGM/B&B must also beat the unsplit prefix's best encountered
assignment to improve what was already available before splitting.

Save original input tables/axis order, seeds, resolved parameters, source hash,
costs/assignments, selected checkpoint state, the two candidate assignments,
and merge outcomes with completion flags. Read new iteration results via
snapshots. Keep full state only for the current best checkpoint and the small
tail needed for verification, rather than retaining every engine object.
Reuse prior data where sufficient; new checkpoint states and assignment tails
need fresh approved execution because cost-only CSVs cannot reconstruct them.

After results are available, prepare cost-versus-update plots with the split
and merge marked, plus a clear view of the two tail assignments' costs and
the MGM/B&B result against the prefix incumbent. Do not present an average
cost curve alone as evidence of oscillation. Additional plots are optional
until the user selects the useful ones.

## Approval record and remaining decision

- Approved: K=1000, damping 0.9 before splitting and zero afterward, equal
  R transfer, 1,000 post-split updates, tail validation, MGM then capped B&B.
- Approved: restore the earliest best full checkpoint from the first 1,000
  updates; run both pilots before the fixed-time full suite and then the
  best-checkpoint full suite.
- Approved: run the pilot locally on 2–3 seeds; selected seeds are 0, 1, 2.
- Pending: final approval of the 50-instance handoff after pilot review.

Only the local three-seed pilot is authorized for execution now.
