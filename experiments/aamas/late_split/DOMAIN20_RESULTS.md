# Domain-20 pilot — 2026-09-21

Completed dense seeds 0, 1, 2 with 50 agents, domain 20, density 0.6 and the
unchanged late-split protocol. Fresh 2,000-update DMS and damped 0.5-splitting
baselines use the same saved original input as each seed's continuations.
The six runs completed; this is a three-instance pilot, not a 50-instance study.

Evidence and plots: `../runs/late_split_domain20_20260921/`. Separate PNG and
vector PDF line figures are in `plots/cost_lines_seed_{0,1,2}.*`. The figures
show native BP cost trajectories, with split labels only on the x-axis. Blue
and orange are unsplit DMS before their respective split points. The best-state
method still observes all 1,000 prefix updates before restoring its checkpoint;
the discarded suffix is not displayed on this trajectory axis.

| Seed | Standard damped split final | Best prefix cost | Fixed-time tail | Best-state split after update | Best-state tail |
| --- | --- | --- | --- | --- | --- |
| 0 | 96,822.244 | 96,588.239 | Two-cycle | 333 | Other/unsettled |
| 1 | 96,643.253 | 95,612.294 | Two-cycle | 944 | Fixed |
| 2 | 97,700.240 | 96,792.246 | Fixed | 199 | Two-cycle |

The fixed-time oscillations on seeds 0 and 1 give much worse branch costs.
MGM reduces them to 97,420.251 and 98,809.281 respectively, still worse than
the standard damped split and the prefix incumbent. Both B&B searches hit
their 300-second limit with the same incumbent; no menu optimum was proved.

The best-state seed-0 tail is neither fixed nor an exact two-cycle, so the
original pilot did not apply its cycle-merge procedure there. Best-state seed 1 stays at
95,612.294 and its merge is a no-op. Best-state seed 2 does form a two-cycle:
MGM reaches 96,762.247, approximately 30 below its prefix incumbent and 938
below the standard damped-splitting baseline. B&B proves the same cost optimal
within the final two assignments' menus. This is not a global optimum claim.
The fixed-time seed-2 run stays at 97,019.284 and its merge is a no-op.

See the run's `COMPARISON.md` and result JSONs for exact outcomes. MGM/B&B are
post-processing and are not inserted into the BP curves.

Validation: 48 focused tests passed, including domain-10 generator equivalence,
domain-20 input preservation, fresh-control prefix equality, checkpoint replay,
native undamped continuation and merge checks. All six runs passed evidence
hash, independent original-cost and tail-classification audits; all three
best checkpoints replayed exactly. `VALIDATION.json` records the completed audit.
`make ci` still stops on the same seven untouched formatting failures noted
in the earlier pilot. The plot legend was moved outside the data after execution;
algorithm sources are unchanged, and the final renderer is saved beside the plots.

## MGM-only follow-up from saved endpoints

The user subsequently requested MGM from these continuations. The follow-up
keeps the original two-value menus and starts MGM from both final assignments.
For the seed-0 best-state run, these are explicitly sampled endpoints of a
non-period-two trajectory. This extends the merge to that pair without claiming
it is an oscillation's two branches. No new BP or B&B run was needed.

The five previously eligible cases reproduce their existing MGM results exactly.
For seed 0's best-state pair, MGM reaches 96,164.240 in four rounds from the
penultimate assignment and 96,122.246 in six rounds from the final assignment.
The better result improves on the standard damped-splitting baseline by 699.998
and on the best original cost visited anywhere in that BP run (96,214.238) by
91.992. Seed 2's best-state MGM cost (96,762.247) had already been visited during
its BP continuation, although it improves on its prefix and final two states.

Seed 0's fixed-time blue tail is exactly period two, alternating between
103,496.246 and 103,827.262. Its best-state orange tail visits seven distinct
assignments in its last 100 updates. In that window, 36 variables are fixed,
12 use two values and two use three values. No exact period of length 1–100
repeats throughout its last 300 assignments. This is evidence about the observed
finite trajectory, not proof that it can never become periodic.

Results, reproducible scripts, MGM-round plots and seed-0 trajectory close-ups
are saved under `../runs/late_split_domain20_20260921/mgm_followup/`. Endpoint
hashes, initial/final objective scores, monotone MGM histories and output menu
membership were checked; the five repeated cases match their original saved
assignments and costs exactly. These are restricted-menu local minima, not
global-optimum claims.

## All observed tail values

The user then requested keeping all values visited by each variable after the
split, including third values. Following the proposed tail definition, the new
menus include every value observed in the final 100 post-split updates. MGM is
started from every distinct complete assignment in that window. Seed 0's
best-state case consequently uses seven starts and preserves three values for
each of x21 and x28. This remains a finite observed-tail definition for its
non-period-two dynamics.

Seed 0's best-state result improves to **96,060.247**, compared with 96,122.246
from the earlier final-pair merge. The other five MGM outcomes are unchanged.
Both the candidate menus and starting assignments have expanded in this
comparison, so it does not isolate the contribution of third values alone.
Results and source snapshots are in `mgm_tail_values/` under the domain-20 run.
The new complete pipeline figures are `plots/pipeline_tail_mgm_seed_{0,1,2}.*`.

Twenty focused tests passed, including recovery of a third value omitted by the
final pair, equivalence with the old two-value MGM, and menu-local optimality.
The actual six cases were independently checked for complete tail-value and
starting-state coverage, original-cost scoring, and absence of improving
single-variable moves within their menus. Full CI still stops at the seven
pre-existing formatting failures. No BP rerun or additional B&B was performed.
