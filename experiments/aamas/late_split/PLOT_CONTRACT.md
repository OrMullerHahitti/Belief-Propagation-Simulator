# Pilot figure contract

Question: does splitting later and releasing damping produce two alternating
assignments, and do their menu merges improve on both branches and on the
incumbent found before splitting?

Deliver two standalone research figures as vector PDF and preview PNG, using
Matplotlib and the experiment frame-removal helper. No population inference
or statistical-significance claim is supported by this three-seed pilot.

1. Cost trajectories: one row per seed, fixed-time and best-checkpoint columns.
   Plot the 1,000-update selection prefix and the 1,000-update continuation
   separately; show the unchanged DMS reference and the prefix incumbent.
   The x-axis counts search work plus continuation work. In the best-checkpoint
   column, annotate the restored original checkpoint index; do not pretend
   discarded search work was free. Native update indices remain in the data.
2. Tail and merges: same panels, last 50 post-split updates, with costs of the
   alternating assignments and horizontal MGM/B&B/prefix-incumbent references.
   State the measured 100-assignment tail classification and B&B completion.
   Equal branch costs do not imply equal assignments; classification uses the
   actual saved assignments.

Use linear original-objective cost on y, per-instance panels (no averaging),
white background, blue for trajectories and orange for MGM, neutral black/grey
for B&B and controls. Use markers, line styles and labels as well as color.
No decorative branding. Inspect the exported previews for clipping, legend
collisions and honest axis ranges. Source data are the hash-verified case
artifacts and retained baseline slices in the supplied run directory.
# Domain-size comparison requested on 2026-09-21

## Additional complete-pipeline plots

Keep the existing figures and add one line figure per seed showing the original
native BP trajectory followed by the saved MGM history. Keep the standard
damped-splitting baseline. Mark split and MGM-start locations only on the
x-axis; preserve blue for fixed-time and orange dashed for best-checkpoint.
MGM starts after 1,000 post-split BP updates. Plot the better of the two MGM
starts, mark its terminal point with a circle, and show any restart from the
penultimate assignment as a dotted initialization segment at the same x value.
After each BP trajectory's endpoint, one x-axis unit is one MGM round. This
is a stage trajectory, not a wall-time or total-search-work comparison.
Use the saved two-value-menu results, including the explicitly sampled pair
for seed 0's non-period-two best-state tail. Do not substitute an all-tail-values
MGM experiment. Validate source hashes, original costs, initial/final MGM scores
and round counts. Export PNG/PDF under the existing run's `plots/` directory
with distinct `pipeline_mgm_seed_*` names, and inspect each PNG.

## Approved 50-seed domain-20 population study (2026-09-21)

Question: does the same late-split and all-tail-values MGM pipeline improve mean
original cost over the standard equal-split, damping-0.9 baseline on seeds 0–49?
Keep fixed-time and best-checkpoint methods in separate figures. Each figure
contains one mean trajectory and the matched standard baseline; show a full
trajectory and a separate close-up around the MGM transition. No smoothing.
All averages include all 50 seeds at every plotted point.

The common x-axis counts 1,000 observation updates, then 1,000 undamped
continuation updates, then MGM rounds. For best-checkpoint selection the entire
observation window is counted before restoring the earlier state. Original
native update indices remain in the raw traces. Show split/MGM events only as
x-axis labels. Show a dotted, zero-round initialization segment when MGM starts
from a selected tail assignment. As in the pilot, the MGM curve follows the
start with the best final outcome among all distinct tail starts; this is a
multistart result, not one prespecified start or total work across starts.

Hold each completed MGM trajectory at its terminal cost when aligning differing
round counts. Baselines stop after 2,000 BP updates; dotted extensions repeat
their terminal value, without claiming extra BP execution. Export mean arrays,
per-seed aligned arrays, per-seed summaries, renderer and checksums alongside
PNG and vector PDF in the run directory. Validate original costs, all tail
menus, MGM starts/endpoints, monotonic MGM rounds and common sample counts;
inspect every exported PNG.

The user's subsequent all-values extension is shown in separate
`pipeline_tail_mgm_seed_*` figures. Use the union of each variable's values in
the final 100 post-split updates, with all distinct tail assignments as MGM
starts. Plot the best result of those starts. Keep the original two-value
figures; the additional dotted boundary initialization may now restore any
observed tail assignment rather than only the penultimate one.

Use one separate line figure per dense seed, showing original cost against
native iterations for standard damped splitting and both late-split methods.
Use neutral solid, blue solid and orange dashed lines; label split points only
on x-axis ticks. Read the newly generated, hash-verified domain-20 baselines
and the full stored traces (2,000 control updates, 1,000 post-split updates).
Plot observed trajectories without averaging or smoothing. Export PNG and
vector PDF under the run's `plots/` directory and inspect every PNG. Record
checkpoint-selection work and post-run merges separately from the plotted axis.
