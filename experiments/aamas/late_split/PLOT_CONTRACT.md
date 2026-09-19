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
