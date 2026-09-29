# Restore damping after the best-checkpoint split

Approved September 21, 2026: use the same random-dense seeds 0–49, 50 agents,
domain size 20, density 0.6. Reuse each best-checkpoint experiment through its
1,000 undamped post-split updates. At the point where MGM starts, continue the
same split graph with old-Q damping 0.9 for another 1,000 BP updates.

The parent experiment is `../runs/late_split_domain20_50seeds_20260921/`.
The new results belong in `../runs/best_split_restore_damping_50seeds_20260921/`.
No fixed-time variant, new MGM search, or B&B is included.

## Exact handoff and partial replay

The undamped engine retains R messages but does not archive Q messages. Damping
must blend the newly computed Q with the **last actually emitted undamped Q**,
not an empty history or the pre-split history. Recover that Q from native Step
messages while replaying the smallest available suffix: 250 updates for seeds
3–49; 1,000 post-split updates for the older pilot seeds 0–2. Check every replayed
assignment and cost against the parent trace, and the complete terminal message
state against its saved checkpoint where available. This replay is recovery of
existing evidence and is not counted again on the plot axis.

Keep all terminal R messages, factor tables, graph order, original-cost scoring,
native iteration indices, and normalization phase. Install the recovered Q as
the previous-message history and use the native DampingEngine. The first new
update already uses `Q_sent = 0.9 * Q_previous + 0.1 * Q_computed`. Save that
initial state and complete runtime checkpoints every 250 new updates, plus every
cost and assignment. Resuming reuses completed seeds and checkpoints.

```bash
uv run --no-sync python -m experiments.aamas.damping_after_split.run \
  --parent experiments/aamas/runs/late_split_domain20_50seeds_20260921 \
  --out experiments/aamas/runs/best_split_restore_damping_50seeds_20260921 \
  --workers 6
uv run --no-sync python -m experiments.aamas.damping_after_split.plot \
  --run-dir experiments/aamas/runs/best_split_restore_damping_50seeds_20260921
```

## Plot contract

Question: after the best-checkpoint split has run undamped for 1,000 updates,
how does restoring damping compare with the existing MGM result and the normal
split-and-damping baseline? Use standalone PNG and vector PDF line plots of
original-objective cost against updates, averaged equally over all 50 seeds
at every point, without smoothing or best-so-far replacement.

The work axis is 1,000 observation updates, then 1,000 undamped split updates,
then 1,000 damped BP updates (or the existing winning-start MGM rounds). The
best checkpoint differs by seed; all 1,000 observations count before restoring
that checkpoint. Event labels appear only on the x axis. Use orange for the
existing best-split/MGM route, blue for restored damping, gray for the normal
baseline; top and right plot borders are removed. Produce a full trajectory
and a separate handoff close-up. Retain all plotted numeric values.

The baseline was measured only through update 2,000; a dotted continuation holds
its last measured cost. Finished MGM trajectories are also held at their final
cost (dotted after every seed finishes). Dotted vertical segments show restored
or selected states. This compares cost trajectories, not equal CPU time or total
work across MGM starts. No earlier incumbent is substituted for a final cost.
