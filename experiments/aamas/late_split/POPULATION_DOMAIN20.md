# Domain-20 random-dense population study

User authorization: 2026-09-21, expand the existing all-tail-values MGM experiment
to 50 seeds, average cost over iterations, and retain evidence for partial reruns.

## Exact experiment

- Seeds **0–49**, one random-dense input per seed: **50 agents**, **20 values**,
  edge probability **0.6**, pairwise integer costs in `[100, 200)`, and the
  historical per-agent unary preference scale **0.01**. Retain the original
  input tables and ordered factor axes; evaluate every method on those tables.
- **DMS control:** unsplit native damping 0.9, 2,000 updates.
- **Standard baseline:** native equal factor splitting from initialization,
  damping 0.9 throughout, 2,000 updates (`DMS_split_0.5`).
- **Fixed-time late split:** first 1,000 unsplit DMS updates; equally split all
  factors and transfer their existing R messages equally; remove damping;
  run 1,000 further BP updates.
- **Best-checkpoint late split:** select the earliest minimum-original-cost
  full message checkpoint within those same first 1,000 DMS updates; restore
  it, apply the same split and transfer, remove damping, and run 1,000 updates.
  Preserve the checkpoint's native iteration index and normalization phase.
- **MGM:** after each complete post-split phase, use every value each variable
  took in its **final 100 assignments**. Run deterministic synchronous MGM-1
  from every distinct joint assignment in that window; retain every run and
  report the one with minimum final original cost. Stop at no positive local
  gain; cap at 10,000 improving rounds and reject cap-hit runs as incomplete.
  These are observed-tail menus; an unsettled tail is not evidence that an
  entire longer cycle was captured. **No new B&B runs.**
- Execute all fixed-time BP cases, all best-checkpoint BP cases, then MGM.
  Six local worker processes. Native message normalization and cost-axis fixes
  remain unchanged. Each baseline/prefix/post-split cost is independently
  reconstructed from its saved assignment and original tables.

Seeds 0–2 reuse the existing domain-20 pilot's exact inputs, traces, split
checkpoints, and all-tail-values MGM. The new manifest records the pilot path
and source hashes. Original algorithm/generator dependencies and all imported
BP evidence hashes are checked before import. Legacy two-value MGM/B&B fields
inside imported BP JSON are historical only; population outcomes always come
from `mgm_tail_values/`.

## Commands

From the repository root, the approved run and its idempotent resume command are:

```bash
.venv/bin/python -u -m experiments.aamas.late_split.population \
  --out experiments/aamas/runs/late_split_domain20_50seeds_20260921 \
  --pilot experiments/aamas/runs/late_split_domain20_20260921 \
  --workers 6
```

Use `--phase fixed`, `--phase best`, or `--phase mgm` to visit only that phase.
Completed seed/phase results are checksum-verified and skipped. Missing MGM
results need only the saved input and post-split assignment trace; neither BP
nor its baselines are rerun. Do not overwrite old results to test a changed
protocol; write a new analysis directory referencing the retained inputs.

Regenerate averages and plots without simulations:

```bash
.venv/bin/python -m experiments.aamas.late_split.plot_population \
  experiments/aamas/runs/late_split_domain20_50seeds_20260921
```

Simulation sources are frozen under the run's `source/`; the renderer is frozen
under `plots/`. Resume rejects altered simulation sources/configuration. A plot
edit does not invalidate BP evidence. The working tree, not only the Git commit,
defines the experiment: consult `manifest.json` for exact source hashes.

## Saved evidence and partial reruns

| Files | What can be investigated or rerun |
| --- | --- |
| `random_dense_<seed>/input.npz` | Exact original graph/tables, no random regeneration required |
| `DMS_trace.npz`, `DMS_split_0.5_trace.npz` | Both controls' cost and all 50 assignments at every update |
| `prefix.npz`, `prefix.json` | All prefix costs/assignments and earliest-best selection |
| `fixed_checkpoint.json.gz`, `best_checkpoint.json.gz` | Full unsplit messages, retained damping history, monitor, normalization phase; rerun only a continuation |
| `fixed_trace.npz`, `best_trace.npz` | Every post-split cost/assignment plus native indices; classify tails, select different menus, or rerun MGM only |
| `checkpoints/<mode>/<count>/` | New seeds: runtime graph, full message state and accumulated trace at every 250 post-split updates and at termination |
| `checkpoints/<baseline>/terminal/` | New seeds: full terminal baseline state for extending a control |
| `mgm_tail_values/seed_<seed>_<mode>_mgm.json` | Every menu and start; each MGM cost history, moves/round, initial/final assignment, selected winner and round count |
| `aligned_costs.npz`, `mean_costs.csv`, `per_seed_summary.csv` | Recompute means, paired comparisons, and alternative plots without simulations |

Runtime checkpoints have checksummed completion records written last. An
interrupted continuation resumes from its latest complete checkpoint, losing
at most 249 updates. A baseline interrupted before completion reruns only that
baseline. Imported seeds 0–2 have the original split checkpoints and complete
traces, but no full terminal BP state: extending those requires replaying the
1,000-update continuation from its saved split checkpoint, not regenerating
the problem, baselines or prefix. Reconstructing intermediate MGM assignments
requires rerunning only MGM from its retained start/menu; per-round costs and
move counts are already saved.

## Plot interpretation

Produce separate fixed-time and best-checkpoint mean line plots, each with the
standard damped-splitting baseline and a separate MGM close-up. The x-axis
counts all 1,000 prefix observation updates, then 1,000 continuation updates,
then MGM rounds. Thus best-checkpoint restoration appears at x=1,000 even when
its native checkpoint was earlier. This counts observation work but is not a
CPU-time comparison; splitting changes per-update work, and the MGM line shows
the winning start rather than cumulative work across all starts.

Every plotted mean uses all 50 seeds. After MGM finishes for a seed, its cost
is held at its terminal value until the longest winning MGM trajectory ends.
The standard baseline is executed for exactly 2,000 updates; any extension
afterward is dotted and holds its final cost. A dotted vertical segment at the
MGM boundary denotes initialization from a selected tail state, not an MGM move.
Retain native traces separately. A flat population mean does not establish
per-seed convergence; classify the saved assignment tails separately.
