# AAAI-derived split control, September 15

The September 14 tiny-graph comparison at 128 iterations is exploratory.
Its apparent fixed-damping advantage reverses at 2000 iterations. It must not
be presented as an improvement over the paper's usual 0.5 split / 0.9 damping.

## Mechanism and source

Use the complete active-minimizer inequalities from the splitting analysis.
For a sender label u and competitor z, the minimizing row changes at
alpha = (Q[z] - Q[u]) / (C[u,v] - C[z,v]). Only effective lower-envelope
crossings count. Complementary clones use alpha=w and alpha=1-w. An action
preserves Q/R history and the original objective exactly.

The September 9 saved audit identifies sub2.tex as the then-current Overleaf
main. Local publish/ is a distinct draft. Its repaired committed-two-cycle
results are a possible later derivation, not a verified submitted theorem.
See results/aaai_derived_control_20260915/theory_review.md for assumptions.

## Development experiment

- Small networks: K4 and bowtie, with random integer tables and unary
  preferences, plus a binary frustrated family. Keep families separate.
- Development seeds 5000--5031. Validation seeds 5100--5131. Fresh confirmation
  seeds 6000--6063 are reserved until a candidate and all settings are fixed.
- Primary horizon: 2000 actual updates. The 128 and 512 values are diagnostics.
- Primary outcome: actual terminal original cost including unary preferences.
- Stability: assignment constancy over the final 100 updates and gauge-invariant
  Q/R residual below 1e-7 times summed factor ranges over the final 100 updates.
  These are finite-window observations, not convergence theorems.
- Usual baseline: scalar split .5 with old-Q damping .9 throughout.
- Fixed controls: split .5/.65/.8/.95 and damping .5/.7/.9/.95, selected only on
  validation data, with stability no worse than the usual baseline.
- Adaptive candidate: sparse interventions chosen from effective split
  boundaries and evaluated by short continuations on the original objective.
  This is a model-based controller; no claim of neural or online-learning gain.
- Attribution controls: same continuation budget with a fixed split grid or
  random split actions; hold-current-action continuation always included.
- Count simulated continuation steps and wall time. Give the usual baseline
  an equally large update budget before claiming a computational advantage.
- Confirm native-paper parity, including split unary factors, startup, raw
  argmin, periodic normalization, and uninterrupted full-horizon execution.

Selection changes, failures, and additional experiments must be recorded.
Confirmation will only follow a promising development result; inspecting
confirmation results makes those seeds ineligible for further tuning.

## Recorded development changes and final confirmation

1. Active-row midpoint proposals failed in the initial small pilot. Completing
   the one-step decoded-region search and adding explicit hold-relative cost,
   assignment-flip and undamped-defect guards did not produce a consistent
   tiny-graph gain with either 64-step or 256-step planning continuations.
2. A separate fixed schedule changed all pairwise splits to .95 before step
   64, then restored .5 before step 256, with .9 damping throughout. It was
   tested without subsequent timing/weight tuning on the actual sparse/dense
   paper builders: development seeds 5000--5001, then validation 5100--5115.
3. Fresh validation improved mean cost in both paper families. Sparse stability
   counts matched baseline; dense stability declined and fixed .95 had lower
   mean cost. The next confirmation was therefore restricted to sparse inputs.
4. `results/aaai_derived_control_20260915/paper_confirmation/PROTOCOL.md` was
   written before any reserved-seed execution. It fixed sparse seeds 6000--6031,
   the unchanged pulse, ordinary baseline and development-selected fixed .8
   control, and both 2,000/10,000 checkpoints. Its confirmation is complete.
   All seeds were included and no settings were tuned using those outcomes.
5. A further small development check used K4/domain10, seeds 5000--5031,
   unchanged pulse, exact optima for evaluation only. Cost uncertainty included
   zero and stability declined, so that track stopped without confirmation.

The winning schedule is a deterministic empirical splitting intervention. It
does not establish an advantage from attention, neural learning, or online
parameter updates. Pattern-interval derivations are a separate theoretical
result and do not supply a cost-improvement theorem for the pulse.
