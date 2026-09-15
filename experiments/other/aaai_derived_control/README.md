# Splitting controls derived from the AAAI investigation

**Confirmed result:** on 32 previously unused 50-variable sparse AAAI graphs,
the temporary split pulse reduces mean terminal cost from 14263.065 to
14180.288 (0.58%), with 25 wins, 7 ties and no losses. These endpoints are
unchanged at 10,000 versus 2,000 updates. Both methods pass strict finite-tail
stability on 31/32 inputs, but their failing seeds differ. See the complete
`paper_confirmation/README.md` under the evidence directory below. This is a
benchmark-specific result; the tiny K4 experiments remain inconclusive.

This experiment tests whether changing the cost decomposition during a run
improves the actual terminal objective while retaining stable message dynamics.
It uses no DABP code or learned neural parameters. The original 145-parameter
adaptive controller did not improve on ordinary .5 splitting / .9 damping at
2,000 updates; its apparent 128-step benefit was horizon-specific.

The current candidate is a temporary asymmetric split of every pairwise factor:

| Zero-based update indices | First clone weight | Second clone weight | Old-Q damping |
| --- | ---: | ---: | ---: |
| 0--63 | .5 | .5 | .9 |
| 64--255 | .95 | .05 | .9 |
| 256 onward | .5 | .5 | .9 |

All existing messages are retained at both changes. Unary factors remain
equally split. The two pairwise clone tables always sum to the original table.
The final cost is evaluated on the original objective, including unary costs.

## Why this follows the splitting investigation

The complete minimizing-row inequalities determine when split changes alter
message dynamics. Small complementary perturbations can cancel while the
same rows remain active. A sufficiently large temporary perturbation can
change the eventual assignment; restoring equal splitting permits a direct
comparison with the usual final representation. This is a mechanism-motivated
empirical policy. The .95 weight and the two intervention times are fixed
experimental choices, not values established by a cost-improvement theorem.

`code/committed.py` supplies a separate theoretical tool: for a prescribed fully
committed pattern, it computes the strict split-weight interval where its fixed
point exists. Those boundaries can differ from immediate-message boundaries.
Leaving that interval rules out that pattern, but does not guarantee a better
cost or convergence. The pulse policy does not use this helper to choose its
times or weights.

The September 9 source audit identified `sub2.tex` as the then-current Overleaf
main; local `publish/` is a distinct draft. We do not attribute later local
theorems to the submitted version. The derivation and source qualifications
are recorded under `results/aaai_derived_control_20260915/`.

## Native usage

From the repository's existing uv environment:

```python
from experiments.aaai.code.engines import (
    CostOnlySnapshotManager,
    run_full_horizon,
)
from experiments.aaai.code.problems import build_random_sparse
from experiments.other.aaai_derived_control.code.pulse import SplitPulseEngine

engine = SplitPulseEngine(
    build_random_sparse(seed=42),
    snapshot_manager=CostOnlySnapshotManager(),
)
costs = run_full_horizon(engine, 2000)
print(costs[-1])
```

Use the full-horizon helper: ordinary early stopping can omit an intervention
or its restoration. This engine is experiment-local; no public PropFlow API or
default algorithm has changed. The vectorized `code/kernel.py` supports fast
reproduction and has been checked against the actual native update path.

## Evidence and unsuccessful approaches

- `paper_validation/README.md`: unchanged pulse on fresh 50-variable paper
  inputs, 16 sparse and 16 dense seeds, 2,000 updates. Mean cost improved in both
  families. Strict stability was 15/16 for pulse and baseline on sparse inputs,
  and 15/16 versus 16/16 on dense inputs; the sparse failing seeds differ.
- `paper_confirmation/PROTOCOL.md`: separate prespecified sparse confirmation,
  seeds 6000--6031, unchanged pulse and fixed .8 control, readouts at 2,000 and
  10,000 updates. `paper_confirmation/README.md` reports the confirmed cost
  reduction, the stronger fixed comparison, and exact stability limitations.
- `experiment_audit.md`: reproduced and extended the old controller's saved
  trajectories. At 2,000 updates both its final cost and stability were worse
  than the usual baseline; online updates added no meaningful benefit.
- `pilot_midpoints`, `pilot_guarded64`, `pilot_guarded256`: exact active-row and
  decoding-region proposals, with short planning continuations, did not show
  a consistent improvement on the tiny graphs. These are development results.
- `fixed_development/RESULTS.md`: 512 small fixed-setting runs. Some settings
  improve particular K4 cases; the random bowtie inputs were already solved.
- `tiny_d10_development/README.md`: 32 four-variable K4 graphs with ten values
  per variable. The pulse reduces sample mean cost, but its interval includes
  zero and strict stability falls from 29/32 to 28/32. No confirmation claim.
- `native_bridge`, `policy_native_replay`, `confirmation_native_replay`: native
  message/assignment checks and independent original-cost reconstruction.

All evidence paths above are relative to
`results/aaai_derived_control_20260915/`. Source copies, saved inputs, seeds,
trajectories and verification hashes are kept with each run. No confirmation
seeds are used for tuning. Finite assignment and message stability are measured
separately from cost; neither is an asymptotic convergence proof.

## Tests

```sh
uv run --no-sync python -m pytest tests/test_aaai_derived_kernel.py tests/test_aaai_derived_control.py tests/test_aaai_committed_control.py tests/test_aaai_split_pulse.py -q
```

All 39 new tests pass. They cover native arithmetic and pulse replay, exact
rational boundary checks, unary behavior, objective preservation, and
committed-pattern intervals. The full suite has 321 passed, 3 existing Figure
5/8 reproduction failures, and 2 skipped; `make ci` stops on seven existing
formatting failures. See `pytest_full.log` in the evidence directory. These
unrelated failures were preserved rather than included in this research change.
