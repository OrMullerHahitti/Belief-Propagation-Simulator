# Structural splitting and damping results

This follow-up separates three questions: why messages oscillate, when they
converge with strict decoding, and whether their final original cost improves.
The results below do not give a universal cost-improving intervention rule.

## Proved mechanisms and limits

- On a bipartite pairwise graph, the undamped message map has the form
  `G(u,v)=(g(v),h(u))`. Any two fixed points generate an exact hybrid two-cycle;
  if both parent fixed points are locally asymptotically stable, that cycle is
  attracting. The existing three-variable path cycle is exactly the hybrid of
  its cost-13 and cost-16 stable message solutions.
- Damping can also remove oscillation without any active-row change. A single
  split binary edge has an open set of full-clone initial states whose
  undamped assignments alternate at cost 16. Damping converges to a fixed set
  with strict decoded assignments of cost zero. The alternating component
  decays by `2*lambda-1`; the component along the fixed set is preserved.
- For any finite label set with strict conditional minimizers and unique
  decoded beliefs, damping cannot create local asymptotic stability of an
  individual fixed point. Referencing messages to the decoded labels makes
  its Jacobian nonnegative and integer-valued. This statement deliberately
  excludes convergence toward a fixed set, as in the preceding example.
- The exact consistent-commitment criterion is
  `Delta_i(a) > max(w_e,1-w_e) * H_ij(a)`, where Delta is an original local
  cost gap and H is the worst interaction cross-difference. Equal splitting
  maximizes this certificate. Losing it does not prove basin escape: a
  four-cycle retains an exact partly committed suboptimal fixed point at .95.

Proofs and assumptions are in [THEORY_CANDIDATES.md](THEORY_CANDIDATES.md),
[BINARY_ACTIVE_STRUCTURE.md](BINARY_ACTIVE_STRUCTURE.md), and
[BEST_LABEL_GAUGE.md](BEST_LABEL_GAUGE.md), with target-feasibility limits in
[OBSTRUCTIONS.md](OBSTRUCTIONS.md). Conditional minimizing-row ties,
isolated-point attraction, fixed-set convergence, and decoded-label ties must
not be conflated.

## Connection to the actual paper instances

The exact full-clone, full-domain checker replayed eight 2,000-update benchmark
trajectories and matched every saved assignment and original cost. Six
successful endpoints have nilpotent active Jacobians. Exact dyadic arithmetic
constructs their branch fixed points, verifies all strict selector inequalities,
and bounds the saved states inside a basin that stays in the same active cell.
For the mathematical recurrence, pairwise Q reaches its target in three or
four undamped updates. The two known failing tails retain an eigenvalue one;
scalar damping does not make that active branch strictly contractive.

There is a separate numerical limitation. Releasing damping in the existing
raw-message kernel produces large additive offsets. On sparse seed 6003 with
the pulse, a 16-update release raises the cost after offsets grow to about
`3.1e18`. Normalizing the same kernel before every update preserves the cost
and assignments; direct updates in message differences do likewise. This is
evidence about the experiment kernel's arithmetic, not a verified change to
the core engine. No core runtime was modified.

In the decoded-label reference, every one of the six exact Jacobians is a
0/1 DAG. Signed cycles in the original reference were coordinate artifacts.

The reproducible state, exact certificates, and numerical controls are under
`results/damping_generalization_20260915/local_stability/`. The previous
32-seed cost comparison remains in
`results/aaai_derived_control_20260915/paper_confirmation/`; the new endpoint
analysis does not establish why the pulse chooses a cheaper basin.

## Verification

The following combined focused run passed 105 tests; the new Python files also
pass Black and flake8:

```sh
uv run --no-sync pytest tests/test_damping_structural_theory.py \
  tests/test_damping_local_stability.py tests/test_decoded_gauge_check.py \
  tests/test_damping_obstructions.py \
  tests/test_damping_causality_theory.py tests/test_damping_causality_feedback.py \
  tests/test_damping_causality_native.py -q
```

`make ci` was also run. It stops at seven existing formatting failures outside
these research changes: `test_splitting.py`, `test_bp_engine.py`,
`test_engines.py`, `conftest.py`, `integrations/dabp/build.py`,
`snapshots/analyzer.py`, and `snapshots/visualizer.py`. Unrelated work is
preserved. The mathematical proofs and experiment-local checks are not a claim
that repository-wide CI passes.

## Independent consultation

A completed consultation and one focused follow-up in the user's `ors research`
project used the visibly selected `6 Pro` model. The chat was then switched
back to **Instant**, as requested. [PRO_CONSULTATION.md](PRO_CONSULTATION.md)
records the advice, independent checks, and correction for the actual first
Q update using retained old R.

The strongest new constructive result is an exact pulse theorem for a cycle
family. On the biased four-cycle, the critical heavier weight is
`w*=1-b/(8C)`, where C is the disagreement-equivalent edge penalty and b the
bias, with `0<b<C/2`. For C=4 and b=1 this is **31/32**. A designated heavy
clone that never exceeds that value cannot escape the bad baseline state
under any allowed damping schedule. Holding a weight above the threshold
long enough selects the good state and permits a certified warm return to .5.
See [PRO_THRESHOLD_AUDIT.md](PRO_THRESHOLD_AUDIT.md) for the exact assumptions
and proof. This does not assert a cold-start advantage over a strong fixed
split on the existing 32-seed benchmark.

The next proposed check is a four-cycle with one competing chord, exact zero
initialization, and fixed damping .9. Test a `.5 -> 63/64 -> .5` intervention
with the corrected capture condition against .5/.9 and near-one fixed splits,
after validating numerical gauge and retained-R switching behavior. The
competing-chord experiment is a plan; it has not been run in this consultation.
