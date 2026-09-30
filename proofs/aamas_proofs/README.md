# AAMAS 2027 Lean proofs

This directory contains the Lean 4 formalization of the binary split-cycle
analysis in Section 4 of the AAMAS 2027 paper.

`AamasProofs/Section4.lean` uses the paper's `Ma`, `Mb`, `Ba`, and `Bb` convention and
contains machine-checked proofs of:

- the two direct min-sum clipping formulas and the one-pass recurrence;
- the minimization claim in Lemma 4.3;
- convergence with no outside input for both bounder orderings (Lemma 4.4 and
  its following alternative case);
- the exact limits for a constant value of `Delta_(bar R_i)`, including the
  neutral boundary case (Lemma 4.5);
- the initial flipping threshold and its change in both directions
  (Lemma 4.7);
- finite-time stabilization of the assignment at `X_j` under a uniform sign
  margin for varying inputs, followed by finite-time arrival at the upper or
  lower message bound under the additional bounder condition, with the paper's
  quantitative pass bounds (Theorem 4.8); and
- the necessary-and-sufficient persistence thresholds, including their
  equivalent min/max forms (Theorem 4.9).

The assignment part of Theorem 4.8 proves eventual strict positivity or strict
negativity of every cycle-aligned message stream.  Its stronger bound-arrival
part uses exact eventual equality, matching the paper's definition of message
convergence: after finitely many complete passes, the message difference reaches
the stated value and remains there.

From the repository's `proofs` directory, verify the complete library with:

```sh
lake build AamasProofs
```

The file contains no `sorry` or `admit` declarations.

## Every ordering of the table entries (added 2026-09-27, not yet compiled)

`AamasProofs/AllOrderings.lean` starts from the direct min-sum maps and assumes only
that `M_a` is the minimal entry:

- every message difference lies between the two caps of its direction, for every
  table (`directToJ_range`, `directToI_range`);
- the clip form of the one-pass recurrence holds exactly when
  `M_a + M_b <= B_a + B_b` (`directToJ_eq_toJ_of_clipCond`,
  `exists_directToJ_ne_toJ_of_not_clipCond`);
- under that condition and a constant `Delta_(bar R_i)` with `2d + delta != 0`,
  every chain becomes constant at the two-sided clip limit
  (`directPass_converges_upper`, `directPass_converges_lower`);
- when `B_a <= M_b` the chain is constant from the second pass on
  (`dominant_constant_from_two`);
- for every ordering, the chain becomes constant and a cap is active at the limit
  (`all_orderings`).

`AamasProofs/Section4Relaxed.lean` re-proves Lemmas 4.4, 4.5, 4.7 and Theorems 4.8,
4.9 with the hypothesis `M_b < B_b` dropped (only `M_a < M_b`, `M_b < B_a`,
`M_a < B_b`), drops `B_b < B_a` from the flipping-threshold change lemmas and
Theorem 4.8, and adds the paper's definition of the flipping threshold
(`flippingThreshold_spec`) and persistence along a trajectory
(`upper_persistence_traj`, `lower_persistence_traj`).

Both files were written without a Lean toolchain at hand; their statements were
checked by exact rational simulation, the proofs have not been compiled yet.
