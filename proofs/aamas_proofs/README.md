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
- finite-time arrival at the upper and lower bounds under varying inputs,
  with the paper's quantitative pass bounds (Theorem 4.8); and
- the necessary-and-sufficient persistence thresholds, including their
  equivalent min/max forms (Theorem 4.9).

The finite-time results use exact eventual equality, matching the paper's
definition of convergence: after finitely many complete passes, the message
difference reaches the stated value and remains there.

From the repository's `proofs` directory, verify the complete library with:

```sh
lake build AamasProofs
```

The file contains no `sorry` or `admit` declarations.
