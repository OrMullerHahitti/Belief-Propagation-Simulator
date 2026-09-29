import Mathlib
import AamasProofs.Section4
import AamasProofs.AllOrderings

/-!
# Section 4 under the relaxed hypothesis `M_a` minimal and `M_b < B_a`

PAPER REFERENCE: `ors_revisions_shir.tex`, Section 4, Lemmas 4.4, 4.5, 4.7 and
Theorems 4.8, 4.9 (numbering as in `Section4.lean`).

`Section4.lean` proves these results under `M_a < M_b < B_a` and `M_b < B_b`.
This file re-proves them with the hypothesis `M_b < B_b` dropped, so `B_b` may lie
below `M_b` (the case "`X_i = a` is optimal against both values of `X_j`", which the
paper's footnote treats as immediate convergence).  The hypotheses used are

* `hMaMb : Ma < Mb`  (`M_a` is the minimal entry, `d > 0`),
* `hMbBa : Mb < Ba`  (no value of `X_j` is optimal against both values of `X_i`),
* `hMaBb : Ma < Bb`  (`M_a` is the minimal entry).

They imply `ClipCond`, so `eq:pass` holds and the flipping threshold exists
(`M_b − B_a < 0 < B_b − M_a`).  Where `Section4.lean` also assumed `B_b < B_a`
(the flipping-threshold change lemmas and Theorem 4.8), that assumption is dropped
as well.  Two statements are new: `flippingThreshold_spec`, the paper's definition
of the flipping threshold, and `upper_persistence_traj`, persistence of Theorem 4.9
along a whole trajectory.

STATUS: written 2026-09-27 without a Lean toolchain at hand; not yet compiled.
-/

namespace BpVerify.Section4

/-! ## The relaxed hypotheses imply the sum condition -/

lemma clipCond_of_relaxed (Ma Mb Ba Bb : ℝ) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) :
    ClipCond Ma Mb Ba Bb := by
  unfold ClipCond
  linarith

/-- `eq:pass` toward `X_j` without `M_b < B_b`. -/
lemma directToJ_eq_clip_relaxed (Ma Mb Ba Bb q : ℝ) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) :
    directToJ Ma Mb Ba Bb q = toJ Ma Mb Ba Bb q :=
  directToJ_eq_toJ_of_clipCond Ma Mb Ba Bb q (clipCond_of_relaxed Ma Mb Ba Bb hMbBa hMaBb)

/-- `eq:pass` toward `X_i` without `M_b < B_b`. -/
lemma directToI_eq_clip_relaxed (Ma Mb Ba Bb q : ℝ) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) :
    directToI Ma Mb Ba Bb q = toI Ma Mb Ba Bb q :=
  directToI_eq_toI_of_clipCond Ma Mb Ba Bb q (clipCond_of_relaxed Ma Mb Ba Bb hMbBa hMaBb)

/-- messages toward `X_j` stay between `M_b − B_a` and `B_b − M_a` under the sum condition. -/
lemma toJ_range_relaxed (Ma Mb Ba Bb q : ℝ) (hS : ClipCond Ma Mb Ba Bb) :
    lowerJ Mb Ba ≤ toJ Ma Mb Ba Bb q ∧ toJ Ma Mb Ba Bb q ≤ upperJ Ma Bb := by
  unfold ClipCond at hS
  simp only [toJ, clip, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-! ## Lemma 4.5 (constant value of `Δ_{\bar R_i}`) -/

/-- with `M_b < B_a` the two-sided limit is the paper's `min` form. -/
lemma upperLimit_eq_upperTarget (Ma Mb Ba Bb delta : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hdelta : 2 * (Ma - Mb) < delta) :
    upperLimit Ma Mb Ba Bb delta = upperTarget Ma Mb Ba Bb delta := by
  simp only [upperLimit, upperTarget, clip, d, lowerJ, upperJ, upperI, max_def, min_def]
  split_ifs <;> linarith

/-- with `M_a < B_b` the two-sided limit is the paper's `max` form. -/
lemma lowerLimit_eq_lowerTarget (Ma Mb Ba Bb delta : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hdelta : delta < 2 * (Ma - Mb)) :
    lowerLimit Ma Mb Ba Bb delta = lowerTarget Ma Mb Ba Bb delta := by
  simp only [lowerLimit, lowerTarget, clip, d, lowerJ, upperJ, lowerI, max_def, min_def]
  split_ifs <;> linarith

/-- Lemma 4.5, first case, without `M_b < B_b`. -/
theorem constant_input_converges_upper_relaxed (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hdelta : 2 * (Ma - Mb) < delta) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((pass Ma Mb Ba Bb delta)^[n]) r = upperTarget Ma Mb Ba Bb delta := by
  have hS := clipCond_of_relaxed Ma Mb Ba Bb hMbBa hMaBb
  have hc : 0 < 2 * (Mb - Ma) + delta := by linarith
  have hfun : directPass Ma Mb Ba Bb delta = pass Ma Mb Ba Bb delta :=
    funext (fun y => directPass_eq_pass Ma Mb Ba Bb delta y hS)
  obtain ⟨N, hN⟩ := directPass_converges_upper Ma Mb Ba Bb delta r hS hc
  rw [hfun, upperLimit_eq_upperTarget Ma Mb Ba Bb delta hMaMb hMbBa hMaBb hdelta] at hN
  exact ⟨N, hN⟩

/-- Lemma 4.5, second case, without `M_b < B_b`. -/
theorem constant_input_converges_lower_relaxed (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hdelta : delta < 2 * (Ma - Mb)) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((pass Ma Mb Ba Bb delta)^[n]) r = lowerTarget Ma Mb Ba Bb delta := by
  have hS := clipCond_of_relaxed Ma Mb Ba Bb hMbBa hMaBb
  have hc : 2 * (Mb - Ma) + delta < 0 := by linarith
  have hfun : directPass Ma Mb Ba Bb delta = pass Ma Mb Ba Bb delta :=
    funext (fun y => directPass_eq_pass Ma Mb Ba Bb delta y hS)
  obtain ⟨N, hN⟩ := directPass_converges_lower Ma Mb Ba Bb delta r hS hc
  rw [hfun, lowerLimit_eq_lowerTarget Ma Mb Ba Bb delta hMaMb hMbBa hMaBb hdelta] at hN
  exact ⟨N, hN⟩

/-- Lemma 4.5, boundary case `δ = 2(M_a − M_b)`, without `M_b < B_b`: one pass projects
    onto the neutral interval and a second pass changes nothing. -/
theorem neutral_pass_projects_and_fixes_relaxed (Ma Mb Ba Bb r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) :
    neutralLower Ma Mb Ba Bb ≤ pass Ma Mb Ba Bb (2 * (Ma - Mb)) r ∧
      pass Ma Mb Ba Bb (2 * (Ma - Mb)) r ≤ neutralUpper Ma Mb Ba Bb ∧
      pass Ma Mb Ba Bb (2 * (Ma - Mb)) (pass Ma Mb Ba Bb (2 * (Ma - Mb)) r) =
        pass Ma Mb Ba Bb (2 * (Ma - Mb)) r := by
  have hLU : neutralLower Ma Mb Ba Bb ≤ neutralUpper Ma Mb Ba Bb := by
    simp only [neutralLower, neutralUpper, lowerJ, upperJ, max_def, min_def]
    split_ifs <;> linarith
  have hclip : ∀ x, pass Ma Mb Ba Bb (2 * (Ma - Mb)) x =
      clip x (neutralLower Ma Mb Ba Bb) (neutralUpper Ma Mb Ba Bb) := by
    intro x
    simp only [neutralLower, neutralUpper, pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI,
      upperI, max_def, min_def]
    split_ifs <;> linarith
  rw [hclip r]
  have hrange := clip_range r (neutralLower Ma Mb Ba Bb) (neutralUpper Ma Mb Ba Bb) hLU
  refine ⟨hrange.1, hrange.2, ?_⟩
  rw [hclip]
  exact clip_idempotent r (neutralLower Ma Mb Ba Bb) (neutralUpper Ma Mb Ba Bb) hLU

/-! ## Lemma 4.4 (no outside input) without `M_b < B_b` -/

theorem no_input_converges_when_Bb_lt_Ba_relaxed (Ma Mb Ba Bb r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) (hBbBa : Bb < Ba) :
    ∃ N : ℕ, ∀ n, N ≤ n → ((pass Ma Mb Ba Bb 0)^[n]) r = upperJ Ma Bb := by
  have hconv := constant_input_converges_upper_relaxed Ma Mb Ba Bb 0 r hMaMb hMbBa hMaBb
    (by linarith)
  have ht : upperTarget Ma Mb Ba Bb 0 = upperJ Ma Bb := by
    simp only [upperTarget, upperJ, upperI, d, add_zero]
    rw [min_eq_left]
    linarith
  simpa [ht] using hconv

/-! ## Lemma 4.7 (flipping threshold) -/

/-- the paper's definition: the outside difference `x` makes the next message toward
    `X_j` vanish exactly when `x` equals the flipping threshold.  Needs
    `M_b − B_a < 0 < B_b − M_a`, i.e. `M_b < B_a` and `M_a < B_b`. -/
theorem flippingThreshold_spec (Ma Mb Ba Bb r x : ℝ) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) :
    toJ Ma Mb Ba Bb (x + toI Ma Mb Ba Bb r) = 0 ↔ x = flippingThreshold Ma Mb Ba Bb r := by
  simp only [flippingThreshold, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI, max_def,
    min_def]
  split_ifs <;> constructor <;> intro h <;> linarith

/-- `FT^1 = 2(M_a − M_b)` without `M_b < B_b`. -/
theorem initial_flippingThreshold_relaxed (Ma Mb Ba Bb : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) :
    flippingThreshold Ma Mb Ba Bb 0 = 2 * (Ma - Mb) := by
  simp only [flippingThreshold, toI, clip, d, lowerI, upperI, max_def, min_def]
  split_ifs <;> linarith

/-- Lemma 4.7, `δ > 2(M_a − M_b)`, without `M_b < B_b` and without `B_b < B_a`. -/
theorem flippingThreshold_upper_change_relaxed (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hdelta : 2 * (Ma - Mb) < delta)
    (hrL : lowerJ Mb Ba ≤ r) (hrU : r ≤ upperJ Ma Bb) :
    flippingThreshold Ma Mb Ba Bb r - delta - 2 * d Ma Mb ≤
        flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r) ∧
      flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r) ≤
        flippingThreshold Ma Mb Ba Bb r := by
  simp only [lowerJ, upperJ] at hrL hrU
  simp only [flippingThreshold, pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI,
    max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-- Lemma 4.7, `δ < 2(M_a − M_b)`, without `M_b < B_b` and without `B_b < B_a`. -/
theorem flippingThreshold_lower_change_relaxed (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hdelta : delta < 2 * (Ma - Mb))
    (hrL : lowerJ Mb Ba ≤ r) (hrU : r ≤ upperJ Ma Bb) :
    flippingThreshold Ma Mb Ba Bb r ≤
        flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r) ∧
      flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r) ≤
        flippingThreshold Ma Mb Ba Bb r - delta - 2 * d Ma Mb := by
  simp only [lowerJ, upperJ] at hrL hrU
  simp only [flippingThreshold, pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI,
    max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-! ## Theorem 4.8 (varying `Δ_{\bar R_i}`) -/

/-- one pass under the two pass conditions, for every message `r` (no seed needed). -/
lemma upper_progress_step_relaxed (Ma Mb Ba Bb delta eps r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hgrowth : eps ≤ delta + 2 * d Ma Mb)
    (hclip : Bb - Ba - d Ma Mb ≤ delta) :
    min (upperJ Ma Bb) (r + eps) ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤ upperJ Ma Bb := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI, max_def, min_def] at *
  split_ifs <;> constructor <;> linarith

lemma lower_progress_step_relaxed (Ma Mb Ba Bb delta eps r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (hgrowth : delta + 2 * d Ma Mb ≤ -eps)
    (hclip : delta ≤ Bb - Ba - d Ma Mb) :
    lowerJ Mb Ba ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤ max (lowerJ Mb Ba) (r - eps) := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI, max_def, min_def] at *
  split_ifs <;> constructor <;> linarith

/-- Theorem 4.8, first case, from any message `r0 ≤ B_b − M_a`: the chain equals the upper
    bounder from pass `n` on as soon as `n · ε ≥ (B_b − M_a) − r0`. -/
theorem varying_input_reaches_upper_relaxed (Ma Mb Ba Bb eps r0 : ℝ) (delta : ℕ → ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (heps : 0 < eps)
    (hgrowth : ∀ k, eps ≤ delta k + 2 * d Ma Mb)
    (hclip : ∀ k, Bb - Ba - d Ma Mb ≤ delta k)
    (hr0U : r0 ≤ upperJ Ma Bb) :
    ∀ n : ℕ, upperJ Ma Bb - r0 ≤ (n : ℝ) * eps →
      trajectory Ma Mb Ba Bb delta r0 n = upperJ Ma Bb := by
  have key : ∀ n : ℕ,
      (trajectory Ma Mb Ba Bb delta r0 n = upperJ Ma Bb ∨
        r0 + (n : ℝ) * eps ≤ trajectory Ma Mb Ba Bb delta r0 n) ∧
      trajectory Ma Mb Ba Bb delta r0 n ≤ upperJ Ma Bb := by
    intro n
    induction n with
    | zero =>
        constructor
        · right; simp
        · simpa using hr0U
    | succ k ih =>
        rw [trajectory_succ]
        rcases ih with ⟨hEq | hGrow, hUpper⟩
        · have hs := upper_progress_step_relaxed Ma Mb Ba Bb (delta k) eps (upperJ Ma Bb)
            hMaMb hMbBa hMaBb (hgrowth k) (hclip k)
          have hm : min (upperJ Ma Bb) (upperJ Ma Bb + eps) = upperJ Ma Bb :=
            min_eq_left (by linarith)
          rw [hm] at hs
          rw [hEq]
          exact ⟨Or.inl (le_antisymm hs.2 hs.1), hs.2⟩
        · have hs := upper_progress_step_relaxed Ma Mb Ba Bb (delta k) eps
            (trajectory Ma Mb Ba Bb delta r0 k) hMaMb hMbBa hMaBb (hgrowth k) (hclip k)
          constructor
          · by_cases hcap : upperJ Ma Bb ≤ trajectory Ma Mb Ba Bb delta r0 k + eps
            · left
              have hm : min (upperJ Ma Bb) (trajectory Ma Mb Ba Bb delta r0 k + eps) =
                  upperJ Ma Bb := min_eq_left hcap
              rw [hm] at hs
              exact le_antisymm hs.2 hs.1
            · right
              push_neg at hcap
              have hm : min (upperJ Ma Bb) (trajectory Ma Mb Ba Bb delta r0 k + eps) =
                  trajectory Ma Mb Ba Bb delta r0 k + eps := min_eq_right hcap.le
              rw [hm] at hs
              push_cast
              linarith
          · exact hs.2
  intro n hn
  rcases key n with ⟨hEq | hGrow, hUpper⟩
  · exact hEq
  · linarith

/-- Theorem 4.8, second case, from any message `r0 ≥ M_b − B_a`. -/
theorem varying_input_reaches_lower_relaxed (Ma Mb Ba Bb eps r0 : ℝ) (delta : ℕ → ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMaBb : Ma < Bb)
    (heps : 0 < eps)
    (hgrowth : ∀ k, delta k + 2 * d Ma Mb ≤ -eps)
    (hclip : ∀ k, delta k ≤ Bb - Ba - d Ma Mb)
    (hr0L : lowerJ Mb Ba ≤ r0) :
    ∀ n : ℕ, r0 - lowerJ Mb Ba ≤ (n : ℝ) * eps →
      trajectory Ma Mb Ba Bb delta r0 n = lowerJ Mb Ba := by
  have key : ∀ n : ℕ,
      (trajectory Ma Mb Ba Bb delta r0 n = lowerJ Mb Ba ∨
        trajectory Ma Mb Ba Bb delta r0 n ≤ r0 - (n : ℝ) * eps) ∧
      lowerJ Mb Ba ≤ trajectory Ma Mb Ba Bb delta r0 n := by
    intro n
    induction n with
    | zero =>
        constructor
        · right; simp
        · simpa using hr0L
    | succ k ih =>
        rw [trajectory_succ]
        rcases ih with ⟨hEq | hDrop, hLower⟩
        · have hs := lower_progress_step_relaxed Ma Mb Ba Bb (delta k) eps (lowerJ Mb Ba)
            hMaMb hMbBa hMaBb (hgrowth k) (hclip k)
          have hm : max (lowerJ Mb Ba) (lowerJ Mb Ba - eps) = lowerJ Mb Ba :=
            max_eq_left (by linarith)
          rw [hm] at hs
          rw [hEq]
          exact ⟨Or.inl (le_antisymm hs.2 hs.1), hs.1⟩
        · have hs := lower_progress_step_relaxed Ma Mb Ba Bb (delta k) eps
            (trajectory Ma Mb Ba Bb delta r0 k) hMaMb hMbBa hMaBb (hgrowth k) (hclip k)
          constructor
          · by_cases hcap : trajectory Ma Mb Ba Bb delta r0 k - eps ≤ lowerJ Mb Ba
            · left
              have hm : max (lowerJ Mb Ba) (trajectory Ma Mb Ba Bb delta r0 k - eps) =
                  lowerJ Mb Ba := max_eq_left hcap
              rw [hm] at hs
              exact le_antisymm hs.2 hs.1
            · right
              push_neg at hcap
              have hm : max (lowerJ Mb Ba) (trajectory Ma Mb Ba Bb delta r0 k - eps) =
                  trajectory Ma Mb Ba Bb delta r0 k - eps := max_eq_right hcap.le
              rw [hm] at hs
              push_cast
              linarith
          · exact hs.1
  intro n hn
  rcases key n with ⟨hEq | hDrop, hLower⟩
  · exact hEq
  · linarith

/-! ## Theorem 4.9 (persistence) -/

/-- the upper threshold needs only the strict sum condition `M_a + M_b < B_a + B_b`
    (with equality the two caps coincide and every `δ` keeps the message there). -/
theorem upper_persistence_iff_relaxed (Ma Mb Ba Bb delta : ℝ) (hS : Ma + Mb < Ba + Bb) :
    pass Ma Mb Ba Bb delta (upperJ Ma Bb) = upperJ Ma Bb ↔
      max (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) ≤ delta := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> intro h <;> linarith

/-- the lower threshold needs `M_b < B_a` (so that `d + (M_b − B_a) ≤ B_a − M_a`) and
    `M_a < B_b`; with equalities the caps coincide and the equivalence fails. -/
theorem lower_persistence_iff_relaxed (Ma Mb Ba Bb delta : ℝ) (hMbBa : Mb < Ba)
    (hMaBb : Ma < Bb) :
    pass Ma Mb Ba Bb delta (lowerJ Mb Ba) = lowerJ Mb Ba ↔
      delta ≤ min (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> intro h <;> linarith

/-- persistence along a trajectory: once a chain is at the upper bounder, it stays there
    as long as every later outside difference satisfies the threshold of Theorem 4.9. -/
theorem upper_persistence_traj (Ma Mb Ba Bb : ℝ) (delta : ℕ → ℝ) (r0 : ℝ)
    (hS : Ma + Mb < Ba + Bb) (N : ℕ)
    (hN : trajectory Ma Mb Ba Bb delta r0 N = upperJ Ma Bb)
    (hdelta : ∀ n, N ≤ n → max (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) ≤ delta n) :
    ∀ n, N ≤ n → trajectory Ma Mb Ba Bb delta r0 n = upperJ Ma Bb := by
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  clear hn
  induction k with
  | zero => simpa using hN
  | succ k ih =>
      rw [show N + (k + 1) = (N + k) + 1 by omega, trajectory_succ, ih]
      exact (upper_persistence_iff_relaxed Ma Mb Ba Bb (delta (N + k)) hS).mpr
        (hdelta (N + k) (Nat.le_add_right N k))

/-- the mirror image for the lower bounder. -/
theorem lower_persistence_traj (Ma Mb Ba Bb : ℝ) (delta : ℕ → ℝ) (r0 : ℝ)
    (hMbBa : Mb < Ba) (hMaBb : Ma < Bb) (N : ℕ)
    (hN : trajectory Ma Mb Ba Bb delta r0 N = lowerJ Mb Ba)
    (hdelta : ∀ n, N ≤ n → delta n ≤ min (-2 * d Ma Mb) (Bb - Ba - d Ma Mb)) :
    ∀ n, N ≤ n → trajectory Ma Mb Ba Bb delta r0 n = lowerJ Mb Ba := by
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  clear hn
  induction k with
  | zero => simpa using hN
  | succ k ih =>
      rw [show N + (k + 1) = (N + k) + 1 by omega, trajectory_succ, ih]
      exact (lower_persistence_iff_relaxed Ma Mb Ba Bb (delta (N + k)) hMbBa hMaBb).mpr
        (hdelta (N + k) (Nat.le_add_right N k))

end BpVerify.Section4
