import Mathlib
import AamasProofs.Section4

/-!
# The split cycle for every ordering of the table entries

PAPER REFERENCE: `ors_revisions_shir.tex`, Section 4, the standing assumption
`M_a < M_b < B_b < B_a` (line 273) with its footnote, and the one-pass
recurrence `eq:pass` in the appendix.

`Section4.lean` works with the clipped-translation maps `toJ`, `toI`, `pass`
and connects them to the direct min-sum maps `directToJ`, `directToI` under
`M_a < M_b < B_a` and `M_b < B_b`.  This file starts from the direct maps and
assumes only that `M_a` is the minimal entry (`IsMinAA`), so every ordering of
`M_b`, `B_a`, `B_b` is covered:

* `directToJ_range`: a message difference always lies between the two caps of
  its direction, for every table.
* `directToJ_eq_toJ_of_clipCond`, `exists_directToJ_ne_toJ_of_not_clipCond`:
  the clip form of `eq:pass` holds exactly when `M_a + M_b ≤ B_a + B_b`.
* `directPass_converges_upper`, `directPass_converges_lower`: under that sum
  condition and a constant outside difference `δ` with `2d + δ ≠ 0`, every
  chain of messages toward `X_j` becomes constant after finitely many complete
  passes, at the two-sided clip `[hi + d + δ]_L^U` or `[lo + d + δ]_L^U`
  (`upperLimit`, `lowerLimit`).  When `M_b < B_a` these are the `min`/`max`
  limits of Lemma 4.5 (see `Section4Relaxed.lean`).
* `dominant_constant_from_two`: when `B_a ≤ M_b`, the value `a` of `X_j` is
  optimal against both values of `X_i`, and the chain is constant from the
  second pass on, for every `δ`.
* `all_orderings`: for every table with `M_a` minimal and every `δ` with
  `2d + δ ≠ 0`, the chain becomes constant and one of the four caps
  ("bounders") is active at the limit.

STATUS: written 2026-09-27 without a Lean toolchain at hand; not yet compiled.
-/

namespace BpVerify.Section4

/-! ## Hypotheses, the direct pass, and the general limits -/

/-- `M_a` is the strict minimum of the half table (w.l.o.g. by relabelling). -/
def IsMinAA (Ma Mb Ba Bb : ℝ) : Prop := Ma < Mb ∧ Ma < Ba ∧ Ma < Bb

/-- `M_a + M_b ≤ B_a + B_b`: the condition under which both direct maps are the
    clipped translations of `eq:pass`.  The paper's ordering implies it. -/
def ClipCond (Ma Mb Ba Bb : ℝ) : Prop := Ma + Mb ≤ Ba + Bb

/-- one complete pass computed with the direct min-sum maps, so that nothing about
    clipping is assumed. -/
def directPass (Ma Mb Ba Bb delta r : ℝ) : ℝ :=
  directToJ Ma Mb Ba Bb (delta + directToI Ma Mb Ba Bb r)

/-- one of the four caps is active at the message difference `r` toward `X_j`:
    `r` itself sits at a cap, or the message it produces toward `X_i` does. -/
def CapActive (Ma Mb Ba Bb r : ℝ) : Prop :=
  r = lowerJ Mb Ba ∨ r = upperJ Ma Bb ∨
    directToI Ma Mb Ba Bb r = lowerI Mb Bb ∨ directToI Ma Mb Ba Bb r = upperI Ma Ba

/-- limit of a chain when `2d + δ > 0`: the general two-sided form of `upperTarget`. -/
def upperLimit (Ma Mb Ba Bb delta : ℝ) : ℝ :=
  clip (upperI Ma Ba + d Ma Mb + delta) (lowerJ Mb Ba) (upperJ Ma Bb)

/-- limit of a chain when `2d + δ < 0`: the general two-sided form of `lowerTarget`. -/
def lowerLimit (Ma Mb Ba Bb delta : ℝ) : ℝ :=
  clip (lowerI Mb Bb + d Ma Mb + delta) (lowerJ Mb Ba) (upperJ Ma Bb)

/-! ## Small facts about `clip` -/

lemma clip_cases (x l h : ℝ) (hlh : l ≤ h) :
    clip x l h = l ∨ clip x l h = h ∨ clip x l h = x := by
  simp only [clip, max_def, min_def]
  split_ifs <;> simp

lemma clip_eq_hi_of_le (x l h : ℝ) (hlh : l ≤ h) (hx : h ≤ x) : clip x l h = h := by
  simp only [clip]
  rw [min_eq_right hx, max_eq_right hlh]

lemma clip_eq_lo_of_le (x l h : ℝ) (hx : x ≤ l) : clip x l h = l := by
  simp only [clip]
  exact max_eq_left (le_trans (min_le_left _ _) hx)

/-! ## Facts that hold for every table -/

/-- a message toward `X_j` lies between its two caps, whichever of them is larger. -/
lemma directToJ_range (Ma Mb Ba Bb q : ℝ) :
    min (lowerJ Mb Ba) (upperJ Ma Bb) ≤ directToJ Ma Mb Ba Bb q ∧
      directToJ Ma Mb Ba Bb q ≤ max (lowerJ Mb Ba) (upperJ Ma Bb) := by
  simp only [directToJ, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-- a message toward `X_i` lies between its two caps, whichever of them is larger. -/
lemma directToI_range (Ma Mb Ba Bb q : ℝ) :
    min (lowerI Mb Bb) (upperI Ma Ba) ≤ directToI Ma Mb Ba Bb q ∧
      directToI Ma Mb Ba Bb q ≤ max (lowerI Mb Bb) (upperI Ma Ba) := by
  simp only [directToI, lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-! ## The clip form of `eq:pass` holds exactly under `M_a + M_b ≤ B_a + B_b` -/

lemma lowerJ_le_upperJ (Ma Mb Ba Bb : ℝ) (hS : ClipCond Ma Mb Ba Bb) :
    lowerJ Mb Ba ≤ upperJ Ma Bb := by
  unfold ClipCond at hS
  simp only [lowerJ, upperJ]
  linarith

lemma lowerI_le_upperI (Ma Mb Ba Bb : ℝ) (hS : ClipCond Ma Mb Ba Bb) :
    lowerI Mb Bb ≤ upperI Ma Ba := by
  unfold ClipCond at hS
  simp only [lowerI, upperI]
  linarith

/-- `eq:pass` toward `X_j` under the sum condition alone (weaker than
    `directToJ_eq_clip`, which assumes `M_a < M_b < B_a` and `M_b < B_b`). -/
lemma directToJ_eq_toJ_of_clipCond (Ma Mb Ba Bb q : ℝ) (hS : ClipCond Ma Mb Ba Bb) :
    directToJ Ma Mb Ba Bb q = toJ Ma Mb Ba Bb q := by
  unfold ClipCond at hS
  simp only [directToJ, toJ, clip, d, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> linarith

/-- `eq:pass` toward `X_i` under the sum condition alone. -/
lemma directToI_eq_toI_of_clipCond (Ma Mb Ba Bb q : ℝ) (hS : ClipCond Ma Mb Ba Bb) :
    directToI Ma Mb Ba Bb q = toI Ma Mb Ba Bb q := by
  unfold ClipCond at hS
  simp only [directToI, toI, clip, d, lowerI, upperI, max_def, min_def]
  split_ifs <;> linarith

/-- under the sum condition the direct pass is the pass of `Section4.lean`. -/
lemma directPass_eq_pass (Ma Mb Ba Bb delta r : ℝ) (hS : ClipCond Ma Mb Ba Bb) :
    directPass Ma Mb Ba Bb delta r = pass Ma Mb Ba Bb delta r := by
  unfold directPass pass
  rw [directToI_eq_toI_of_clipCond Ma Mb Ba Bb r hS,
    directToJ_eq_toJ_of_clipCond Ma Mb Ba Bb _ hS]

/-- when the sum condition fails, the clip form is wrong for some incoming
    difference: the direct map reaches `B_b − M_a` while the clip returns `M_b − B_a`. -/
lemma exists_directToJ_ne_toJ_of_not_clipCond (Ma Mb Ba Bb : ℝ)
    (hS : Ba + Bb < Ma + Mb) :
    ∃ q : ℝ, directToJ Ma Mb Ba Bb q ≠ toJ Ma Mb Ba Bb q := by
  refine ⟨max (Ma - Ba) (Bb - Mb) + 1, ?_⟩
  simp only [directToJ, toJ, clip, d, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> intro h <;> linarith

/-! ## Constant outside difference under the sum condition -/

/-- `2d + δ > 0`: one pass moves a chain up by at least `2d + δ` until it reaches
    `upperLimit`, and never above it. -/
lemma directPass_upper_bounds (Ma Mb Ba Bb delta r : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 0 < 2 * (Mb - Ma) + delta) :
    min (upperLimit Ma Mb Ba Bb delta) (r + (2 * (Mb - Ma) + delta)) ≤
        directPass Ma Mb Ba Bb delta r ∧
      directPass Ma Mb Ba Bb delta r ≤ upperLimit Ma Mb Ba Bb delta := by
  unfold ClipCond at hS
  simp only [directPass, directToJ, directToI, upperLimit, clip, d, lowerJ, upperJ,
    lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-- `2d + δ < 0`: one pass moves a chain down by at least `|2d + δ|` until it reaches
    `lowerLimit`, and never below it (written with `r - c` for `finite_stabilization_down`). -/
lemma directPass_lower_bounds (Ma Mb Ba Bb delta r : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 2 * (Mb - Ma) + delta < 0) :
    lowerLimit Ma Mb Ba Bb delta ≤ directPass Ma Mb Ba Bb delta r ∧
      directPass Ma Mb Ba Bb delta r ≤
        max (lowerLimit Ma Mb Ba Bb delta) (r - (-(2 * (Mb - Ma) + delta))) := by
  unfold ClipCond at hS
  simp only [directPass, directToJ, directToI, lowerLimit, clip, d, lowerJ, upperJ,
    lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

lemma upperLimit_fixed (Ma Mb Ba Bb delta : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 0 < 2 * (Mb - Ma) + delta) :
    directPass Ma Mb Ba Bb delta (upperLimit Ma Mb Ba Bb delta) =
      upperLimit Ma Mb Ba Bb delta := by
  have h := directPass_upper_bounds Ma Mb Ba Bb delta (upperLimit Ma Mb Ba Bb delta) hS hc
  have hm : min (upperLimit Ma Mb Ba Bb delta)
      (upperLimit Ma Mb Ba Bb delta + (2 * (Mb - Ma) + delta)) =
      upperLimit Ma Mb Ba Bb delta := min_eq_left (by linarith)
  rw [hm] at h
  exact le_antisymm h.2 h.1

lemma lowerLimit_fixed (Ma Mb Ba Bb delta : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 2 * (Mb - Ma) + delta < 0) :
    directPass Ma Mb Ba Bb delta (lowerLimit Ma Mb Ba Bb delta) =
      lowerLimit Ma Mb Ba Bb delta := by
  have h := directPass_lower_bounds Ma Mb Ba Bb delta (lowerLimit Ma Mb Ba Bb delta) hS hc
  have hm : max (lowerLimit Ma Mb Ba Bb delta)
      (lowerLimit Ma Mb Ba Bb delta - (-(2 * (Mb - Ma) + delta))) =
      lowerLimit Ma Mb Ba Bb delta := max_eq_left (by linarith)
  rw [hm] at h
  exact le_antisymm h.2 h.1

/-- general form of Lemma 4.5, first case: with `2d + δ > 0` every chain is eventually
    equal to `[ (B_a − M_a) + d + δ ]_{M_b − B_a}^{B_b − M_a}`, from any start. -/
theorem directPass_converges_upper (Ma Mb Ba Bb delta r : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 0 < 2 * (Mb - Ma) + delta) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((directPass Ma Mb Ba Bb delta)^[n]) r = upperLimit Ma Mb Ba Bb delta := by
  have hstep : ∀ y, min (upperLimit Ma Mb Ba Bb delta) (y + (2 * (Mb - Ma) + delta)) ≤
      directPass Ma Mb Ba Bb delta y ∧
      directPass Ma Mb Ba Bb delta y ≤ upperLimit Ma Mb Ba Bb delta :=
    fun y => directPass_upper_bounds Ma Mb Ba Bb delta y hS hc
  have hfix := upperLimit_fixed Ma Mb Ba Bb delta hS hc
  -- the first pass lands inside the range, then the generic counting lemma applies
  obtain ⟨N, hN⟩ := finite_stabilization_up (directPass Ma Mb Ba Bb delta)
    (upperLimit Ma Mb Ba Bb delta) (2 * (Mb - Ma) + delta)
    (directPass Ma Mb Ba Bb delta r) hc (hstep r).2 hstep hfix
  refine ⟨N + 1, fun n hn => ?_⟩
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  rw [show N + 1 + k = (N + k) + 1 by omega, Function.iterate_succ_apply]
  exact hN (N + k) (Nat.le_add_right N k)

/-- general form of Lemma 4.5, second case: with `2d + δ < 0` every chain is eventually
    equal to `[ (M_b − B_b) + d + δ ]_{M_b − B_a}^{B_b − M_a}`, from any start. -/
theorem directPass_converges_lower (Ma Mb Ba Bb delta r : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 2 * (Mb - Ma) + delta < 0) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((directPass Ma Mb Ba Bb delta)^[n]) r = lowerLimit Ma Mb Ba Bb delta := by
  have hc' : 0 < -(2 * (Mb - Ma) + delta) := by linarith
  have hstep : ∀ y, lowerLimit Ma Mb Ba Bb delta ≤ directPass Ma Mb Ba Bb delta y ∧
      directPass Ma Mb Ba Bb delta y ≤
        max (lowerLimit Ma Mb Ba Bb delta) (y - (-(2 * (Mb - Ma) + delta))) :=
    fun y => directPass_lower_bounds Ma Mb Ba Bb delta y hS hc
  have hfix := lowerLimit_fixed Ma Mb Ba Bb delta hS hc
  obtain ⟨N, hN⟩ := finite_stabilization_down (directPass Ma Mb Ba Bb delta)
    (lowerLimit Ma Mb Ba Bb delta) (-(2 * (Mb - Ma) + delta))
    (directPass Ma Mb Ba Bb delta r) hc' (hstep r).1 hstep hfix
  refine ⟨N + 1, fun n hn => ?_⟩
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  rw [show N + 1 + k = (N + k) + 1 by omega, Function.iterate_succ_apply]
  exact hN (N + k) (Nat.le_add_right N k)

/-- at `upperLimit` a cap is active: the limit is a cap of `X_j`, or the message it
    produces toward `X_i` is at the upper cap `B_a − M_a`. -/
lemma capActive_upperLimit (Ma Mb Ba Bb delta : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 0 < 2 * (Mb - Ma) + delta) :
    CapActive Ma Mb Ba Bb (upperLimit Ma Mb Ba Bb delta) := by
  have hLU := lowerJ_le_upperJ Ma Mb Ba Bb hS
  have hlh := lowerI_le_upperI Ma Mb Ba Bb hS
  unfold CapActive upperLimit
  rcases clip_cases (upperI Ma Ba + d Ma Mb + delta) (lowerJ Mb Ba) (upperJ Ma Bb) hLU
    with h | h | h
  · left; exact h
  · right; left; exact h
  · right; right; right
    rw [directToI_eq_toI_of_clipCond Ma Mb Ba Bb _ hS, h]
    simp only [toI]
    apply clip_eq_hi_of_le _ _ _ hlh
    simp only [d, upperI]
    linarith

/-- at `lowerLimit` a cap is active: the limit is a cap of `X_j`, or the message it
    produces toward `X_i` is at the lower cap `M_b − B_b`. -/
lemma capActive_lowerLimit (Ma Mb Ba Bb delta : ℝ) (hS : ClipCond Ma Mb Ba Bb)
    (hc : 2 * (Mb - Ma) + delta < 0) :
    CapActive Ma Mb Ba Bb (lowerLimit Ma Mb Ba Bb delta) := by
  have hLU := lowerJ_le_upperJ Ma Mb Ba Bb hS
  unfold CapActive lowerLimit
  rcases clip_cases (lowerI Mb Bb + d Ma Mb + delta) (lowerJ Mb Ba) (upperJ Ma Bb) hLU
    with h | h | h
  · left; exact h
  · right; left; exact h
  · right; right; left
    rw [directToI_eq_toI_of_clipCond Ma Mb Ba Bb _ hS, h]
    simp only [toI]
    apply clip_eq_lo_of_le
    simp only [d, lowerI]
    linarith

/-! ## The dominant case `B_a ≤ M_b`: `X_j = a` is optimal against both values of `X_i` -/

/-- a non-negative message toward `X_j` produces the upper cap toward `X_i`. -/
lemma directToI_eq_upperI_of_nonneg (Ma Mb Ba Bb r : ℝ) (hMaBb : Ma ≤ Bb) (hBaMb : Ba ≤ Mb)
    (hr : 0 ≤ r) : directToI Ma Mb Ba Bb r = upperI Ma Ba := by
  simp only [directToI, upperI]
  rw [min_eq_left (by linarith), min_eq_left (by linarith)]

/-- every message toward `X_j` is non-negative when `B_a ≤ M_b`. -/
lemma directToJ_nonneg_of_dominant (Ma Mb Ba Bb q : ℝ) (hMaBb : Ma ≤ Bb) (hBaMb : Ba ≤ Mb) :
    0 ≤ directToJ Ma Mb Ba Bb q := by
  simp only [directToJ, min_def]
  split_ifs <;> linarith

/-- when `B_a ≤ M_b` the chain is constant from the second pass on, for every `δ`:
    the message toward `X_i` is pinned at `B_a − M_a` from the first pass. -/
theorem dominant_constant_from_two (Ma Mb Ba Bb delta r0 : ℝ) (hMaBb : Ma ≤ Bb)
    (hBaMb : Ba ≤ Mb) (n : ℕ) :
    ((directPass Ma Mb Ba Bb delta)^[n + 2]) r0 =
      directToJ Ma Mb Ba Bb (delta + upperI Ma Ba) := by
  rw [Function.iterate_succ_apply']
  have hpos : 0 ≤ ((directPass Ma Mb Ba Bb delta)^[n + 1]) r0 := by
    rw [Function.iterate_succ_apply']
    exact directToJ_nonneg_of_dominant Ma Mb Ba Bb _ hMaBb hBaMb
  show directToJ Ma Mb Ba Bb
      (delta + directToI Ma Mb Ba Bb (((directPass Ma Mb Ba Bb delta)^[n + 1]) r0)) = _
  rw [directToI_eq_upperI_of_nonneg Ma Mb Ba Bb _ hMaBb hBaMb hpos]

/-! ## Every ordering -/

/-- **All orderings.**  For every half table with `M_a` minimal and every constant
    outside difference `δ` with `2d + δ ≠ 0`, the chain of messages toward `X_j`
    becomes constant after finitely many complete passes, and at the limit one of
    the four caps is active.  The proof splits on `B_a ≤ M_b` (`X_j = a` dominant,
    no flipping threshold exists) against `M_b ≤ B_a` (then `ClipCond` holds and
    `eq:pass` applies). -/
theorem all_orderings (Ma Mb Ba Bb delta r0 : ℝ) (hmin : IsMinAA Ma Mb Ba Bb)
    (hc : 2 * (Mb - Ma) + delta ≠ 0) :
    ∃ N : ℕ,
      (∀ n, N ≤ n →
        ((directPass Ma Mb Ba Bb delta)^[n]) r0 = ((directPass Ma Mb Ba Bb delta)^[N]) r0) ∧
      CapActive Ma Mb Ba Bb (((directPass Ma Mb Ba Bb delta)^[N]) r0) := by
  have hmin' : Ma < Mb ∧ Ma < Ba ∧ Ma < Bb := hmin
  obtain ⟨hMaMb, hMaBa, hMaBb⟩ := hmin'
  rcases le_total Ba Mb with hBaMb | hMbBa
  · -- `X_j = a` is optimal against both values of `X_i`
    refine ⟨2, ?_, ?_⟩
    · intro n hn
      obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
      rw [show 2 + k = k + 2 by omega,
        dominant_constant_from_two Ma Mb Ba Bb delta r0 hMaBb.le hBaMb k]
      exact (dominant_constant_from_two Ma Mb Ba Bb delta r0 hMaBb.le hBaMb 0).symm
    · unfold CapActive
      right; right; right
      apply directToI_eq_upperI_of_nonneg Ma Mb Ba Bb _ hMaBb.le hBaMb
      rw [show (2 : ℕ) = 1 + 1 by rfl, Function.iterate_succ_apply']
      exact directToJ_nonneg_of_dominant Ma Mb Ba Bb _ hMaBb.le hBaMb
  · -- `M_b ≤ B_a`: the sum condition holds and `eq:pass` applies
    have hS : ClipCond Ma Mb Ba Bb := by
      unfold ClipCond
      linarith
    rcases lt_or_gt_of_ne hc with hneg | hpos
    · obtain ⟨N, hN⟩ := directPass_converges_lower Ma Mb Ba Bb delta r0 hS hneg
      refine ⟨N, ?_, ?_⟩
      · intro n hn
        rw [hN n hn, hN N le_rfl]
      · rw [hN N le_rfl]
        exact capActive_lowerLimit Ma Mb Ba Bb delta hS hneg
    · obtain ⟨N, hN⟩ := directPass_converges_upper Ma Mb Ba Bb delta r0 hS hpos
      refine ⟨N, ?_, ?_⟩
      · intro n hn
        rw [hN n hn, hN N le_rfl]
      · rw [hN N le_rfl]
        exact capActive_upperLimit Ma Mb Ba Bb delta hS hpos

end BpVerify.Section4
