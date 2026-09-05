import Mathlib

/-!
# WLOG relabelling for the two-node binary pairwise factor graph

PAPER REFERENCE: Section 6, "WLOG relabelling" paragraph (just before the
statement of Lemma 6.1).

Formalises the two-step WLOG procedure of Section 6: any cost table with
`d ≠ 0` can be relabelled so that `d < 0` and `τ_U + τ_L ≥ 0`.

Step 1: if `d > 0`, swap `a ↔ b` in both `X₁` and `X₂` (sends `d → -d`).
Step 2: if `τ_U + τ_L < 0`, transpose (`X₁ ↔ X₂`); transpose preserves `d`
        and the old/new sums satisfy `old + new = -2d`, so at least one is ≥ 0.

The main theorem here is `WLOG_orientation` (bottom of file). All the
intermediate lemmas (`d_swapBoth`, `d_transpose`, `sumD_transpose_identity`,
etc.) are the algebraic identities that the paper asserts informally in
the WLOG paragraph.
-/

namespace BpVerify

/-- Cost table for the two-node binary pairwise factor graph. -/
structure CostTable where
  caa : ℝ  -- C'(a, a)
  cab : ℝ  -- C'(a, b)
  cba : ℝ  -- C'(b, a)
  cbb : ℝ  -- C'(b, b)

namespace CostTable

/-- Drift constant `d := C'(a,a) − C'(b,b)`. -/
def d (C : CostTable) : ℝ := C.caa - C.cbb

/-- Column-`a` cost difference `Δ_a := C'(b,a) − C'(a,a)`. -/
def Delta_a (C : CostTable) : ℝ := C.cba - C.caa

/-- Column-`b` cost difference `Δ_b := C'(b,b) − C'(a,b)`. -/
def Delta_b (C : CostTable) : ℝ := C.cbb - C.cab

/-- Sum `τ_U + τ_L = Δ_a + Δ_b` (under the convention `Δ_a ≥ Δ_b`). -/
def sumD (C : CostTable) : ℝ := Delta_a C + Delta_b C

/-- Step 1 relabelling: swap `a ↔ b` in both `X₁` and `X₂`. -/
def swapBoth (C : CostTable) : CostTable :=
  { caa := C.cbb, cab := C.cba, cba := C.cab, cbb := C.caa }

/-- Step 2 relabelling: swap `X₁ ↔ X₂` (transpose of `C'`). -/
def transpose (C : CostTable) : CostTable :=
  { caa := C.caa, cab := C.cba, cba := C.cab, cbb := C.cbb }

/-- `swapBoth` sends `d` to `−d`. -/
lemma d_swapBoth (C : CostTable) : d (swapBoth C) = -(d C) := by
  unfold d swapBoth
  ring

/-- `transpose` preserves `d`. -/
lemma d_transpose (C : CostTable) : d (transpose C) = d C := by
  unfold d transpose
  ring

/-- `swapBoth` preserves the WLOG ordering `Δ_a ≥ Δ_b`. -/
lemma swapBoth_preserves_order (C : CostTable) :
    Delta_a (swapBoth C) ≥ Delta_b (swapBoth C) ↔ Delta_a C ≥ Delta_b C := by
  unfold Delta_a Delta_b swapBoth
  constructor
  · intro h; linarith
  · intro h; linarith

/-- `swapBoth` flips the sign of `sumD`. -/
lemma sumD_swapBoth (C : CostTable) : sumD (swapBoth C) = -(sumD C) := by
  unfold sumD Delta_a Delta_b swapBoth
  ring

/-- Step 2 identity: old sum + new sum (after transpose) `= -2d`. -/
lemma sumD_transpose_identity (C : CostTable) :
    sumD C + sumD (transpose C) = -2 * d C := by
  unfold sumD Delta_a Delta_b transpose d
  ring

/-- WLOG: given any cost table with `d ≠ 0`, there exists a relabelled cost
    table with `d < 0` and `sumD ≥ 0`.

    PAPER REFERENCE: Section 6, "WLOG relabelling" paragraph. The paper asserts
    this two-step procedure works; this theorem proves it. The two cases
    (`hd_neg` / its negation) correspond exactly to "if d < 0 already, skip
    Step 1" and "if d > 0, apply Step 1 first" in the paper. -/
theorem WLOG_orientation (C : CostTable) (hd : d C ≠ 0) :
    ∃ C' : CostTable, d C' < 0 ∧ sumD C' ≥ 0 := by
  by_cases hd_neg : d C < 0
  · -- d C < 0 already.
    by_cases hs : sumD C ≥ 0
    · exact ⟨C, hd_neg, hs⟩
    · -- Apply transpose to fix the sum.
      refine ⟨transpose C, ?_, ?_⟩
      · rw [d_transpose]; exact hd_neg
      · push_neg at hs
        have hsum := sumD_transpose_identity C
        linarith
  · -- d C ≥ 0 and d C ≠ 0, so d C > 0; apply swapBoth.
    push_neg at hd_neg
    have hd_pos : 0 < d C := lt_of_le_of_ne hd_neg (Ne.symm hd)
    have d_after_swap : d (swapBoth C) < 0 := by
      rw [d_swapBoth]; linarith
    by_cases hs : sumD (swapBoth C) ≥ 0
    · exact ⟨swapBoth C, d_after_swap, hs⟩
    · refine ⟨transpose (swapBoth C), ?_, ?_⟩
      · rw [d_transpose]; exact d_after_swap
      · push_neg at hs
        have hsum := sumD_transpose_identity (swapBoth C)
        linarith

end CostTable
end BpVerify
