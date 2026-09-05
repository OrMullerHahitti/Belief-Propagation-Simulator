import Mathlib
import BpVerify.FiniteTime

/-!
# Summable perturbations (Remark 7.10)

PAPER REFERENCE: Section 7, Remark 7.10 (immediately after Theorem 7.7).

The remark gives two relaxations of the static `B`-bounded perturbation
hypothesis:

* **Eventual bound** — handled directly by `reach_lower_regime_dynamic`
  starting from step `i_0`.
* **Summable** — `Σ |δⁱ_φ| < ∞` implies `|δⁱ_φ| → 0`, hence eventually
  bounded by any positive constant. This is the lemma below; combining
  with the eventual-bound case gives the same conclusion as Theorem 7.7
  applied from the iteration at which the perturbation becomes small.

The main theorem `summable_implies_eventual_bound` is exactly Remark
7.10 (ii). The helper `exists_eventually_small_of_summable` is the
generic real-analysis fact it relies on.
-/

namespace BpVerify

open Filter Topology

/-- A summable real sequence eventually falls below any positive bound. -/
lemma exists_eventually_small_of_summable
    (a : ℕ → ℝ) (h : Summable (fun i => |a i|)) {B : ℝ} (hB : 0 < B) :
    ∃ N : ℕ, ∀ i, N ≤ i → |a i| < B := by
  have h_tendsto : Tendsto (fun i => |a i|) atTop (𝓝 0) :=
    h.tendsto_atTop_zero
  rw [Metric.tendsto_atTop] at h_tendsto
  obtain ⟨N, hN⟩ := h_tendsto B hB
  refine ⟨N, fun i hi => ?_⟩
  have := hN i hi
  simp only [Real.dist_eq, sub_zero, abs_abs] at this
  exact this

/-- Remark 7.10 part (ii): a summable perturbation eventually behaves as
a `B`-bounded perturbation for any positive `B`. -/
theorem summable_implies_eventual_bound
    (delta_phi : ℕ → ℝ) (h_summable : Summable (fun i => |delta_phi i|))
    (B : ℝ) (hB : 0 < B) :
    ∃ i_0 : ℕ, ∀ i, i_0 ≤ i → |delta_phi i| < B :=
  exists_eventually_small_of_summable delta_phi h_summable hB

end BpVerify
