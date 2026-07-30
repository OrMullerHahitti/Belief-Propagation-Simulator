import Mathlib
import BpVerify.Basic
import BpVerify.FiniteTime
import BpVerify.RoundTrip

/-!
# Dynamic round-trip map and Theorem 7.7 (concrete)

PAPER REFERENCE: Section 7 of main.tex.

A unary perturbation `φ` on `X_1` shifts the rows of `C'` by `φ`. The
forward step uses the perturbed table; the backward step uses bare `C'`
(since `X_2` carries no perturbation). The key identity is

  T_dyn(φ, q) = T(q + φ).

This reduces all dynamic claims to claims about the static `T` at a
shifted argument. `thm77_concrete` is the end-to-end Lean version of
Theorem 7.7, with both halves of the paper's `B < min(2|d|, τ_U + τ_L)`
bound exposed as named hypotheses.
-/

namespace BpVerify
namespace CostTable

/-! ## Dynamic forward step and round-trip -/

/-- Forward step with unary perturbation `φ` on `X_1` (rows shifted by φ). -/
def deltaR_dyn (C : CostTable) (phi q : ℝ) : ℝ :=
  min (C.caa + phi + q) C.cba - min (C.cab + phi + q) C.cbb

/-- Dynamic round-trip: perturbed forward step, bare backward step. -/
def T_dyn (C : CostTable) (phi q : ℝ) : ℝ := deltaQ_next C (deltaR_dyn C phi q)

/-- Key identity: `T_dyn(φ, q) = T(q + φ)`. The perturbation just shifts
    the input to the static round-trip. -/
lemma T_dyn_eq_shift (C : CostTable) (phi q : ℝ) :
    T_dyn C phi q = T C (q + phi) := by
  unfold T_dyn T deltaR_dyn deltaR
  congr 2 <;> ring_nf

/-! ## Absorption under bounded perturbation (uses BOTH bounds) -/

/-- Persistence step of Theorem 7.7: the safe lower zone `(-∞, τ_L − B]`
    is absorbing under any perturbation `|φ| ≤ B`, provided `B` satisfies
    BOTH halves of the paper's bound:
        `B < 2|d|`         (drift bound)
        `B < τ_U + τ_L`    (margin bound)
    This is where Option A makes both halves visible. -/
lemma T_dyn_absorbs_lower_of_bounds
    (C : CostTable) (hd : d C < 0)
    (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C)
    (B : ℝ)
    (hB_drift  : B < 2 * |d C|)
    (hB_margin : B < tau_U C + tau_L C)
    (phi : ℝ) (hphi : |phi| ≤ B) :
    ∀ q, q ≤ tau_L C - B → T_dyn C phi q ≤ tau_L C - B := by
  intro q hq
  rw [T_dyn_eq_shift]
  -- q + φ ≤ τ_L (since φ ≤ B and q ≤ τ_L − B)
  have hphi_le : phi ≤ B := (abs_le.mp hphi).2
  have hq' : q + phi ≤ tau_L C := by linarith
  -- So T(q + φ) = cL_closed; show cL_closed ≤ τ_L − B using both bounds.
  rw [T_eq_cL_closed_of_lower C hd hsum hΔ hq']
  unfold cL_closed
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd)
  split_ifs with h1 h2
  · -- Case I (τ_L ≥ |d|): cL = -τ_L. Need -τ_L ≤ τ_L − B, i.e., B ≤ 2τ_L.
    -- B < 2|d| ≤ 2τ_L. (uses hB_drift)
    linarith
  · -- Case II: cL = τ_L - 2|d|. Need τ_L - 2|d| ≤ τ_L − B, i.e., B ≤ 2|d|.
    -- From hB_drift.
    linarith
  · -- Case III (τ_L < |d|, sum ≤ 2|d|): cL = -τ_U. Need -τ_U ≤ τ_L − B,
    -- i.e., B ≤ τ_U + τ_L. From hB_margin.
    linarith

/-! ## Descent in the wobbled interval -/

/-- Lemma 7.6 (concrete). In the wobbled interval `(τ_L − B, τ_U + B]`,
    one round-trip either reaches the safe lower zone or descends by
    `2|d| − B`. Like the absorption lemma, this needs BOTH halves of the
    bound (drift bound for the descent magnitude, margin bound for
    boundary cases of the case analysis). -/
lemma T_dyn_descent_in_wobbled
    (C : CostTable) (hd : d C < 0)
    (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C)
    (B : ℝ)
    (hB_drift  : B < 2 * |d C|)
    (hB_margin : B < tau_U C + tau_L C)
    (phi : ℝ) (hphi : |phi| ≤ B) :
    ∀ q, tau_L C - B < q → q ≤ tau_U C + B →
         T_dyn C phi q ≤ tau_L C - B ∨
         T_dyn C phi q ≤ q - (2 * |d C| - B) := by
  intro q hL hU
  rw [T_dyn_eq_shift]
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd)
  have hphi_lo : -B ≤ phi := (abs_le.mp hphi).1
  have hphi_hi : phi ≤ B := (abs_le.mp hphi).2
  -- Three cases on where q + phi lies: lower, transition, or upper.
  by_cases hLow : q + phi ≤ tau_L C
  · -- Lower: T(q + phi) = cL_closed; show cL_closed ≤ τ_L − B.
    left
    rw [T_eq_cL_closed_of_lower C hd hsum hΔ hLow]
    unfold cL_closed
    have hB_drift' : B < 2 * |d C| := hB_drift
    have hB_margin' : B < tau_U C + tau_L C := hB_margin
    split_ifs with h1 h2 <;> linarith
  · push_neg at hLow
    by_cases hHigh : q + phi ≤ tau_U C
    · -- Transition: direct case analysis on the backward min branches,
      -- mirroring `T_descent_in_interval` but with q + phi in place of q.
      unfold T
      rw [deltaR_transition C hLow hHigh]
      unfold deltaQ_next
      have e_cbb_caa := cbb_sub_caa C hd
      have e_cab_caa := cab_sub_caa C hd
      have e_cbb_cba := cbb_sub_cba C hd
      have habs : |d C| = -d C := abs_of_neg hd
      by_cases hQa : C.caa + (q + phi + d C) ≤ C.cab
      · by_cases hQb : C.cba + (q + phi + d C) ≤ C.cbb
        · -- TII (both row a): T = caa - cba = -τ_U. LEFT via hB_margin.
          left
          rw [min_eq_left hQa, min_eq_left hQb]
          rw [tau_U_def] at hsum hB_margin; rw [tau_L_def] at *; linarith
        · -- TI (rows-split): T = caa + (q+phi+d) - cbb. RIGHT.
          right
          push_neg at hQb
          rw [min_eq_left hQa, min_eq_right hQb.le]
          linarith
      · push_neg at hQa
        by_cases hQb : C.cba + (q + phi + d C) ≤ C.cbb
        · -- TIII: vacuous via hΔ.
          exfalso
          rw [tau_U_def, tau_L_def] at hΔ; linarith
        · -- TIV (both row b): T = cab - cbb = -τ_L. RIGHT (uses ¬hQa).
          right
          push_neg at hQb
          rw [min_eq_right hQa.le, min_eq_right hQb.le]
          rw [tau_L_def] at *; linarith
    · push_neg at hHigh
      -- Upper: T(q+phi) = cU_closed.
      rw [T_eq_cU_closed_of_upper C hd hsum hΔ hHigh]
      unfold cU_closed
      split_ifs with hc1 hc2
      · right; linarith   -- Case I: c_U = -τ_L. RIGHT (uses sum ≥ 2|d|).
      · right; linarith   -- Case II: c_U = τ_U - 2|d|. RIGHT.
      · left; linarith    -- Case III: c_U = -τ_U. LEFT via hB_margin.

/-! ## End-to-end Theorem 7.7 (concrete) -/

/-- Theorem 7.7, end-to-end in Lean, for arbitrary starting point in `ℝ`.
    For `x0 > τ_U + B` the first round-trip lands at `cU_closed ≤ τ_U`
    (one application of `T_eq_cU_closed_of_upper` to the shifted argument
    `x0 + φ_0`), which sits below `τ_U + B`; the sequence-varying
    descent argument then takes over. -/
theorem thm77_concrete
    (C : CostTable) (hd_neg : d C < 0)
    (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ   : tau_L C ≤ tau_U C)
    (B : ℝ)
    (hB_drift  : B < 2 * |d C|)          -- paper's first half
    (hB_margin : B < tau_U C + tau_L C)  -- paper's second half (named!)
    (phi : ℕ → ℝ) (hphi : ∀ i, |phi i| ≤ B)
    (x0 : ℝ) :
    ∃ N : ℕ, ∀ n, N ≤ n →
        traj (fun i => T_dyn C (phi i)) x0 n ≤ tau_L C - B := by
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd_neg)
  have hphi_hi : phi 0 ≤ B := (abs_le.mp (hphi 0)).2
  by_cases hx0 : x0 ≤ tau_U C + B
  · exact reach_lower_regime_dynamic_seq
      (fun i => T_dyn C (phi i)) (tau_L C) (tau_U C) |d C| B
      habs_pos hB_drift
      (fun i x hx_L hx_U =>
        T_dyn_descent_in_wobbled C hd_neg hsum hΔ B hB_drift hB_margin
          (phi i) (hphi i) x hx_L hx_U)
      (fun i x hx =>
        T_dyn_absorbs_lower_of_bounds C hd_neg hsum hΔ B hB_drift hB_margin
          (phi i) (hphi i) x hx)
      x0 hx0
  · push_neg at hx0
    -- x0 > τ_U + B. We first take one dynamic step. The dynamic round-trip
    -- equals `T (x0 + φ_0)`. Since x0 + φ_0 > τ_U + B + φ_0 ≥ τ_U + B − B = τ_U
    -- (using φ_0 ≥ −B), apply `T_eq_cU_closed_of_upper` and `cU_closed ≤ τ_U`
    -- to conclude the next state is ≤ τ_U ≤ τ_U + B.
    have hphi_lo : -B ≤ phi 0 := (abs_le.mp (hphi 0)).1
    have h_shift : tau_U C < x0 + phi 0 := by linarith
    have h_step1 : T_dyn C (phi 0) x0 ≤ tau_U C + B := by
      rw [T_dyn_eq_shift, T_eq_cU_closed_of_upper C hd_neg hsum hΔ h_shift]
      have h_cU := cU_closed_le_tau_U C hd_neg hsum hΔ
      linarith
    -- Apply the seq-varying theorem starting from the state after one step,
    -- using the shifted perturbation sequence `phi (· + 1)`.
    obtain ⟨N, hN⟩ := reach_lower_regime_dynamic_seq
      (fun i => T_dyn C (phi (i + 1))) (tau_L C) (tau_U C) |d C| B
      habs_pos hB_drift
      (fun i x hx_L hx_U =>
        T_dyn_descent_in_wobbled C hd_neg hsum hΔ B hB_drift hB_margin
          (phi (i + 1)) (hphi (i + 1)) x hx_L hx_U)
      (fun i x hx =>
        T_dyn_absorbs_lower_of_bounds C hd_neg hsum hΔ B hB_drift hB_margin
          (phi (i + 1)) (hphi (i + 1)) x hx)
      (T_dyn C (phi 0) x0) h_step1
    -- Relate the shifted trajectory to the original.
    refine ⟨N + 1, ?_⟩
    intro n hn
    cases n with
    | zero => omega
    | succ m =>
      have hm : N ≤ m := by omega
      -- traj (fun i => T_dyn C (phi i)) x0 (m+1)
      --   = T_dyn C (phi m) (traj ... x0 m)
      -- We need to relate to traj (fun i => T_dyn C (phi (i+1))) (T_dyn C (phi 0) x0) m.
      have rel : ∀ k, traj (fun i => T_dyn C (phi i)) x0 (k + 1) =
                     traj (fun i => T_dyn C (phi (i + 1))) (T_dyn C (phi 0) x0) k := by
        intro k
        induction k with
        | zero => rfl
        | succ j ih =>
          show T_dyn C (phi (j + 1)) (traj (fun i => T_dyn C (phi i)) x0 (j + 1))
             = T_dyn C (phi (j + 1)) (traj (fun i => T_dyn C (phi (i + 1)))
                  (T_dyn C (phi 0) x0) j)
          rw [ih]
      rw [rel m]
      exact hN m hm

end CostTable
end BpVerify
