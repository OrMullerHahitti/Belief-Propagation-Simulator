import Mathlib
import BpVerify.Basic
import BpVerify.FiniteTime

/-!
# Concrete round-trip map T : ΔQ ↦ ΔQ_next

PAPER REFERENCE: Sections 4-6 of main.tex.

This file bridges the abstract `G : ℝ → ℝ` of `FiniteTime.lean` to the actual
round-trip map of min-sum message passing on the binary pairwise factor graph.

After this file, `thm63_concrete` is the end-to-end Lean version of Theorem 6.3
— no hypothesis about a generic `G`, only hypotheses about the cost table.
-/

namespace BpVerify
namespace CostTable

/-! ## Definitions -/

/-- Upper threshold τ_U := Δ_a = cba - caa. -/
def tau_U (C : CostTable) : ℝ := Delta_a C

/-- Lower threshold τ_L := Δ_b = cbb - cab. -/
def tau_L (C : CostTable) : ℝ := Delta_b C

lemma tau_U_def (C : CostTable) : tau_U C = C.cba - C.caa := rfl

lemma tau_L_def (C : CostTable) : tau_L C = C.cbb - C.cab := rfl

/-- Forward step (eq:minsumR with Q_b normalized to 0, Q_a = q). -/
def deltaR (C : CostTable) (q : ℝ) : ℝ :=
  min (C.caa + q) C.cba - min (C.cab + q) C.cbb

/-- Backward step (eq:minsumQ with R_b normalized to 0, R_a = r). -/
def deltaQ_next (C : CostTable) (r : ℝ) : ℝ :=
  min (C.caa + r) C.cab - min (C.cba + r) C.cbb

/-- Round-trip ΔQ^i ↦ ΔQ^{i+2}. -/
def T (C : CostTable) (q : ℝ) : ℝ := deltaQ_next C (deltaR C q)

/-- Closed form of the lower round-trip constant (paper eq:cL,
    verified algebraically by `verify.py :: c_L closed form equals eq:cL`). -/
noncomputable def cL_closed (C : CostTable) : ℝ :=
  if tau_L C ≥ |d C| then -tau_L C
  else if tau_U C + tau_L C > 2 * |d C| then tau_L C - 2 * |d C|
  else -tau_U C

/-! ## Algebraic identities under WLOG (d < 0) -/

lemma abs_d_eq_neg (C : CostTable) (hd : d C < 0) : |d C| = -d C :=
  abs_of_neg hd

/-- caa - cab = τ_L - |d|. -/
lemma caa_sub_cab (C : CostTable) (hd : d C < 0) :
    C.caa - C.cab = tau_L C - |d C| := by
  rw [abs_d_eq_neg C hd, tau_L_def, d]; ring

/-- cba - cbb = τ_U - |d|. -/
lemma cba_sub_cbb (C : CostTable) (hd : d C < 0) :
    C.cba - C.cbb = tau_U C - |d C| := by
  rw [abs_d_eq_neg C hd, tau_U_def, d]; ring

/-- cbb - caa = |d|. -/
lemma cbb_sub_caa (C : CostTable) (hd : d C < 0) :
    C.cbb - C.caa = |d C| := by
  rw [abs_d_eq_neg C hd, d]; ring

/-- cab - caa = |d| - τ_L. -/
lemma cab_sub_caa (C : CostTable) (hd : d C < 0) :
    C.cab - C.caa = |d C| - tau_L C := by
  rw [abs_d_eq_neg C hd, tau_L_def, d]; ring

/-- cbb - cba = |d| - τ_U. -/
lemma cbb_sub_cba (C : CostTable) (hd : d C < 0) :
    C.cbb - C.cba = |d C| - tau_U C := by
  rw [abs_d_eq_neg C hd, tau_U_def, d]; ring

/-! ## Forward-step case lemmas -/

/-- Lower regime: both columns pick row a, deltaR = caa - cab. -/
lemma deltaR_lower (C : CostTable) (hΔ : tau_L C ≤ tau_U C)
    {q : ℝ} (hq : q ≤ tau_L C) :
    deltaR C q = C.caa - C.cab := by
  unfold deltaR
  have hUq : C.caa + q ≤ C.cba := by
    rw [tau_U_def, tau_L_def] at hΔ; rw [tau_L_def] at hq; linarith
  have hLq : C.cab + q ≤ C.cbb := by
    rw [tau_L_def] at hq; linarith
  rw [min_eq_left hUq, min_eq_left hLq]; ring

/-- Transition: column a picks row a, column b picks row b, deltaR = q + d. -/
lemma deltaR_transition (C : CostTable)
    {q : ℝ} (hL : tau_L C < q) (hU : q ≤ tau_U C) :
    deltaR C q = q + d C := by
  unfold deltaR d
  have hUq : C.caa + q ≤ C.cba := by rw [tau_U_def] at hU; linarith
  have hLq : C.cbb ≤ C.cab + q := by rw [tau_L_def] at hL; linarith
  rw [min_eq_left hUq, min_eq_right hLq]; ring

/-- Upper regime: both columns pick row b, deltaR = cba - cbb. -/
lemma deltaR_upper (C : CostTable) (hΔ : tau_L C ≤ tau_U C)
    {q : ℝ} (hq : tau_U C < q) :
    deltaR C q = C.cba - C.cbb := by
  unfold deltaR
  have hUq : C.cba ≤ C.caa + q := by rw [tau_U_def] at hq; linarith
  have hLq : C.cbb ≤ C.cab + q := by
    rw [tau_U_def] at hq; rw [tau_U_def, tau_L_def] at hΔ; linarith
  rw [min_eq_right hUq, min_eq_right hLq]

/-! ## Lower regime: T(q) is constant and equals cL_closed -/

lemma T_eq_cL_closed_of_lower
    (C : CostTable) (hd : d C < 0)
    (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C)
    {q : ℝ} (hq : q ≤ tau_L C) :
    T C q = cL_closed C := by
  unfold T
  rw [deltaR_lower C hΔ hq]
  unfold deltaQ_next cL_closed
  have e_caa_cab := caa_sub_cab C hd
  have e_cba_cbb := cba_sub_cbb C hd
  have e_cbb_caa := cbb_sub_caa C hd
  have e_cab_caa := cab_sub_caa C hd
  have e_cbb_cba := cbb_sub_cba C hd
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd)
  by_cases h1 : tau_L C ≥ |d C|
  · -- Case I: τ_L ≥ |d|. Both Q branches pick row b. T = cab - cbb = -τ_L.
    have h_Qa : C.cab ≤ C.caa + (C.caa - C.cab) := by linarith
    -- For Q_b: cbb ≤ cba + (caa - cab) ⟺ τ_U + τ_L ≥ 2|d|, from τ_U ≥ τ_L ≥ |d|.
    have h_Qb : C.cbb ≤ C.cba + (C.caa - C.cab) := by linarith
    rw [min_eq_right h_Qa, min_eq_right h_Qb, if_pos h1]
    rw [tau_L_def]; ring
  · push_neg at h1
    -- τ_L < |d|. Q_a picks caa + r (i.e., row a, since caa < cab).
    have h_Qa : C.caa + (C.caa - C.cab) ≤ C.cab := by linarith
    by_cases h2 : tau_U C + tau_L C > 2 * |d C|
    · -- Case II: T = 2caa - cab - cbb = τ_L - 2|d|.
      have h_Qb : C.cbb ≤ C.cba + (C.caa - C.cab) := by linarith
      rw [min_eq_left h_Qa, min_eq_right h_Qb,
          if_neg (not_le.mpr h1), if_pos h2]
      linarith
    · -- Case III: T = 2caa - cab - (cba + caa - cab) = caa - cba = -τ_U.
      push_neg at h2
      have h_Qb : C.cba + (C.caa - C.cab) ≤ C.cbb := by linarith
      rw [min_eq_left h_Qa, min_eq_left h_Qb,
          if_neg (not_le.mpr h1), if_neg (not_lt.mpr h2)]
      rw [tau_U_def]; ring

/-! ## Absorption -/

lemma cL_closed_le_tau_L
    (C : CostTable) (hd : d C < 0) (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C) :
    cL_closed C ≤ tau_L C := by
  unfold cL_closed
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd)
  split_ifs with h1 h2
  · linarith   -- Case I: -τ_L ≤ τ_L, from τ_L ≥ |d| > 0.
  · linarith   -- Case II: τ_L - 2|d| ≤ τ_L.
  · linarith   -- Case III: -τ_U ≤ τ_L iff τ_U + τ_L ≥ 0.

/-- Lower regime is absorbing: once `q ≤ τ_L`, `T C q ≤ τ_L`. -/
lemma T_absorbs_lower
    (C : CostTable) (hd : d C < 0) (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C) :
    ∀ q, q ≤ tau_L C → T C q ≤ tau_L C := fun q hq => by
  rw [T_eq_cL_closed_of_lower C hd hsum hΔ hq]
  exact cL_closed_le_tau_L C hd hsum hΔ

/-! ## Descent in the transition interval (Lemma 6.1) -/

lemma T_descent_in_interval
    (C : CostTable) (hd : d C < 0) (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C) :
    ∀ q, tau_L C < q → q ≤ tau_U C →
         T C q ≤ tau_L C ∨ T C q ≤ q - 2 * |d C| := by
  intro q hL hU
  unfold T
  rw [deltaR_transition C hL hU]
  unfold deltaQ_next
  have e_cbb_caa := cbb_sub_caa C hd
  have e_cab_caa := cab_sub_caa C hd
  have e_cbb_cba := cbb_sub_cba C hd
  have habs : |d C| = -d C := abs_of_neg hd
  -- Two min branches at r = q + d:
  --   Q_a: caa + (q + d) vs cab  switches at q = 2|d| − τ_L
  --   Q_b: cba + (q + d) vs cbb  switches at q = 2|d| − τ_U
  by_cases hQa : C.caa + (q + d C) ≤ C.cab
  · by_cases hQb : C.cba + (q + d C) ≤ C.cbb
    · -- Both pick row a. T = caa - cba = -τ_U ≤ τ_L (from hsum).
      left
      rw [min_eq_left hQa, min_eq_left hQb]
      rw [tau_U_def] at hsum; rw [tau_L_def] at *; linarith
    · -- Q_a row a, Q_b row b. T = caa + (q+d) - cbb = q + 2d = q - 2|d|.
      right
      push_neg at hQb
      rw [min_eq_left hQa, min_eq_right hQb.le]
      linarith
  · push_neg at hQa
    by_cases hQb : C.cba + (q + d C) ≤ C.cbb
    · -- Vacuous: from hΔ this case is impossible.
      exfalso
      -- hQa: cab < caa + q + d  ⟹  q > cab - caa - d = (|d| - τ_L) + |d| = 2|d| - τ_L
      -- hQb: cba + q + d ≤ cbb  ⟹  q ≤ cbb - cba - d = -(τ_U - |d|) + |d| = 2|d| - τ_U
      -- so τ_U < τ_L, contradicting hΔ.
      linarith
    · -- Both pick row b. T = cab - cbb = -τ_L. Use RIGHT: -τ_L ≤ q - 2|d|.
      right
      push_neg at hQb
      rw [min_eq_right hQa.le, min_eq_right hQb.le]
      -- Need cab - cbb ≤ q - 2|d|, i.e., -τ_L ≤ q - 2|d|, i.e., q ≥ 2|d| - τ_L.
      -- From hQa: q > cab - caa - d = (|d| - τ_L) + |d| = 2|d| - τ_L. ✓
      rw [tau_L_def] at *; linarith

/-! ## Upper regime characterisation (used to lift `x0 ≤ τ_U` restriction) -/

/-- Closed form of the upper round-trip constant (paper eq:cU,
    verified algebraically by `verify.py :: c_U closed form equals eq:cU`). -/
noncomputable def cU_closed (C : CostTable) : ℝ :=
  if tau_U C + tau_L C ≥ 2 * |d C| then -tau_L C
  else if tau_U C ≥ |d C| then tau_U C - 2 * |d C|
  else -tau_U C

/-- In the upper regime (`q > τ_U`), `T C q` is independent of `q` and
    equals `cU_closed`. -/
lemma T_eq_cU_closed_of_upper
    (C : CostTable) (hd : d C < 0)
    (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C)
    {q : ℝ} (hq : tau_U C < q) :
    T C q = cU_closed C := by
  unfold T
  rw [deltaR_upper C hΔ hq]
  unfold deltaQ_next cU_closed
  have e_caa_cab := caa_sub_cab C hd
  have e_cba_cbb := cba_sub_cbb C hd
  have e_cbb_caa := cbb_sub_caa C hd
  have e_cab_caa := cab_sub_caa C hd
  have e_cbb_cba := cbb_sub_cba C hd
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd)
  by_cases h1 : tau_U C + tau_L C ≥ 2 * |d C|
  · have h_Qa : C.cab ≤ C.caa + (C.cba - C.cbb) := by linarith
    have h_Qb : C.cbb ≤ C.cba + (C.cba - C.cbb) := by linarith
    rw [min_eq_right h_Qa, min_eq_right h_Qb, if_pos h1]
    rw [tau_L_def]; ring
  · push_neg at h1
    by_cases h2 : tau_U C ≥ |d C|
    · have h_Qa : C.caa + (C.cba - C.cbb) ≤ C.cab := by linarith
      have h_Qb : C.cbb ≤ C.cba + (C.cba - C.cbb) := by linarith
      rw [min_eq_left h_Qa, min_eq_right h_Qb,
          if_neg (not_le.mpr h1), if_pos h2]
      linarith
    · push_neg at h2
      have h_Qa : C.caa + (C.cba - C.cbb) ≤ C.cab := by linarith
      have h_Qb : C.cba + (C.cba - C.cbb) ≤ C.cbb := by linarith
      rw [min_eq_left h_Qa, min_eq_left h_Qb,
          if_neg (not_le.mpr h1), if_neg (not_le.mpr h2)]
      rw [tau_U_def]; ring

/-- `c_U ≤ τ_U`: the upper round-trip constant lies at or below the upper
    threshold (so the upper regime is non-absorbing — Lemma 6.2). -/
lemma cU_closed_le_tau_U
    (C : CostTable) (hd : d C < 0) (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ : tau_L C ≤ tau_U C) :
    cU_closed C ≤ tau_U C := by
  unfold cU_closed
  have habs_pos : 0 < |d C| := abs_pos.mpr (ne_of_lt hd)
  split_ifs with h1 h2 <;> linarith

/-! ## End-to-end Theorem 6.3 (concrete, full) -/

/-- Theorem 6.3, end-to-end in Lean, for arbitrary starting point in `ℝ`.
    For `x0 > τ_U` the trajectory enters the upper regime, lands at
    `cU_closed ≤ τ_U` after one round-trip, and then follows the static
    descent argument. -/
theorem thm63_concrete
    (C : CostTable) (hd_neg : d C < 0)
    (hsum : 0 ≤ tau_U C + tau_L C)
    (hΔ   : tau_L C ≤ tau_U C)
    (x0 : ℝ) :
    ∃ N : ℕ, ∀ n, N ≤ n → (T C)^[n] x0 ≤ tau_L C := by
  by_cases hx0 : x0 ≤ tau_U C
  · exact reach_lower_regime (T C) (tau_L C) (tau_U C) |d C|
      (abs_pos.mpr (ne_of_lt hd_neg))
      (T_descent_in_interval C hd_neg hsum hΔ)
      (T_absorbs_lower    C hd_neg hsum hΔ)
      x0 hx0
  · push_neg at hx0
    -- One round-trip from the upper regime lands at `cU_closed ≤ τ_U`.
    have hT_in : T C x0 ≤ tau_U C := by
      rw [T_eq_cU_closed_of_upper C hd_neg hsum hΔ hx0]
      exact cU_closed_le_tau_U C hd_neg hsum hΔ
    obtain ⟨N, hN⟩ := reach_lower_regime (T C) (tau_L C) (tau_U C) |d C|
      (abs_pos.mpr (ne_of_lt hd_neg))
      (T_descent_in_interval C hd_neg hsum hΔ)
      (T_absorbs_lower    C hd_neg hsum hΔ)
      (T C x0) hT_in
    refine ⟨N + 1, ?_⟩
    intro n hn
    cases n with
    | zero => omega
    | succ m =>
      rw [Function.iterate_succ_apply]
      exact hN m (by omega)

end CostTable
end BpVerify
