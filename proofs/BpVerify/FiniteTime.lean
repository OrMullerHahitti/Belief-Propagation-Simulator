import Mathlib

/-!
# Finite-time reach (Theorem 6.3 inductive structure)

PAPER REFERENCES:
  * `reach_lower_regime`           ↔ Theorem 6.3 (static, Section 6).
  * `reach_lower_regime_dynamic`   ↔ Theorem 7.7 (dynamic, Section 7).

Both theorems are *abstract* — they take the algebraic descent and absorption
facts as premises and prove the inductive/counting step that turns them into
"finite-time reach". The algebraic premises themselves are verified in
`verify.py` (Z3); the counting argument is what's verified here.

Concretely, when applying these to the paper's actual dynamics, the
`h_descent` premise corresponds to Lemma 6.1 (static) / Lemma 7.6 (dynamic),
and `h_absorbs` corresponds to Theorem 5.2 Part 1 plus `c_L ≤ τ_L`.
-/

namespace BpVerify

/-- Finite-time descent under the round-trip bound.

Models Theorem 6.3 Step 1 (in-interval reach) abstractly:
* `h_descent` is Lemma 6.1: in the interval, one round-trip either enters
  the lower regime or decreases `Δ Q` by at least `2 |d|`.
* `h_absorbs` is Theorem 5.2 Part 1 + Step 2 (`c_L ≤ τ_L`): once in the
  lower regime, the trajectory stays.

Conclusion: from any starting point `≤ τ_U`, iterating `G` reaches the
lower regime in finitely many steps and stays there. -/
theorem reach_lower_regime
    (G : ℝ → ℝ) (tau_L tau_U abs_d : ℝ)
    (h_pos : 0 < abs_d)
    (h_descent : ∀ x, tau_L < x → x ≤ tau_U → G x ≤ tau_L ∨ G x ≤ x - 2 * abs_d)
    (h_absorbs : ∀ x, x ≤ tau_L → G x ≤ tau_L)
    (x0 : ℝ) (hx0 : x0 ≤ tau_U) :
    ∃ N : ℕ, ∀ n, N ≤ n → G^[n] x0 ≤ tau_L := by
  -- Key invariant: for any `n`, either we've reached the lower regime or
  -- the value has decreased by at least `2 n * abs_d`.
  have key : ∀ n : ℕ,
      G^[n] x0 ≤ tau_L ∨ G^[n] x0 ≤ x0 - 2 * (n : ℝ) * abs_d := by
    intro n
    induction n with
    | zero =>
        simp only [Function.iterate_zero, id_eq, Nat.cast_zero, mul_zero,
                   zero_mul, sub_zero]
        right; rfl
    | succ k ih =>
        rcases ih with h1 | h2
        · left
          rw [Function.iterate_succ_apply']
          exact h_absorbs _ h1
        · rw [Function.iterate_succ_apply']
          by_cases hlt : G^[k] x0 ≤ tau_L
          · left; exact h_absorbs _ hlt
          · push Not at hlt
            have h_in_U : G^[k] x0 ≤ tau_U := by
              have h_nn : (0 : ℝ) ≤ 2 * (k : ℝ) * abs_d := by positivity
              linarith
            rcases h_descent (G^[k] x0) hlt h_in_U with hL | hD
            · left; exact hL
            · right; push_cast; linarith
  -- Pick `N` large enough that `x0 - 2 * N * abs_d ≤ tau_L`.
  obtain ⟨N, hN_below⟩ : ∃ N : ℕ, G^[N] x0 ≤ tau_L := by
    have h2d : (0 : ℝ) < 2 * abs_d := by linarith
    obtain ⟨N, hN⟩ := exists_nat_gt ((x0 - tau_L) / (2 * abs_d))
    have h_lt : x0 - tau_L < (N : ℝ) * (2 * abs_d) :=
      (div_lt_iff₀ h2d).mp hN
    refine ⟨N, ?_⟩
    rcases key N with h1 | h2
    · exact h1
    · linarith
  -- Absorption keeps the trajectory below `tau_L` for all `n ≥ N`.
  refine ⟨N, ?_⟩
  intro n hn
  -- By induction on `n - N`, the iterate stays ≤ tau_L.
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  clear hn
  induction k with
  | zero => simpa using hN_below
  | succ k ih =>
      have step_eq : G^[N + (k + 1)] x0 = G (G^[N + k] x0) := by
        rw [show N + (k + 1) = (N + k) + 1 from by ring]
        exact Function.iterate_succ_apply' G (N + k) x0
      rw [step_eq]
      exact h_absorbs _ ih

end BpVerify

/-! ## Dynamic version: Theorem 7.7 inductive structure -/

namespace BpVerify

/-- Finite-time descent under a `B`-bounded dynamic perturbation
(Theorem 7.7 inductive structure).

* `h_descent` is Lemma 7.6: in the wobbled interval (endpoints shifted
  by `±B`), one round-trip either reaches the safe lower zone
  `(-∞, τ_L − B]` or decreases `ΔQ` by at least `2|d| − B > 0`.
* `h_absorbs` is the persistence step of Theorem 7.7: under the
  perturbation bound, the safe lower zone is absorbing.

Conclusion: from any `x₀ ≤ τ_U + B`, the iterate `G^[n] x₀` lands in
`(-∞, τ_L − B]` (guaranteed lower regime at the next step regardless of
the next `δ_φ`) and stays there for all subsequent `n`. -/
theorem reach_lower_regime_dynamic
    (G : ℝ → ℝ) (tau_L tau_U abs_d B : ℝ)
    (h_pos : 0 < abs_d)
    (h_B_drift : B < 2 * abs_d)
    (h_descent : ∀ x, tau_L - B < x → x ≤ tau_U + B →
                  G x ≤ tau_L - B ∨ G x ≤ x - (2 * abs_d - B))
    (h_absorbs : ∀ x, x ≤ tau_L - B → G x ≤ tau_L - B)
    (x0 : ℝ) (hx0 : x0 ≤ tau_U + B) :
    ∃ N : ℕ, ∀ n, N ≤ n → G^[n] x0 ≤ tau_L - B := by
  -- Reduce to the static `reach_lower_regime` with shifted parameters.
  have h_pos' : 0 < (2 * abs_d - B) / 2 := by linarith
  refine reach_lower_regime G (tau_L - B) (tau_U + B)
    ((2 * abs_d - B) / 2) h_pos' ?_ h_absorbs x0 hx0
  intro x hx_L hx_U
  rcases h_descent x hx_L hx_U with hL | hD
  · left; exact hL
  · right; linarith

/-! ## Sequence-varying dynamic version (faithful Theorem 7.7) -/

/-- Trajectory under a sequence of maps: `traj G x0 0 = x0`,
    `traj G x0 (n+1) = G n (traj G x0 n)`. Used to model a per-iteration
    perturbation `φ^i` where each step uses its own map `G i`. -/
def traj (G : ℕ → ℝ → ℝ) (x0 : ℝ) : ℕ → ℝ
  | 0 => x0
  | n + 1 => G n (traj G x0 n)

@[simp] lemma traj_zero (G : ℕ → ℝ → ℝ) (x0 : ℝ) : traj G x0 0 = x0 := rfl

@[simp] lemma traj_succ (G : ℕ → ℝ → ℝ) (x0 : ℝ) (n : ℕ) :
    traj G x0 (n + 1) = G n (traj G x0 n) := rfl

/-- Finite-time descent for a sequence of `B`-bounded perturbations
(Theorem 7.7, faithful to the paper's per-iteration `φ^i`).

The hypotheses are uniform across the iteration index: each `G i`
must satisfy the wobbled-interval descent and the safe-zone absorption.
Concretely, when we instantiate `G i := T_dyn (φ i)`, these uniform
hypotheses are exactly Lemma 7.6 and the absorption clause of
Theorem 7.7, both of which depend on `φ i` only through `|φ i| ≤ B`. -/
theorem reach_lower_regime_dynamic_seq
    (G : ℕ → ℝ → ℝ) (tau_L tau_U abs_d B : ℝ)
    (h_pos : 0 < abs_d)
    (h_B_drift : B < 2 * abs_d)
    (h_descent : ∀ i x, tau_L - B < x → x ≤ tau_U + B →
                  G i x ≤ tau_L - B ∨ G i x ≤ x - (2 * abs_d - B))
    (h_absorbs : ∀ i x, x ≤ tau_L - B → G i x ≤ tau_L - B)
    (x0 : ℝ) (hx0 : x0 ≤ tau_U + B) :
    ∃ N : ℕ, ∀ n, N ≤ n → traj G x0 n ≤ tau_L - B := by
  -- Invariant: at step `n`, either we've entered the safe lower zone or
  -- the value has descended by at least `n * (2|d| − B)`.
  have decr_pos : (0 : ℝ) < 2 * abs_d - B := by linarith
  have key : ∀ n : ℕ,
      traj G x0 n ≤ tau_L - B ∨
      traj G x0 n ≤ x0 - (n : ℝ) * (2 * abs_d - B) := by
    intro n
    induction n with
    | zero =>
        right
        simp
    | succ k ih =>
        rcases ih with h1 | h2
        · left
          rw [traj_succ]
          exact h_absorbs k _ h1
        · by_cases hlt : traj G x0 k ≤ tau_L - B
          · left
            rw [traj_succ]
            exact h_absorbs k _ hlt
          · push_neg at hlt
            have h_in_U : traj G x0 k ≤ tau_U + B := by
              have h_nn : (0 : ℝ) ≤ (k : ℝ) * (2 * abs_d - B) := by positivity
              linarith
            rcases h_descent k _ hlt h_in_U with hL | hD
            · left; rw [traj_succ]; exact hL
            · right
              rw [traj_succ]
              push_cast
              linarith
  -- Pick `N` large enough that `x0 - N * (2|d| - B) ≤ tau_L - B`.
  obtain ⟨N, hN_below⟩ : ∃ N : ℕ, traj G x0 N ≤ tau_L - B := by
    obtain ⟨N, hN⟩ := exists_nat_gt ((x0 - (tau_L - B)) / (2 * abs_d - B))
    have h_lt : x0 - (tau_L - B) < (N : ℝ) * (2 * abs_d - B) :=
      (div_lt_iff₀ decr_pos).mp hN
    refine ⟨N, ?_⟩
    rcases key N with h1 | h2
    · exact h1
    · linarith
  -- Absorption keeps the trajectory in the safe zone for all `n ≥ N`.
  refine ⟨N, ?_⟩
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  clear hn
  induction k with
  | zero => simpa using hN_below
  | succ k ih =>
      have step : traj G x0 (N + (k + 1)) = G (N + k) (traj G x0 (N + k)) := by
        rw [show N + (k + 1) = (N + k) + 1 from by ring]; rfl
      rw [step]
      exact h_absorbs (N + k) _ ih

end BpVerify
