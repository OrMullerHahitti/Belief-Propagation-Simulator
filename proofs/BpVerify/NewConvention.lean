import Mathlib
import BpVerify.Basic
import BpVerify.FiniteTime
import BpVerify.RoundTrip

/-!
# New (advisor / margin) convention for the two-node binary factor graph

PAPER REFERENCE: the "Effect of Splitting" rewrite that switches every message
difference to the *margin* orientation `Δ = (b-entry) − (a-entry)`.

The existing files (`Basic`, `RoundTrip`, ...) formalise the theory in the
*old* convention: `d = C'(a,a) − C'(b,b)`, `ΔQ = Q_a − Q_b`, canonical
orientation `d < 0`, and convergence into the **lower** regime.

The advisor's new convention flips the message-difference axis:
`d' = C'(b,b) − C'(a,a)`, `ΔQ = Q_b − Q_a`, canonical orientation `d' > 0`,
and convergence into the **upper** regime (row `a` wins), whose stable outgoing
difference is `C'(a,b) − C'(a,a) = B'_b − M'_a`.

This file does NOT re-derive anything from scratch.  It defines the new-convention
dynamics from the same min-sum updates and PROVES they are the negation-conjugate
of the old ones (`T_new x = − T (−x)`), then discharges the new-convention
convergence theorem from the already-verified `thm63_concrete`.  Lean therefore
*checks the entire sign mapping*: if the paper's relabelling were wrong, this
file would not compile.

Entry dictionary (Lean field ↔ paper split entry):
  `caa = M'_a`,  `cba = B'_a`,  `cab = B'_b`,  `cbb = M'_b`.
-/

namespace BpVerify
namespace CostTable

/-! ## New-convention quantities -/

/-- New drift constant `d' := C'(b,b) − C'(a,a) = M'_b − M'_a  ( = − d ). -/
def dN (C : CostTable) : ℝ := C.cbb - C.caa

/-- New upper threshold `τ'_U := C'(a,b) − C'(b,b) = B'_b − M'_b  ( = − τ_L ). -/
def tauU_N (C : CostTable) : ℝ := C.cab - C.cbb

/-- New lower threshold `τ'_L := C'(a,a) − C'(b,a) = M'_a − B'_a  ( = − τ_U ). -/
def tauL_N (C : CostTable) : ℝ := C.caa - C.cba

/-- New forward step: `ΔR = R_b − R_a` with `Q_a` normalised to `0`, `Q_b = q`. -/
def deltaR_N (C : CostTable) (q : ℝ) : ℝ :=
  min C.cab (C.cbb + q) - min C.caa (C.cba + q)

/-- New backward step: `ΔQ_next = Q_b − Q_a` with `R_a` normalised to `0`,
    `R_b = r`. -/
def deltaQ_next_N (C : CostTable) (r : ℝ) : ℝ :=
  min C.cba (C.cbb + r) - min C.caa (C.cab + r)

/-- New round-trip map on `ΔQ = Q_b − Q_a`. -/
def T_N (C : CostTable) (q : ℝ) : ℝ := deltaQ_next_N C (deltaR_N C q)

/-! ## Bridge lemmas: the new dynamics are the negation-conjugate of the old. -/

lemma dN_eq (C : CostTable) : dN C = - d C := by
  unfold dN d; ring

lemma tauU_N_eq (C : CostTable) : tauU_N C = - tau_L C := by
  unfold tauU_N tau_L Delta_b; ring

lemma tauL_N_eq (C : CostTable) : tauL_N C = - tau_U C := by
  unfold tauL_N tau_U Delta_a; ring

/-- Forward step is negation-conjugate: `ΔR_new(q) = − ΔR_old(−q)`. -/
lemma deltaR_N_eq (C : CostTable) (q : ℝ) :
    deltaR_N C q = - deltaR C (-q) := by
  unfold deltaR_N deltaR
  simp only [min_def]
  split_ifs <;> linarith

/-- Backward step is negation-conjugate: `ΔQ_next_new(r) = − ΔQ_next_old(−r)`. -/
lemma deltaQ_next_N_eq (C : CostTable) (r : ℝ) :
    deltaQ_next_N C r = - deltaQ_next C (-r) := by
  unfold deltaQ_next_N deltaQ_next
  simp only [min_def]
  split_ifs <;> linarith

/-- Round-trip is negation-conjugate: `T_new(q) = − T_old(−q)`. -/
lemma T_N_eq (C : CostTable) (q : ℝ) : T_N C q = - T C (-q) := by
  unfold T_N T
  rw [deltaR_N_eq, deltaQ_next_N_eq]
  simp

/-- Iterated conjugation: `(T_new)^[n] x = − (T_old)^[n] (−x)`. -/
lemma iterate_T_N (C : CostTable) (n : ℕ) (x : ℝ) :
    (T_N C)^[n] x = - (T C)^[n] (-x) := by
  induction n generalizing x with
  | zero => simp
  | succ k ih =>
      rw [Function.iterate_succ_apply, Function.iterate_succ_apply,
          T_N_eq, ih]
      simp

/-! ## New-convention stability: the upper regime returns `B'_b − M'_a`. -/

/-- In the new upper regime (`q > τ'_U`), row `a` wins both columns and the
    outgoing difference is `ΔR = C'(a,b) − C'(a,a) = B'_b − M'_a`, independent
    of `q`.  (Paper: the single-cycle Lemma's limit.) -/
lemma deltaR_N_upper (C : CostTable) (hΔ : tauL_N C ≤ tauU_N C)
    {q : ℝ} (hq : tauU_N C < q) :
    deltaR_N C q = C.cab - C.caa := by
  rw [deltaR_N_eq]
  have hΔ' : tau_L C ≤ tau_U C := by
    rw [tauL_N_eq, tauU_N_eq] at hΔ; linarith
  have hq' : -q ≤ tau_L C := by rw [tauU_N_eq] at hq; linarith
  rw [deltaR_lower C hΔ' hq']; ring

/-! ## New-convention convergence theorem (discharged from `thm63_concrete`). -/

/-- **Theorem (new convention).**  Under the mirror WLOG orientation
    `d' > 0`, `τ'_U + τ'_L ≤ 0`, `τ'_L ≤ τ'_U`, from any starting `ΔQ⁰ ∈ ℝ`
    the round-trip trajectory enters the **upper** regime in finite time and
    stays there:  eventually `τ'_U ≤ (T_new)^[n] x0`.

    This is the exact new-convention restatement of `thm63_concrete`; the proof
    is entirely by the negation bridge, so Lean is certifying that the paper's
    sign relabelling is sound. -/
theorem thm63_concrete_new
    (C : CostTable) (hd : 0 < dN C)
    (hsum : tauU_N C + tauL_N C ≤ 0)
    (hΔ : tauL_N C ≤ tauU_N C)
    (x0 : ℝ) :
    ∃ N : ℕ, ∀ n, N ≤ n → tauU_N C ≤ (T_N C)^[n] x0 := by
  -- Translate the new-convention hypotheses to the old-convention ones.
  have hd' : d C < 0 := by rw [dN_eq] at hd; linarith
  have hsum' : 0 ≤ tau_U C + tau_L C := by
    rw [tauU_N_eq, tauL_N_eq] at hsum; linarith
  have hΔ' : tau_L C ≤ tau_U C := by
    rw [tauU_N_eq, tauL_N_eq] at hΔ; linarith
  -- Apply the verified old theorem at starting point `−x0`.
  obtain ⟨N, hN⟩ := thm63_concrete C hd' hsum' hΔ' (-x0)
  refine ⟨N, fun n hn => ?_⟩
  have hold : (T C)^[n] (-x0) ≤ tau_L C := hN n hn
  rw [iterate_T_N, tauU_N_eq]
  linarith

/-! ## New-convention building-block lemmas (transcribed into the paper).

Each is discharged from its old-convention counterpart in `RoundTrip.lean`
through the negation bridge, so the exact formulas that appear in the paper
proofs are Lean-checked. -/

/-- `|d'| = |d|` (since `d' = −d`). -/
lemma abs_dN (C : CostTable) : |dN C| = |d C| := by rw [dN_eq, abs_neg]

/-- Lower-regime forward value (new convention): `ΔR = C'(b,b) − C'(b,a)`. -/
lemma deltaR_N_lower (C : CostTable) (hΔ : tauL_N C ≤ tauU_N C)
    {q : ℝ} (hq : q < tauL_N C) :
    deltaR_N C q = C.cbb - C.cba := by
  rw [deltaR_N_eq]
  have hΔ' : tau_L C ≤ tau_U C := by rw [tauL_N_eq, tauU_N_eq] at hΔ; linarith
  have hq' : tau_U C < -q := by rw [tauL_N_eq] at hq; linarith
  rw [deltaR_upper C hΔ' hq']; ring

/-- Transition-regime drift identity (new convention): `ΔR = ΔQ + d'`. -/
lemma deltaR_N_transition (C : CostTable)
    {q : ℝ} (hL : tauL_N C ≤ q) (hU : q < tauU_N C) :
    deltaR_N C q = q + dN C := by
  rw [deltaR_N_eq, dN_eq]
  have hL' : -q ≤ tau_U C := by rw [tauL_N_eq] at hL; linarith
  have hU' : tau_L C < -q := by rw [tauU_N_eq] at hU; linarith
  rw [deltaR_transition C hU' hL']; ring

/-- New upper round-trip constant `c'_U := −c_L` (closed form).
    Paper eq (new convention). -/
noncomputable def cU_N_closed (C : CostTable) : ℝ :=
  if tauU_N C ≤ -|dN C| then - tauU_N C
  else if tauU_N C + tauL_N C < -2 * |dN C| then tauU_N C + 2 * |dN C|
  else - tauL_N C

/-- The paper's new-convention closed form equals `−c_L` of the verified
    old-convention `cL_closed`. -/
lemma cU_N_closed_eq (C : CostTable) : cU_N_closed C = - cL_closed C := by
  unfold cU_N_closed cL_closed
  rw [abs_dN, tauU_N_eq, tauL_N_eq]
  split_ifs <;> linarith

/-- Upper regime: `T'` is constant and equals `c'_U`. -/
lemma T_N_eq_cU_N_closed_of_upper
    (C : CostTable) (hd : 0 < dN C)
    (hsum : tauU_N C + tauL_N C ≤ 0) (hΔ : tauL_N C ≤ tauU_N C)
    {q : ℝ} (hq : tauU_N C < q) :
    T_N C q = cU_N_closed C := by
  have hd' : d C < 0 := by rw [dN_eq] at hd; linarith
  have hsum' : 0 ≤ tau_U C + tau_L C := by rw [tauU_N_eq, tauL_N_eq] at hsum; linarith
  have hΔ' : tau_L C ≤ tau_U C := by rw [tauU_N_eq, tauL_N_eq] at hΔ; linarith
  have hq' : -q ≤ tau_L C := by rw [tauU_N_eq] at hq; linarith
  rw [T_N_eq, cU_N_closed_eq, T_eq_cL_closed_of_lower C hd' hsum' hΔ' hq']

/-- The upper regime is absorbing: `c'_U ≥ τ'_U`. -/
lemma cU_N_closed_ge_tauU
    (C : CostTable) (hd : 0 < dN C)
    (hsum : tauU_N C + tauL_N C ≤ 0) (hΔ : tauL_N C ≤ tauU_N C) :
    tauU_N C ≤ cU_N_closed C := by
  have hd' : d C < 0 := by rw [dN_eq] at hd; linarith
  have hsum' : 0 ≤ tau_U C + tau_L C := by rw [tauU_N_eq, tauL_N_eq] at hsum; linarith
  have hΔ' : tau_L C ≤ tau_U C := by rw [tauU_N_eq, tauL_N_eq] at hΔ; linarith
  rw [cU_N_closed_eq, tauU_N_eq]
  have := cL_closed_le_tau_L C hd' hsum' hΔ'; linarith

/-- Once `ΔQ ≥ τ'_U`, one round-trip stays `≥ τ'_U` (upper regime absorbing). -/
lemma T_N_absorbs_upper
    (C : CostTable) (hd : 0 < dN C)
    (hsum : tauU_N C + tauL_N C ≤ 0) (hΔ : tauL_N C ≤ tauU_N C) :
    ∀ q, tauU_N C ≤ q → tauU_N C ≤ T_N C q := by
  intro q hq
  have hd' : d C < 0 := by rw [dN_eq] at hd; linarith
  have hsum' : 0 ≤ tau_U C + tau_L C := by rw [tauU_N_eq, tauL_N_eq] at hsum; linarith
  have hΔ' : tau_L C ≤ tau_U C := by rw [tauU_N_eq, tauL_N_eq] at hΔ; linarith
  have hq' : -q ≤ tau_L C := by rw [tauU_N_eq] at hq; linarith
  have := T_absorbs_lower C hd' hsum' hΔ' (-q) hq'
  rw [T_N_eq, tauU_N_eq]; linarith

/-- Drift in the transition interval (new convention): one round-trip either
    enters the upper regime or increases `ΔQ` by at least `2|d'|`. -/
lemma T_N_ascent_in_interval
    (C : CostTable) (hd : 0 < dN C)
    (hsum : tauU_N C + tauL_N C ≤ 0) (hΔ : tauL_N C ≤ tauU_N C) :
    ∀ q, tauL_N C ≤ q → q < tauU_N C →
         tauU_N C ≤ T_N C q ∨ q + 2 * |dN C| ≤ T_N C q := by
  intro q hL hU
  have hd' : d C < 0 := by rw [dN_eq] at hd; linarith
  have hsum' : 0 ≤ tau_U C + tau_L C := by rw [tauU_N_eq, tauL_N_eq] at hsum; linarith
  have hΔ' : tau_L C ≤ tau_U C := by rw [tauU_N_eq, tauL_N_eq] at hΔ; linarith
  have hxL : tau_L C < -q := by rw [tauU_N_eq] at hU; linarith
  have hxU : -q ≤ tau_U C := by rw [tauL_N_eq] at hL; linarith
  rcases T_descent_in_interval C hd' hsum' hΔ' (-q) hxL hxU with h | h
  · left; rw [T_N_eq, tauU_N_eq]; linarith
  · right; rw [T_N_eq, abs_dN]; linarith

end CostTable
end BpVerify
