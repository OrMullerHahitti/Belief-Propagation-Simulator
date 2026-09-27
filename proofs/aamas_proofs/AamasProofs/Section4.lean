import Mathlib

/-!
# Section 4: the binary split cycle

This file formalizes the recurrence and the revised results in Section 4 of
`ors_revisions_shir.tex`.  The variables `Ma`, `Mb`, `Ba`, and `Bb` use the
paper's notation.  A complete pass returns a directed message to the same
copy of the split factor after four message-passing iterations.
-/

namespace BpVerify.Section4

def clip (x l h : ℝ) : ℝ := max l (min x h)

def d (Ma Mb : ℝ) : ℝ := Mb - Ma
def lowerJ (Mb Ba : ℝ) : ℝ := Mb - Ba
def upperJ (Ma Bb : ℝ) : ℝ := Bb - Ma
def lowerI (Mb Bb : ℝ) : ℝ := Mb - Bb
def upperI (Ma Ba : ℝ) : ℝ := Ba - Ma

/-- Difference sent to `X_j` for an incoming difference `q`. -/
def toJ (Ma Mb Ba Bb q : ℝ) : ℝ :=
  clip (d Ma Mb + q) (lowerJ Mb Ba) (upperJ Ma Bb)

/-- Difference sent to `X_i` for an incoming difference `q`. -/
def toI (Ma Mb Ba Bb q : ℝ) : ℝ :=
  clip (d Ma Mb + q) (lowerI Mb Bb) (upperI Ma Ba)

/-- A complete pass around the split cycle. -/
def pass (Ma Mb Ba Bb delta r : ℝ) : ℝ :=
  toJ Ma Mb Ba Bb (delta + toI Ma Mb Ba Bb r)

/-- The direct min-sum calculation for a message sent to `X_j`. -/
def directToJ (Ma Mb Ba Bb q : ℝ) : ℝ :=
  min Bb (Mb + q) - min Ma (Ba + q)

/-- The direct min-sum calculation for a message sent to `X_i`. -/
def directToI (Ma Mb Ba Bb q : ℝ) : ℝ :=
  min Ba (Mb + q) - min Ma (Bb + q)

lemma directToJ_eq_clip
    (Ma Mb Ba Bb q : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    directToJ Ma Mb Ba Bb q = toJ Ma Mb Ba Bb q := by
  simp only [directToJ, toJ, clip, d, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> linarith

lemma directToI_eq_clip
    (Ma Mb Ba Bb q : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    directToI Ma Mb Ba Bb q = toI Ma Mb Ba Bb q := by
  simp only [directToI, toI, clip, d, lowerI, upperI, max_def, min_def]
  split_ifs <;> linarith

/-- Lemma 4.3 (`Lem:cons`): both minima use the `a` row. -/
theorem selected_row_gives_upper_bound
    (Ma Mb Ba Bb Qa Qb : ℝ)
    (hcolA : Qa + Ma < Qb + Ba)
    (hcolB : Qa + Bb < Qb + Mb) :
    min (Qa + Bb) (Qb + Mb) - min (Qa + Ma) (Qb + Ba) = Bb - Ma := by
  rw [min_eq_left hcolB.le, min_eq_left hcolA.le]
  ring

/-- Equation (one-pass recurrence) in the appendix. -/
theorem one_pass_recurrence (Ma Mb Ba Bb delta r : ℝ) :
    pass Ma Mb Ba Bb delta r =
      clip (d Ma Mb + delta +
        clip (d Ma Mb + r) (lowerI Mb Bb) (upperI Ma Ba))
        (lowerJ Mb Ba) (upperJ Ma Bb) := by
  simp only [pass, toJ, toI]
  congr 1
  ring

lemma toJ_range
    (Ma Mb Ba Bb q : ℝ) (hMaMb : Ma < Mb)
    (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    lowerJ Mb Ba ≤ toJ Ma Mb Ba Bb q ∧
      toJ Ma Mb Ba Bb q ≤ upperJ Ma Bb := by
  simp only [toJ, clip, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> constructor <;> linarith

lemma pass_range
    (Ma Mb Ba Bb delta r : ℝ) (hMaMb : Ma < Mb)
    (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    lowerJ Mb Ba ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤ upperJ Ma Bb := by
  exact toJ_range Ma Mb Ba Bb _ hMaMb hMbBa hMbBb

def upperTarget (Ma Mb Ba Bb delta : ℝ) : ℝ :=
  min (upperJ Ma Bb) (upperI Ma Ba + d Ma Mb + delta)

def lowerTarget (Ma Mb Ba Bb delta : ℝ) : ℝ :=
  max (lowerJ Mb Ba) (lowerI Mb Bb + d Ma Mb + delta)

/-! ## Constant value of `Δ_{\bar R_i}`: Lemma 4.5 -/

lemma upper_pass_bounds
    (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : 2 * (Ma - Mb) < delta) :
    min (upperTarget Ma Mb Ba Bb delta) (r + 2 * d Ma Mb + delta)
        ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤ upperTarget Ma Mb Ba Bb delta := by
  simp only [pass, toJ, toI, upperTarget, clip, d, lowerJ, upperJ,
    lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

lemma lower_pass_bounds
    (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : delta < 2 * (Ma - Mb)) :
    lowerTarget Ma Mb Ba Bb delta ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤
        max (lowerTarget Ma Mb Ba Bb delta) (r + 2 * d Ma Mb + delta) := by
  simp only [pass, toJ, toI, lowerTarget, clip, d, lowerJ, upperJ,
    lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

lemma upperTarget_fixed
    (Ma Mb Ba Bb delta : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : 2 * (Ma - Mb) < delta) :
    pass Ma Mb Ba Bb delta (upperTarget Ma Mb Ba Bb delta) =
      upperTarget Ma Mb Ba Bb delta := by
  have ht := upper_pass_bounds Ma Mb Ba Bb delta
    (upperTarget Ma Mb Ba Bb delta) hMaMb hMbBa hMbBb hdelta
  have htL : lowerJ Mb Ba ≤ upperTarget Ma Mb Ba Bb delta := by
    simp only [upperTarget, lowerJ, upperJ, upperI, d, min_def]
    split_ifs <;> linarith
  have htU : upperTarget Ma Mb Ba Bb delta ≤ upperJ Ma Bb := by
    simp [upperTarget]
  rcases ht with ⟨hlo, hhi⟩
  have hc : 0 < 2 * d Ma Mb + delta := by
    simp only [d]; linarith
  have hm : min (upperTarget Ma Mb Ba Bb delta)
      (upperTarget Ma Mb Ba Bb delta + 2 * d Ma Mb + delta) =
      upperTarget Ma Mb Ba Bb delta := by
    rw [min_eq_left]; linarith
  rw [hm] at hlo
  linarith

lemma lowerTarget_fixed
    (Ma Mb Ba Bb delta : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : delta < 2 * (Ma - Mb)) :
    pass Ma Mb Ba Bb delta (lowerTarget Ma Mb Ba Bb delta) =
      lowerTarget Ma Mb Ba Bb delta := by
  have htL : lowerJ Mb Ba ≤ lowerTarget Ma Mb Ba Bb delta := by
    simp [lowerTarget]
  have htU : lowerTarget Ma Mb Ba Bb delta ≤ upperJ Ma Bb := by
    simp only [lowerTarget, lowerJ, upperJ, lowerI, d, max_def]
    split_ifs <;> linarith
  have ht := lower_pass_bounds Ma Mb Ba Bb delta
    (lowerTarget Ma Mb Ba Bb delta) hMaMb hMbBa hMbBb hdelta
  rcases ht with ⟨hlo, hhi⟩
  have hc : 2 * d Ma Mb + delta < 0 := by
    simp only [d]; linarith
  have hm : max (lowerTarget Ma Mb Ba Bb delta)
      (lowerTarget Ma Mb Ba Bb delta + 2 * d Ma Mb + delta) =
      lowerTarget Ma Mb Ba Bb delta := by
    rw [max_eq_left]; linarith
  rw [hm] at hhi
  linarith

/-! Generic finite-time stabilization lemmas used for Lemma 4.5. -/

theorem finite_stabilization_up
    (f : ℝ → ℝ) (T c x : ℝ)
    (hc : 0 < c) (hx : x ≤ T)
    (hstep : ∀ y, min T (y + c) ≤ f y ∧ f y ≤ T)
    (hfix : f T = T) :
    ∃ N : ℕ, ∀ n, N ≤ n → (f^[n]) x = T := by
  have key : ∀ n : ℕ,
      (((f^[n]) x = T) ∨ x + (n : ℝ) * c ≤ (f^[n]) x) ∧
        (f^[n]) x ≤ T := by
    intro n
    induction n with
    | zero =>
        constructor
        · right; simp
        · simpa using hx
    | succ k ih =>
        rw [Function.iterate_succ_apply']
        rcases ih with ⟨hEq | hGrow, hUpper⟩
        · rw [hEq, hfix]
          exact ⟨Or.inl rfl, le_rfl⟩
        · have hs := hstep ((f^[k]) x)
          constructor
          · by_cases hcap : T ≤ (f^[k]) x + c
            · left
              have hm : min T ((f^[k]) x + c) = T := min_eq_left hcap
              rw [hm] at hs
              linarith
            · right
              push Not at hcap
              have hm : min T ((f^[k]) x + c) = (f^[k]) x + c :=
                min_eq_right hcap.le
              rw [hm] at hs
              push_cast
              linarith
          · exact hs.2
  obtain ⟨N, hN⟩ := exists_nat_gt ((T - x) / c)
  have hgap : T < x + (N : ℝ) * c := by
    have hmul : T - x < (N : ℝ) * c := (div_lt_iff₀ hc).mp hN
    linarith
  have hreach : (f^[N]) x = T := by
    rcases key N with ⟨hEq | hGrow, hUpper⟩
    · exact hEq
    · linarith
  refine ⟨N, ?_⟩
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  clear hn
  induction k with
  | zero => simpa using hreach
  | succ k ih =>
      rw [show N + (k + 1) = (N + k) + 1 by omega]
      rw [Function.iterate_succ_apply', ih, hfix]

theorem finite_stabilization_down
    (f : ℝ → ℝ) (T c x : ℝ)
    (hc : 0 < c) (hx : T ≤ x)
    (hstep : ∀ y, T ≤ f y ∧ f y ≤ max T (y - c))
    (hfix : f T = T) :
    ∃ N : ℕ, ∀ n, N ≤ n → (f^[n]) x = T := by
  have key : ∀ n : ℕ,
      (((f^[n]) x = T) ∨ (f^[n]) x ≤ x - (n : ℝ) * c) ∧
        T ≤ (f^[n]) x := by
    intro n
    induction n with
    | zero =>
        constructor
        · right; simp
        · simpa using hx
    | succ k ih =>
        rw [Function.iterate_succ_apply']
        rcases ih with ⟨hEq | hDrop, hLower⟩
        · rw [hEq, hfix]
          exact ⟨Or.inl rfl, le_rfl⟩
        · have hs := hstep ((f^[k]) x)
          constructor
          · by_cases hcap : (f^[k]) x - c ≤ T
            · left
              have hm : max T ((f^[k]) x - c) = T := max_eq_left hcap
              rw [hm] at hs
              linarith
            · right
              push Not at hcap
              have hm : max T ((f^[k]) x - c) = (f^[k]) x - c :=
                max_eq_right hcap.le
              rw [hm] at hs
              push_cast
              linarith
          · exact hs.1
  obtain ⟨N, hN⟩ := exists_nat_gt ((x - T) / c)
  have hgap : x - (N : ℝ) * c < T := by
    have hmul : x - T < (N : ℝ) * c := (div_lt_iff₀ hc).mp hN
    linarith
  have hreach : (f^[N]) x = T := by
    rcases key N with ⟨hEq | hDrop, hLower⟩
    · exact hEq
    · linarith
  refine ⟨N, ?_⟩
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  clear hn
  induction k with
  | zero => simpa using hreach
  | succ k ih =>
      rw [show N + (k + 1) = (N + k) + 1 by omega]
      rw [Function.iterate_succ_apply', ih, hfix]

theorem constant_input_converges_upper
    (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : 2 * (Ma - Mb) < delta) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((pass Ma Mb Ba Bb delta)^[n]) r = upperTarget Ma Mb Ba Bb delta := by
  let f : ℝ → ℝ := pass Ma Mb Ba Bb delta
  let T : ℝ := upperTarget Ma Mb Ba Bb delta
  let c : ℝ := 2 * d Ma Mb + delta
  have hc : 0 < c := by simp only [c, d]; linarith
  have hstep : ∀ y, min T (y + c) ≤ f y ∧ f y ≤ T := by
    intro y
    dsimp [f, T, c]
    convert upper_pass_bounds Ma Mb Ba Bb delta y hMaMb hMbBa hMbBb hdelta using 1 <;>
      ring_nf
  have hfix : f T = T := by
    exact upperTarget_fixed Ma Mb Ba Bb delta hMaMb hMbBa hMbBb hdelta
  by_cases hr : r ≤ T
  · exact finite_stabilization_up f T c r hc hr hstep hfix
  · push Not at hr
    have hs := hstep r
    have hm : min T (r + c) = T := min_eq_left (by linarith)
    rw [hm] at hs
    have hfr : f r = T := le_antisymm hs.2 hs.1
    refine ⟨1, ?_⟩
    intro n hn
    cases n with
    | zero => omega
    | succ k =>
      clear hn
      induction k with
      | zero => simpa [Function.iterate_succ_apply'] using hfr
      | succ k ih =>
        rw [Function.iterate_succ_apply', ih]
        simpa [f, T] using hfix

theorem constant_input_converges_lower
    (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : delta < 2 * (Ma - Mb)) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((pass Ma Mb Ba Bb delta)^[n]) r = lowerTarget Ma Mb Ba Bb delta := by
  let f : ℝ → ℝ := pass Ma Mb Ba Bb delta
  let T : ℝ := lowerTarget Ma Mb Ba Bb delta
  let c : ℝ := -(2 * d Ma Mb + delta)
  have hc : 0 < c := by simp only [c, d]; linarith
  have hstep : ∀ y, T ≤ f y ∧ f y ≤ max T (y - c) := by
    intro y
    have hs := lower_pass_bounds Ma Mb Ba Bb delta y hMaMb hMbBa hMbBb hdelta
    dsimp [f, T, c]
    convert hs using 1 <;> ring_nf
  have hfix : f T = T := by
    exact lowerTarget_fixed Ma Mb Ba Bb delta hMaMb hMbBa hMbBb hdelta
  by_cases hr : T ≤ r
  · exact finite_stabilization_down f T c r hc hr hstep hfix
  · push Not at hr
    have hs := hstep r
    have hm : max T (r - c) = T := max_eq_left (by linarith)
    rw [hm] at hs
    have hfr : f r = T := le_antisymm hs.2 hs.1
    refine ⟨1, ?_⟩
    intro n hn
    cases n with
    | zero => omega
    | succ k =>
      clear hn
      induction k with
      | zero => simpa [Function.iterate_succ_apply'] using hfr
      | succ k ih =>
        rw [Function.iterate_succ_apply', ih]
        simpa [f, T] using hfix

def neutralLower (Ma Mb Ba Bb : ℝ) : ℝ :=
  max (lowerJ Mb Ba) (Ma - Bb)

def neutralUpper (Ma Mb Ba Bb : ℝ) : ℝ :=
  min (upperJ Ma Bb) (Ba - Mb)

lemma clip_range (x l h : ℝ) (hlh : l ≤ h) :
    l ≤ clip x l h ∧ clip x l h ≤ h := by
  simp only [clip, max_def, min_def]
  split_ifs <;> constructor <;> linarith

lemma clip_idempotent (x l h : ℝ) (hlh : l ≤ h) :
    clip (clip x l h) l h = clip x l h := by
  simp only [clip, max_def, min_def]
  split_ifs <;> linarith

lemma neutralLower_le_neutralUpper
    (Ma Mb Ba Bb : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    neutralLower Ma Mb Ba Bb ≤ neutralUpper Ma Mb Ba Bb := by
  simp only [neutralLower, neutralUpper, lowerJ, upperJ, max_def, min_def]
  split_ifs <;> linarith

lemma neutral_pass_eq_clip
    (Ma Mb Ba Bb r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    pass Ma Mb Ba Bb (2 * (Ma - Mb)) r =
      clip r (neutralLower Ma Mb Ba Bb) (neutralUpper Ma Mb Ba Bb) := by
  simp only [neutralLower, neutralUpper, pass, toJ, toI, clip, d,
    lowerJ, upperJ, lowerI, upperI, max_def, min_def]
  split_ifs <;> linarith

theorem neutral_pass_projects_and_fixes
    (Ma Mb Ba Bb r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    neutralLower Ma Mb Ba Bb ≤
        pass Ma Mb Ba Bb (2 * (Ma - Mb)) r ∧
      pass Ma Mb Ba Bb (2 * (Ma - Mb)) r ≤
        neutralUpper Ma Mb Ba Bb ∧
      pass Ma Mb Ba Bb (2 * (Ma - Mb))
          (pass Ma Mb Ba Bb (2 * (Ma - Mb)) r) =
        pass Ma Mb Ba Bb (2 * (Ma - Mb)) r := by
  have hLU := neutralLower_le_neutralUpper Ma Mb Ba Bb hMaMb hMbBa hMbBb
  rw [neutral_pass_eq_clip Ma Mb Ba Bb r hMaMb hMbBa hMbBb]
  have hrange := clip_range r (neutralLower Ma Mb Ba Bb)
    (neutralUpper Ma Mb Ba Bb) hLU
  refine ⟨hrange.1, hrange.2, ?_⟩
  rw [neutral_pass_eq_clip Ma Mb Ba Bb _ hMaMb hMbBa hMbBb]
  exact clip_idempotent r (neutralLower Ma Mb Ba Bb)
    (neutralUpper Ma Mb Ba Bb) hLU

/-! ## No outside input: Lemma 4.4 and the alternative bounder ordering -/

theorem no_input_converges_when_Bb_lt_Ba
    (Ma Mb Ba Bb r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hBbBa : Bb < Ba) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((pass Ma Mb Ba Bb 0)^[n]) r = upperJ Ma Bb := by
  have hconv := constant_input_converges_upper Ma Mb Ba Bb 0 r
    hMaMb hMbBa hMbBb (by linarith)
  have ht : upperTarget Ma Mb Ba Bb 0 = upperJ Ma Bb := by
    simp only [upperTarget, upperJ, upperI, d, add_zero]
    rw [min_eq_left]
    linarith
  simpa [ht] using hconv

theorem no_input_converges_when_Ba_lt_Bb
    (Ma Mb Ba Bb r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hBaBb : Ba < Bb) :
    ∃ N : ℕ, ∀ n, N ≤ n →
      ((pass Ma Mb Ba Bb 0)^[n]) r =
        min (Bb - Ma) ((Ba - Ma) + (Mb - Ma)) := by
  simpa [upperTarget, upperJ, upperI, d] using
    constant_input_converges_upper Ma Mb Ba Bb 0 r
      hMaMb hMbBa hMbBb (by linarith)

theorem toI_at_alternative_limit
    (Ma Mb Ba Bb : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hBaBb : Ba < Bb) :
    toI Ma Mb Ba Bb
      (min (Bb - Ma) ((Ba - Ma) + (Mb - Ma))) = Ba - Ma := by
  simp only [toI, clip, d, lowerI, upperI, max_def, min_def]
  split_ifs <;> linarith

/-! ## Flipping threshold: Lemma 4.7 -/

def flippingThreshold (Ma Mb Ba Bb r : ℝ) : ℝ :=
  -d Ma Mb - toI Ma Mb Ba Bb r

theorem initial_flippingThreshold
    (Ma Mb Ba Bb : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    flippingThreshold Ma Mb Ba Bb 0 = 2 * (Ma - Mb) := by
  simp only [flippingThreshold, toI, clip, d, lowerI, upperI,
    max_def, min_def]
  split_ifs <;> linarith

theorem flippingThreshold_constant_of_message_constant
    (Ma Mb Ba Bb r r' : ℝ) (h : r' = r) :
    flippingThreshold Ma Mb Ba Bb r' = flippingThreshold Ma Mb Ba Bb r := by
  rw [h]

theorem flippingThreshold_upper_change
    (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hBbBa : Bb < Ba)
    (hdelta : 2 * (Ma - Mb) < delta)
    (hrL : lowerJ Mb Ba ≤ r) (hrU : r ≤ upperJ Ma Bb) :
    flippingThreshold Ma Mb Ba Bb r - delta - 2 * d Ma Mb
        ≤ flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r) ∧
      flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r)
        ≤ flippingThreshold Ma Mb Ba Bb r := by
  simp only [lowerJ, upperJ] at hrL hrU
  simp only [flippingThreshold, pass, toJ, toI, clip, d, lowerJ, upperJ,
    lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

theorem flippingThreshold_lower_change
    (Ma Mb Ba Bb delta r : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hBbBa : Bb < Ba)
    (hdelta : delta < 2 * (Ma - Mb))
    (hrL : lowerJ Mb Ba ≤ r) (hrU : r ≤ upperJ Ma Bb) :
    flippingThreshold Ma Mb Ba Bb r
        ≤ flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r) ∧
      flippingThreshold Ma Mb Ba Bb (pass Ma Mb Ba Bb delta r)
        ≤ flippingThreshold Ma Mb Ba Bb r - delta - 2 * d Ma Mb := by
  simp only [lowerJ, upperJ] at hrL hrU
  simp only [flippingThreshold, pass, toJ, toI, clip, d, lowerJ, upperJ,
    lowerI, upperI, max_def, min_def]
  split_ifs <;> constructor <;> linarith

/-! ## Varying `Δ_{\bar R_i}`: finite-time arrival, Theorem 4.8 -/

def trajectory (Ma Mb Ba Bb : ℝ) (delta : ℕ → ℝ) (r0 : ℝ) : ℕ → ℝ
  | 0 => r0
  | n + 1 => pass Ma Mb Ba Bb (delta n) (trajectory Ma Mb Ba Bb delta r0 n)

@[simp] lemma trajectory_zero (Ma Mb Ba Bb : ℝ) (delta : ℕ → ℝ) (r0 : ℝ) :
    trajectory Ma Mb Ba Bb delta r0 0 = r0 := rfl

@[simp] lemma trajectory_succ (Ma Mb Ba Bb : ℝ) (delta : ℕ → ℝ)
    (r0 : ℝ) (n : ℕ) :
    trajectory Ma Mb Ba Bb delta r0 (n + 1) =
      pass Ma Mb Ba Bb (delta n) (trajectory Ma Mb Ba Bb delta r0 n) := rfl

lemma upper_progress_step
    (Ma Mb Ba Bb delta eps r : ℝ)
    (hMaMb : Ma < Mb) (hMbBb : Mb < Bb) (hBbBa : Bb < Ba)
    (heps : 0 < eps)
    (hgrowth : eps ≤ delta + 2 * d Ma Mb)
    (hclip : Bb - Ba - d Ma Mb ≤ delta)
    (hr : -d Ma Mb ≤ r) :
    min (upperJ Ma Bb) (r + eps) ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤ upperJ Ma Bb := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI,
    max_def, min_def] at *
  split_ifs <;> constructor <;> linarith

lemma lower_progress_step
    (Ma Mb Ba Bb delta eps r : ℝ)
    (hMaMb : Ma < Mb) (hMbBb : Mb < Bb) (hBbBa : Bb < Ba)
    (heps : 0 < eps)
    (hgrowth : delta + 2 * d Ma Mb ≤ -eps)
    (hclip : delta ≤ Bb - Ba - d Ma Mb)
    (hr : r ≤ d Ma Mb) :
    lowerJ Mb Ba ≤ pass Ma Mb Ba Bb delta r ∧
      pass Ma Mb Ba Bb delta r ≤ max (lowerJ Mb Ba) (r - eps) := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI,
    max_def, min_def] at *
  split_ifs <;> constructor <;> linarith

lemma first_message_upper_seed
    (Ma Mb Ba Bb delta q : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : 2 * (Ma - Mb) < delta)
    (hq : q = 0 ∨ q = delta ∨ q = d Ma Mb + delta) :
    -d Ma Mb < toJ Ma Mb Ba Bb q ∧
      toJ Ma Mb Ba Bb q ≤ upperJ Ma Bb := by
  rcases hq with rfl | rfl | rfl <;>
    simp only [toJ, clip, d, lowerJ, upperJ, max_def, min_def] <;>
    split_ifs <;> constructor <;> linarith

lemma first_message_lower_seed
    (Ma Mb Ba Bb delta q : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb)
    (hdelta : delta < 2 * (Ma - Mb))
    (hq : q = 0 ∨ q = delta ∨ q = d Ma Mb + delta) :
    lowerJ Mb Ba ≤ toJ Ma Mb Ba Bb q ∧
      toJ Ma Mb Ba Bb q ≤ d Ma Mb := by
  rcases hq with rfl | rfl | rfl <;>
    simp only [toJ, clip, d, lowerJ, upperJ, max_def, min_def] <;>
    split_ifs <;> constructor <;> linarith

theorem varying_input_reaches_upper
    (Ma Mb Ba Bb eps r0 : ℝ) (delta : ℕ → ℝ)
    (hMaMb : Ma < Mb) (hMbBb : Mb < Bb) (hBbBa : Bb < Ba)
    (heps : 0 < eps)
    (hgrowth : ∀ k, eps ≤ delta k + 2 * d Ma Mb)
    (hclip : ∀ k, Bb - Ba - d Ma Mb ≤ delta k)
    (hr0L : -d Ma Mb ≤ r0) (hr0U : r0 ≤ upperJ Ma Bb) :
    ∀ n : ℕ,
      upperJ Ma Bb + d Ma Mb ≤ (n : ℝ) * eps →
      trajectory Ma Mb Ba Bb delta r0 n = upperJ Ma Bb := by
  have key : ∀ n : ℕ,
      (trajectory Ma Mb Ba Bb delta r0 n = upperJ Ma Bb ∨
        -d Ma Mb + (n : ℝ) * eps ≤ trajectory Ma Mb Ba Bb delta r0 n) ∧
      trajectory Ma Mb Ba Bb delta r0 n ≤ upperJ Ma Bb := by
    intro n
    induction n with
    | zero =>
        constructor
        · right; simpa using hr0L
        · simpa using hr0U
    | succ k ih =>
        rw [trajectory_succ]
        rcases ih with ⟨hEq | hGrow, hUpper⟩
        · have hs := upper_progress_step Ma Mb Ba Bb (delta k) eps
            (upperJ Ma Bb) hMaMb hMbBb hBbBa heps
            (hgrowth k) (hclip k) (by
              simp only [d, upperJ]; linarith)
          have hm : min (upperJ Ma Bb) (upperJ Ma Bb + eps) =
              upperJ Ma Bb := min_eq_left (by linarith)
          rw [hm] at hs
          rw [hEq]
          exact ⟨Or.inl (le_antisymm hs.2 hs.1), hs.2⟩
        · have hr : -d Ma Mb ≤ trajectory Ma Mb Ba Bb delta r0 k := by
            have hn : 0 ≤ (k : ℝ) * eps := by positivity
            linarith
          have hs := upper_progress_step Ma Mb Ba Bb (delta k) eps
            (trajectory Ma Mb Ba Bb delta r0 k) hMaMb hMbBb hBbBa
            heps (hgrowth k) (hclip k) hr
          constructor
          · by_cases hcap : upperJ Ma Bb ≤
                trajectory Ma Mb Ba Bb delta r0 k + eps
            · left
              have hm : min (upperJ Ma Bb)
                  (trajectory Ma Mb Ba Bb delta r0 k + eps) =
                  upperJ Ma Bb := min_eq_left hcap
              rw [hm] at hs
              exact le_antisymm hs.2 hs.1
            · right
              push Not at hcap
              have hm : min (upperJ Ma Bb)
                  (trajectory Ma Mb Ba Bb delta r0 k + eps) =
                  trajectory Ma Mb Ba Bb delta r0 k + eps :=
                min_eq_right hcap.le
              rw [hm] at hs
              push_cast
              linarith
          · exact hs.2
  intro n hn
  rcases key n with ⟨hEq | hGrow, hUpper⟩
  · exact hEq
  · linarith

theorem varying_input_reaches_lower
    (Ma Mb Ba Bb eps r0 : ℝ) (delta : ℕ → ℝ)
    (hMaMb : Ma < Mb) (hMbBb : Mb < Bb) (hBbBa : Bb < Ba)
    (heps : 0 < eps)
    (hgrowth : ∀ k, delta k + 2 * d Ma Mb ≤ -eps)
    (hclip : ∀ k, delta k ≤ Bb - Ba - d Ma Mb)
    (hr0L : lowerJ Mb Ba ≤ r0) (hr0U : r0 ≤ d Ma Mb) :
    ∀ n : ℕ,
      d Ma Mb - lowerJ Mb Ba ≤ (n : ℝ) * eps →
      trajectory Ma Mb Ba Bb delta r0 n = lowerJ Mb Ba := by
  have key : ∀ n : ℕ,
      (trajectory Ma Mb Ba Bb delta r0 n = lowerJ Mb Ba ∨
        trajectory Ma Mb Ba Bb delta r0 n ≤ d Ma Mb - (n : ℝ) * eps) ∧
      lowerJ Mb Ba ≤ trajectory Ma Mb Ba Bb delta r0 n := by
    intro n
    induction n with
    | zero =>
        constructor
        · right; simpa using hr0U
        · simpa using hr0L
    | succ k ih =>
        rw [trajectory_succ]
        rcases ih with ⟨hEq | hDrop, hLower⟩
        · have hs := lower_progress_step Ma Mb Ba Bb (delta k) eps
            (lowerJ Mb Ba) hMaMb hMbBb hBbBa heps
            (hgrowth k) (hclip k) (by
              simp only [d, lowerJ]; linarith)
          have hm : max (lowerJ Mb Ba) (lowerJ Mb Ba - eps) =
              lowerJ Mb Ba := max_eq_left (by linarith)
          rw [hm] at hs
          rw [hEq]
          exact ⟨Or.inl (le_antisymm hs.2 hs.1), hs.1⟩
        · have hr : trajectory Ma Mb Ba Bb delta r0 k ≤ d Ma Mb := by
            have hn : 0 ≤ (k : ℝ) * eps := by positivity
            linarith
          have hs := lower_progress_step Ma Mb Ba Bb (delta k) eps
            (trajectory Ma Mb Ba Bb delta r0 k) hMaMb hMbBb hBbBa
            heps (hgrowth k) (hclip k) hr
          constructor
          · by_cases hcap : trajectory Ma Mb Ba Bb delta r0 k - eps ≤
                lowerJ Mb Ba
            · left
              have hm : max (lowerJ Mb Ba)
                  (trajectory Ma Mb Ba Bb delta r0 k - eps) =
                  lowerJ Mb Ba := max_eq_left hcap
              rw [hm] at hs
              exact le_antisymm hs.2 hs.1
            · right
              push Not at hcap
              have hm : max (lowerJ Mb Ba)
                  (trajectory Ma Mb Ba Bb delta r0 k - eps) =
                  trajectory Ma Mb Ba Bb delta r0 k - eps :=
                max_eq_right hcap.le
              rw [hm] at hs
              push_cast
              linarith
          · exact hs.1
  intro n hn
  rcases key n with ⟨hEq | hDrop, hLower⟩
  · exact hEq
  · linarith

/-! ## Exact persistence thresholds: Theorem 4.9 -/

theorem upper_persistence_iff
    (Ma Mb Ba Bb delta : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    pass Ma Mb Ba Bb delta (upperJ Ma Bb) = upperJ Ma Bb ↔
      max (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) ≤ delta := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI,
    max_def, min_def]
  split_ifs <;> constructor <;> intro h <;> linarith

theorem lower_persistence_iff
    (Ma Mb Ba Bb delta : ℝ)
    (hMaMb : Ma < Mb) (hMbBa : Mb < Ba) (hMbBb : Mb < Bb) :
    pass Ma Mb Ba Bb delta (lowerJ Mb Ba) = lowerJ Mb Ba ↔
      delta ≤ min (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) := by
  simp only [pass, toJ, toI, clip, d, lowerJ, upperJ, lowerI, upperI,
    max_def, min_def]
  split_ifs <;> constructor <;> intro h <;> linarith

theorem upper_threshold_equivalent_form (Ma Mb Ba Bb : ℝ) :
    max (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) =
      -min (2 * d Ma Mb) ((Ba - Bb) + d Ma Mb) := by
  simp only [d, max_def, min_def]
  split_ifs <;> linarith

theorem lower_threshold_equivalent_form (Ma Mb Ba Bb : ℝ) :
    min (-2 * d Ma Mb) (Bb - Ba - d Ma Mb) =
      -max (2 * d Ma Mb) ((Ba - Bb) + d Ma Mb) := by
  simp only [d, max_def, min_def]
  split_ifs <;> linarith

end BpVerify.Section4
