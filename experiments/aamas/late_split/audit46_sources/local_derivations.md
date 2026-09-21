# audit46 — local derivations (my own hand check, 2026-09-21)

Scope: what each local result assumes, whether the written proof uses anything it does not state,
and whether the late-split dense case (pairwise, synchronous, domain 10-20, symmetric split applied
mid-run from a warm state, no damping after the split, float64) meets the assumptions.
Line-by-line proof checking of the selection-rule chain is ticket #48, not this ticket.

Sources read in full: analysis_oscillation/oscillation_section.tex (Rev 3, 567 lines),
analysis_oscillation/why_two_route_oscillation.md (231 lines, has retraction banner),
experiments/aamas/late_split/OSCILLATION_MECHANISM.md (230 lines),
experiments/aamas/late_split/oscillation_explanation_sources.md (43 lines),
publish/sec4_effect_of_splitting.tex (355 lines).

## A. Paper Section 4 (publish/sec4_effect_of_splitting.tex)

A1. thm:stability (l.73), thm:absorbing (l.104), lem:drift (l.138), thm:convergence (l.175).
- Assumes: ONE pairwise factor, split in two equal halves, two variables, TWO values per domain,
  undamped, the two variables have no other neighbours (isolated 4-edge cycle X1-F'-X2-F''-X1).
- Initial difference is a free real parameter (l.55) -> these results do NOT need zero start.
- Index convention: stamps count message hops around the cycle (l.9), not engine iterations.
- Object: message DIFFERENCES (scalar). Conclusion: reaches one regime in finitely many round trips and stays.
- Late-split dense domain-20: does not apply (binary scalar; isolated cycle).

A2. thm:dynstability (l.228), lem:dyndrift (l.261), thm:dyncvg (l.283).
- Same binary single-factor setting; the rest of the graph is modelled as a time-varying unary
  term phi^t on X1 ONLY; the backward step uses the bare table (l.277, l.299) = X2 has no other neighbours.
- Needs |delta_phi^t| <= B < min(2|d|, -(tau_U+tau_L)) for ALL t (l.284). Remark l.308 says the bound is sharp.
- In a dense graph the external term is the sum of differences from ~2(deg-1) other copies, while 2|d|
  comes from one half table. oscillation_section.tex l.128-130 itself says the bound "does not hold everywhere".
- Late-split dense domain-20: does not apply as a theorem. Usable only as the binary picture of
  "row forwarding" (what a locked regime means).

A3. prop:damped (l.321): conditional on staying in the regime; text l.332 says the closed induction is future work.
A4. prop:asym (l.336): binary; asymmetric split; not the 0.5/0.5 case.

## B. analysis_oscillation/oscillation_section.tex (Rev 3)

B1. lem:clones (l.102) Clone synchronization.
- States: symmetric split + ZERO-initialised messages => two copies send identical R, and X_i sends identical Q to both, every iteration.
- Proof (l.108-114): induction. The inductive step uses only "the two copies' messages were equal at t-1" and "equal tables".
  Zero is used only for the base case.
- Consequence for late split: the lemma AS STATED does not cover a warm start. The same induction goes through
  from the split iteration on IF the two copies start in equal state. That is a code fact (hand-over rule), to be
  taken from the code audit, plus a float caveat: Q to F' and Q to F'' sum the same values but maybe in a
  different order; float addition is not associative, so bitwise equality depends on how compute_Q sums.
- Status: sound in exact arithmetic under its stated assumptions; for late split = "narrow: replace base case,
  check on the code and on the saved trace".

B2. "stable active minimizer" definition (l.88-100).
- Same minimiser u* of C'(u,v)+Q(u) for EVERY v => R = row u* of C' + constant. Any domain size. Exact.
- Ties in the inner minimisation do not hurt (the min value is what is sent).
- Binary case = "in a regime" of A1.

B3. lem:decode (l.157). Needs: clone sync + ALL messages into X_i committed at t. Then decoded value =
  argmin_v [phi_i(v) + sum_j C_ij(v, u*_j)]. Proof is a direct sum of half rows. Sound as a conditional statement.
  Needs clone sync so both halves forward the SAME row.

B4. lem:sibling (l.177). Needs: B3's hypothesis at t, commitment of X_i's outgoing message at t+1, UNIQUE argmins.
  Key step: evaluate the t+1 minimisation at v = u*_j^t so the two half costs add to the full C_ij.
  Time indices are consistent with the engine convention (Q^{t+1} from R^t; decode at t uses R^t).
  Orientation consistent with rem:conv. I found no gap on a careful read; full check belongs to #48.

B5. cor:rule (l.205). Needs FULL commitment: every variable, every incident factor, every t >= t0.
  The tex itself reports this premise is only partly true on benchmarks (l.130-142: 0.73-0.92 of messages on
  domain-10 random; 0.03-0.36 on colouring; can hold on one parity only). All of those measurements are
  zero-start runs, domain 10 / colouring / binary, n = 30-50. NONE is a late-split or domain-20 run.

B6. thm:tworoutes (l.251). Hypothesis is on the DECODED SEQUENCE only: it follows the synchronous local
  selection recursion for all t >= t0, with unique argmins. Conclusion: cost_2 non-increasing, period 1 or 2.
  Proof checked by hand: separability in z, swap symmetry, finiteness. Sound. No assumption on initialisation,
  topology, domain size, splitting or damping. Pairwise shared tables + unary only.
  - It is a theorem about a map on the finite set of assignments. The hypothesis is what makes the decoded
    sequence a closed finite-state system. Without it, nothing follows (see sources note sec.5 bullet 3).
  - Contrapositive, used as a CHECK not a new claim: a run whose decoded tail has period > 2 must violate the
    recursion (or uniqueness) at least once per period. Seed 0 (42-update cycle) therefore violates it.
    Where and how often is a measurable quantity for #50.
  - Uniqueness: "as the tie-breaking phi_i ensure" is an exact-arithmetic statement. In float64 with raw
    beliefs ~1e13 (normalised only every 6 updates) the spacing is ~0.004 while phi <= 0.01 (from my notes of
    the seed-0 diagnosis; to be confirmed by the code audit). So uniqueness is numerically fragile in native runs.
  - Evidence that the recursion holds: 32 audited period-2 runs, sampled every 7th transition of a
    100-iteration tail (l.225-233). Zero-start only.

B7. cor:localmin (l.275): follows from B6. cor:bipartite (l.325): bipartite graphs only -> not for dense random graphs.

B8. lem:ema (l.393): algebra, checked by hand: fixed alternating pair (u+lv)/(1+l), (v+lu)/(1+l), swing factor (1-l)/(1+l).
  Assumes damping formula m = l*m_old_sent + (1-l)*m_new. In late split, damping is 0.9 before the split and 0 after:
  the lemma speaks only about the pre-split phase.

B9. thm:lambdastar (l.408): BINARY domains, fixed fully committed period-2 pattern. Not for domain 10-20.
  The text after it (l.448-466) lists what it does not say.
B10. prop:kn (l.468): binary K_n, zero start, exact symmetry. Role: counterexample to "lambda=0.9 always works".

B11. Empirical statements in B (census 276 runs; 274 period 1/2, two longer: 16 and 42; commitment fractions;
  32 audited; damping grid). Limits stated in the tex: window = last 150 of 700, periods <= 64, ASSIGNMENT
  periodicity only. All zero-start. Provenance vs the compute_R axis fix (2026-09-09): to be settled by the code audit.
  Note only: the census has one period-42 run and seed 0 has a 42-update cycle. Same number, no known link.

## C. experiments/aamas/late_split/OSCILLATION_MECHANISM.md

C1. sec.1-2 equations (belief, Q, R, readout) — must match the code (code audit).
C2. sec.3: update map is continuous piecewise affine (sums and minima of affine functions). True. "Need not contract": true
  as a statement of what is NOT guaranteed.
C3. sec.4 envelope condition: b(v)-b(a) > eps_v+eps_a => v never beats a. Trivially sound; the bounds eps must be
  established separately (the note says so).
C4. sec.4 table-difference bound: min_w[C(v,w)-C(a,w)] <= R(v)-R(a) <= max_w[C(v,w)-C(a,w)]. Checked by hand:
  take the minimiser for a, bound C(v,.) by C(a,.)+max diff, minimise. Exact arithmetic, any finite Q, any
  initialisation, any topology, any domain. For a split copy use C/2. => APPLIES to late split as stated.
  Whether it excludes any value in the dense domain-20 instance is unmeasured (it is a static calculation on the tables).
C5. sec.5 constructed example (two variables, 3 values, one constraint split in two, warm start R=[0,1,10]).
  Re-derived by hand: T([0,1,10]) = [1,0,10], T([1,0,10]) = [0,1,10]. Correct. Strict minima.
  Two observations from the hand check:
   (i) the decoded sequence (A,A),(B,B) DOES follow the selection recursion (BR(A,A)=(B,B), BR(B,B)=(A,A));
   (ii) the messages are NOT committed in the sense of B2: with Q=[0,1,10] the minimiser is B for v=A but A for v=B, C.
  So this example shows feedback + switching with ZERO commitment. It supports "feedback can sustain switching";
  it does not illustrate the commitment account, and it shows full commitment is not necessary for the recursion.
  It is a 2-value two-cycle on a 3-value domain, not a three-value case.
C6. sec.6 splitting / damping description — code facts (code audit).

## D. oscillation_explanation_sources.md sec.5 "exact consequences"
All five bullets are direct consequences of the update rules. The most load-bearing one: a finite output
alphabet does not make decoded assignments a closed finite-state system. Keep.

## E. why_two_route_oscillation.md — what must NOT be used
Its own banner retracts 6 points. Going through the body, statements that are retracted, or that the corrected
Rev 3 silently dropped, or that the group's own data contradict:
 E1. Title/abstract "damping always removes it", "literally always" — retracted (banner 3). K_12 counterexample, 2/17 non-monotone scans.
 E2. Lemma 4 "half-weight field" + "coherence" condition (l.68-73) — wrong as written (banner 1). Replaced by B4.
 E3. "the continuous system collapses onto a finite-state system" (l.18), "finite-state collapse" (l.159) — true only under
     full commitment at every t; false in general (D). Seed 0 is a direct counterexample to reading it as general.
 E4. "period is always 2 and never 3, 5, or chaotic" (l.87) — conditional on the recursion; the same census has 16 and 42.
 E5. "MS-SCFG is never aperiodic" (l.159) — window-limited detection; assignments only (banner 6).
 E6. Theorem 7 "iff", the cascade monotonicity claim, and lambda-dagger (l.111-114) — Rev 3 keeps only a one-direction
     "can persist only if" + local attraction; the monotone cascade is contradicted by the non-monotone flip census (l.195).
 E7. Theorem 8.2-8.3 (l.118-119) "frozen pattern => convergence to a local minimum", "longer periods cannot hide" — not in Rev 3. Unproved.
 E8. "a split 2-cycle can never be frustrated", "splitting forces commitment" (l.54) — heuristic, binary-scalar reasoning
     resting on A2 whose bound fails on dense graphs. Hypothesis, not a result.
 E9. Proposition 2 "oscillation, not divergence, is the only possible failure mode" (l.42) — the boundedness of message
     DIFFERENCES is true for any domain (it is C4). "Only failure mode is oscillation" does not follow: bounded != periodic.
 E10. "Poljak-Sura 1983 for arbitrary finite domains" (l.18, l.87) — narrowed in banner 5; to be confirmed by the literature audit.
 E11. "almost always on larger problems" + the geometric-decay argument (l.95) — empirical only (banner 4).
 Keep from this file: the numerical hygiene note (l.197-199): raw messages grow, precision is lost without per-step
 normalisation. It agrees with what was seen on seed 0.

## F. Cross-cutting conventions that every later ticket must state
 F1. Time unit. Engine iteration (Q phase + R phase) vs message hop (paper sec.4, AAAI-2020/AIJ style). "Period 2" means
     different things in the two units.
 F2. Object. Assignment period vs message-difference period vs raw message period (raw messages carry a growing common offset).
 F3. Exact arithmetic vs float64 with sparse normalisation.
 F4. Zero start vs warm start. Every measured number in B is zero-start.
