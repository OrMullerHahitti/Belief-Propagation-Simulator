# Which existing oscillation results can be used for the late-split case

Ticket [#46](https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/issues/46), part of map [#44](https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/issues/44). Written 2026-09-21.

**What this is.** A list of every result the group has been leaning on to explain why late-split Min-sum runs keep switching between assignments. For each result it says what the result needs, whether the late-split dense case meets that, and one decision: **use**, **narrow**, or **exclude**.

**What this is not.** It is not a new theorem and not a new experiment. Nothing was run. No engine, no script, no test.

**The target case.** Synchronous Min-sum on a dense random pairwise graph (50 agents, 20 values, edge probability 0.6). The run is damped with 0.9 on the unsplit graph. Then every factor is split 0.5/0.5 in the middle of the run. The existing messages are handed to the two copies. Damping is switched off. Arithmetic is float64.

**How it was checked.** Three separate readings, then a cross-check.

1. The papers. Read from the primary text, page by page. Two sources could not be opened in full and are marked as such.
2. The code. Read at the cited lines of the committed code on branch `aamas`, HEAD `2fc381b`. Not run.
3. The local proofs. Checked by hand.
4. I then re-checked the claims the decisions rest on: the hand-over code, the Q and R code, the saved JSON of the fixed-time seed-0 study, and the two pages in Yedidia (p. 19) and Ruozzi–Tatikonda (p. 5) that state the equal-copies property.

The three full reports are in [`audit46_sources/`](audit46_sources/). Each claim below points into them.

---

## 1. The short answer

No published theorem covers the target case. Every convergence or optimality guarantee in the ten sources needs at least one thing the target case does not have: a tree, a single constraint, a fixed point, strong damping, or binary values.

What *can* carry an explanation is small, and all of it is conditional.

**Use**

- The Min-sum update and readout equations. They match the code.
- "Splitting keeps the cost function and changes the algorithm" (Yedidia, Ruozzi–Tatikonda, Cohen–Galiki–Zivan).
- The published update rule for a split graph: a variable's message to one copy contains the *other* copy's returning message with weight 1. It needs the two copies to start with equal messages. The hand-over code gives exactly that.
- The table-difference bound in `OSCILLATION_MECHANISM.md` §4. It holds for any start, any graph, any domain size.
- The definition of a "stable active minimizer" (a message that forwards one row of the table).
- The two-assignments theorem (`thm:tworoutes`). Its proof is sound. Its hypothesis is about the decoded assignments only, so it does not care how the run started. The hypothesis must be **measured on each run**. It is not derived.
- The statement "a finite set of visited values does not make the assignments a closed finite-state system".
- The published 3-variable examples where an undamped symmetric split does not converge. They are a precedent, not a theorem.

**Narrow**

- The equal-copies lemma (`lem:clones`). Replace "zero-initialized" with "from any state in which the two copies' messages are equal". True for a 0.5/0.5 split only. Measured on one run.
- The chain `lem:decode` → `lem:sibling` → `cor:rule`. It needs *every* message to forward one row. On the one late-split run where this was measured, 90–94% of messages did, not 100%. Yet the selection rule held on every check. So this chain is not the reason the rule holds there.
- Zivan–Lev–Galiki Lemma 1. It is about assignments only, from a zero start, with no bound on the period. Its printed proof skips steps. Cite it as a prior claim. Do not rest anything on it.
- The computation-tree reading. The papers state it for a constant start only. Use it as a picture, with the handed-over messages at the leaves. Draw no theorem from it.
- The census numbers in `oscillation_section.tex`. They come from a separate re-implementation, split from iteration 0, zero start, at most 10 values. They say nothing about a late split.
- The constructed A/B/C example. It shows that feedback through the other copy can keep a warm start switching. It does not show row-forwarding, and it is not a three-value case.
- The EMA lemma (`lem:ema`). It describes the damped phase before the split only.
- Paper Section 4. Binary values, one split factor, an isolated cycle. Usable as a picture of "locked to one row". Not a theorem for 20 values.
- The 50-seed tail labels. They describe the last 100 assignments. "Fixed" does not mean the messages converged.

**Exclude**

- Every convergence or optimality guarantee in the sources (list in Table A).
- `thm:lambdastar` and `prop:kn` for 20 values. They are binary. Keep `K_n` only as the counterexample to "damping 0.9 always works".
- `cor:bipartite`. Dense random graphs are not bipartite.
- Goles–Olivos, Poljak–Sůra and Ashkenazi-Golan et al. as a justification for period two. Their settings do not contain Min-sum on general tables.
- Eleven statements in the older note `why_two_route_oscillation.md` (Table C).

---

## 2. One result, followed end to end: the equal-copies lemma

This example shows how each decision was reached.

**What the lemma says.** After a symmetric split, the two copies of a factor always send the same message. A variable always sends the same message to both copies. If this is true, one copy can stand for both, and the algebra gets much shorter. Everything else in `oscillation_section.tex` uses it.

**What the written proof needs.** The lemma is stated for "zero-initialized messages" (`oscillation_section.tex:102`). The proof is an induction (`:108-114`). The step from one iteration to the next uses two facts only: the two copies had equal messages last time, and the two copies have equal tables. The zero start is used once, to begin the induction.

**The problem.** A late split does not begin from zero. It begins from whatever the damped run had reached. So the lemma, as written, does not cover it.

**What the code does at the split.** Each variable holds one message `R_old` from each original factor. The engine gives the variable `0.5 · R_old` from the first copy and `0.5 · R_old` from the second. It also empties every mailbox and erases every stored "last sent" message. So at the split the two copies have equal tables, equal messages, and no history. That is exactly what the induction needs to begin.

**Can anything break the equality later?** I went through the code and found nothing.

- A variable's two messages are `total − R'` and `total − R''`. Both use the same `total`. Equal inputs give equal outputs.
- A factor's message uses elementwise addition, elementwise subtraction and a minimum. Same inputs, same bits.
- There is no damping after the split, so no different history can build up.
- Factor names are used as dictionary keys and for sorting. They never enter a number.
- Normalization subtracts each message's own minimum. Equal messages stay equal.

This holds for a 0.5/0.5 split only. With any other ratio the tables differ and so do the handed-over messages.

**What the papers say.** Both Yedidia (p. 19) and Ruozzi–Tatikonda (p. 5) state the same property, and both state it for *any* start in which the copies' messages are equal, not only a zero start. So the narrowed lemma is the published one. Ruozzi–Tatikonda's equation (19) then gives the returning message the weight `c − 1`, which is 1 for two copies.

**What was measured.** On the fixed-time seed-0 run, both copies' Q and R were exactly equal at all 1,000 updates after the split. That is one run. Nobody checked the best-checkpoint run, where the split happens at iteration 333 instead of 1000. The same code runs there.

**Decision: narrow.** Restate the start condition. Say "0.5/0.5 only". Cite the two papers. Say it was measured on one run.

Evidence: `src/propflow/bp/engines.py:162-173, 198-222`; `src/propflow/policies/splitting.py:32-43`; `src/propflow/bp/computators.py:145-152, 199-214`; `experiments/aamas/late_split/core.py:71-81`; `src/propflow/policies/normalize_cost.py:81-88`; `FIXED_SEED0_TWO_CYCLE.md:42-45`; `audit46_sources/code.md` Q7; `audit46_sources/literature.md` §1b, §9a.

---

## 3. The target case, as the code actually runs it

These facts come from reading the code. They matter because several notes assume something slightly different.

| Fact | What the code does | Why it matters here |
|---|---|---|
| Order inside one iteration | All variables compute Q from last iteration's R. All send. All factors compute R from the fresh Q. All send. | One engine iteration is one Q update plus one R update. The paper's Section 4 counts message hops instead. "Period 2" means different things in the two counts. |
| Readout | Assignment and cost are read at the end of the iteration, before any normalization. | — |
| Q | `total − R_f`, by subtraction. No unary term. No normalization. | Same `total` for both copies, so copies stay equal. |
| Unary preferences | Separate one-variable factors `u1..u50`, uniform on [0, 0.01). They are split too. Their message is `(c + q) − q`. | At message size 1e13 the float64 step is about 2e-3, close to the preferences themselves. |
| R | `min` over the other variable of `table + Q`. Table axes follow `connection_number` (fix `9c2f4c8`, present). | The old axis bug is not in these runs. |
| Pairwise tables | Integers in [100, 200), stored as int64. | See next row. |
| Rounding before the split | `compute_R` casts every incoming Q to the table's type. With int64 tables this cuts every Q to an integer. | The damped phase runs on truncated Q. After the split the half tables are floats and the cutting stops. |
| Damping | Only Q is damped. `sent = 0.9 · last_sent + 0.1 · computed`. Off from the split iteration on. | Same convention as Zivan/Cohen. Opposite to Ruozzi–Tatikonda. |
| Normalization | Every 6 iterations (the graph diameter), on the global iteration count. Subtracts each R message's minimum, and the minimum of each stored last-sent Q. | Between normalizations undamped messages grow to about 1e13. |
| The split | Every factor, unary ones included. Tables `0.5 · C`. Each variable gets `0.5 · R_old` from each copy. Mailboxes and damping history are erased. | The two halves add back to `R_old`, so beliefs do not jump. |
| First message after the split | `Q(X→F') = [sum of the other factors' R] + 0.5 · R_F(old)` | Three things change in the same iteration: damping stops, every Q gains `0.5 · R_F(old)`, and integer cutting stops. |
| The state right after the split | Equal copies, but not a state the split graph reaches from zero, and not a fixed point of the split update. | Results that assume a zero start do not apply automatically. |

Evidence: `audit46_sources/code.md` Q1–Q8, with file and line for each row.

---

## 4. Table A — published results

Verification: **V** = read in the primary text. **P** = only part of the text was available. **N** = not read in the primary text.

| # | Result | Where | What it needs | What the conclusion is about | Decision for the late-split case |
|---|---|---|---|---|---|
| A1 | Min-sum message, belief and readout rules | Yedidia §5–6, pp. 10–14, eqs (4)–(8); Weiss–Freeman p. 3, eqs (6)–(7); Cohen et al. pp. 5–6, eqs (1)–(4). **V** | Finite domains | Messages | **Use.** Matches the code. Yedidia's readout rule has no equation number; it is in the §5 step list (p. 10) and the Fig. 7 caption. |
| A2 | Splitting keeps the cost function and changes the algorithm | Yedidia §9, pp. 18–19, Fig. 13; Cohen et al. §5.1, p. 9; Ruozzi–Tatikonda §III. **V** | — | The algorithm | **Use.** |
| A3 | Update rule on a split graph; the returning message has weight `k_a − 1` | Yedidia eqs (13)–(14), p. 19; Ruozzi–Tatikonda eq (19), p. 5, and Algorithm 1, p. 6. **V** | The copies' messages start equal. Both papers say so. | Messages | **Use, with that premise stated.** The hand-over provides it for 0.5/0.5. With two copies the returning weight is 1 and every other weight is 2. |
| A4 | Guarantees reported for splitting: global optimum at a fixed point; schedules that converge | Yedidia pp. 19–20; Ruozzi–Tatikonda Thm V.5, Cor. V.7, Thm VI.2, Algorithms 2–3. **V** | Split weights at each variable sum to at most 1 (Yedidia: `k_a < 1/d`). A fixed point with unique minima. A sequential schedule, or damping `δ = 1/n`. Zero start. | Assignment at a fixed point | **Exclude.** A two-way split has weight 2 per factor. |
| A5 | Local optimality at a fixed point with unique minima | Ruozzi–Tatikonda Thm V.2, Cor. V.3. **V** | A fixed point of the messages | Assignment at a fixed point | **Exclude.** A switching run has no fixed point. For runs with a fixed assignment tail, nobody has checked that the *messages* stopped. |
| A6 | Unwrapped (computation) tree | Weiss–Freeman §II-A, pp. 4–5, Fig. 3; §II-B, p. 6. **V** | Stated for a constant start | Messages after `t` iterations | **Narrow.** Use as a picture only: the tree of the *split* graph, with the handed-over messages at the leaves. That extension is standard but is not printed in the paper. |
| A7 | "Claim 1": optimal against changes on trees and single loops | Weiss–Freeman p. 4, proof pp. 6–7. **V** | A fixed point, unique maximizers | Assignment at a fixed point | **Exclude.** |
| A8 | Why tree conclusions do not transfer directly | Weiss–Freeman §IV, p. 9. **V** | — | — | **Use**, as a caution. |
| A9 | Backtrack Cost Tree, Definition 1 | Zivan–Lev–Galiki p. 7337, end of "Preliminaries". **V** | Zero start. No ties (footnote 5). | — | **Narrow.** Same as A6. The paper has no section called "Backtrack Cost Tree". |
| A10 | Lemma 1: assignments become periodic | Zivan–Lev–Galiki pp. 7337–7338. **V** | Zero start. No ties. | **Assignments only.** No bound on the period or on when it starts. | **Narrow.** Cite as a prior claim. It fits period 2 and period 42 equally, so it explains neither. The printed proof does not show why a finite return sequence must repeat, does not treat equal average costs, asserts "never chosen again", and never shows that the messages repeat. |
| A11 | Lemma 2, Corollary 1 | Zivan–Lev–Galiki p. 7338. **V** | Rest on Lemma 1 | — | **Exclude.** Not needed, and they inherit A10's gaps. |
| A12 | Theorem 1, Corollary 2, Proposition 1 | Zivan–Lev–Galiki pp. 7338–7339. **V** | Damping with `1 − λ < 1/(2d)`, `d` the largest degree. Proposition 1 also needs the original graph to be a tree. Corollary 2 also needs a consistent assignment tree. | End state of a damped run | **Exclude.** Damping is off after the split. My own substitution, not in the paper: with about 29 neighbours per agent the condition needs `λ` above about 0.983, so even the damped phase at 0.9 is outside it. |
| A13 | A 3-variable chain, both constraints split 0.5/0.5, two values: undamped Max-sum does not converge | Zivan–Lev–Galiki Example 1, Fig. 4, p. 7339; Cohen et al. §5.3, p. 11, Fig. 5. **V** | Zero start. Empirical. | Assignments | **Use**, as a published precedent only. No proof, no statement about the period. `oscillation_explanation_sources.md` does not cite either passage. |
| A14 | Damping formula `m(k) = λ·m(k−1) + (1−λ)·new`; only variables damp | Cohen et al. eq (5), p. 7; Zivan–Lev–Galiki p. 7335. **V** | — | — | **Use** for the phase before the split. Matches `src/propflow/policies/damping.py:14-22`. Warning: in Ruozzi–Tatikonda `δ` weights the *new* message and the damped message is factor-to-variable, so `δ = 1 − λ`. |
| A15 | Damped Max-sum converges on trees; any constant split of a single constraint converges after one iteration | Cohen et al. Lemma 2, Prop. 2 (p. 8); Lemmas 3–5, Prop. 3 (pp. 9–10). **V** | A tree, or a single constraint. Zero start. A unique smallest table entry. | Assignments | **Exclude.** The paper itself says (p. 11) that the single-constraint result does not extend to 3 variables and 2 constraints. |
| A16 | "No theoretical guarantee identifies when damped BP converges" | Cohen et al. p. 7. **V** | — | — | **Use**, as a statement of what is not known. |
| A17 | Symmetric threshold networks end in period 1 or 2 | Goles–Olivos 1980. **N** (abstract and a secondary source only) | Binary states. A threshold rule on the neighbours' current states. | Assignments | **Exclude**, and unverified. |
| A18 | Symmetric opinion dynamics end in period 1 or 2 | Poljak–Sůra 1983, pp. 119–120. **V** for the theorem and the whole proof | The pairwise term is a weight applied only when the two opinions are **equal**. Ties go to the highest-numbered opinion. The assignment is the whole state. | Assignments | **Exclude** as a justification. General cost tables are not allowed there, and in Min-sum the next assignment is not a function of the current one. It can be cited as related work; `oscillation_section.tex:284-296` describes it accurately. |
| A19 | Simultaneous best response in random potential games ends in a cycle of length 1 or 2 | Ashkenazi-Golan et al. 2025, Lemma 3.2, Thm 3.1, p. 5. **V** | **Two players.** A generic random table over joint profiles. | Assignments | **Exclude** as a justification. For three or more players the paper has simulations only. |
| A20 | Min-sum on a single cycle: converge or oscillate periodically | Forney et al. 2001. **P** (introduction only) | A single cycle | — | **Exclude.** Many cycles here. The propositions were not read. |
| A21 | Min-Sum Splitting for consensus | Rebeschini–Tatikonda 2017, Thm 4, p. 11. **V** (pp. 1–12) | Real variables, quadratic costs, linear updates | Real-valued estimates | **Exclude.** No discrete choice is involved. |

**An open check, not a decision.** Poljak–Turzik 1986 (*Discrete Applied Mathematics* 13, pp. 27–32) could not be opened. Its abstract says that `y(t+1) = f(A·y(t))` with `A` symmetric and `f` a subgradient of a convex function has period 1 or 2 only. With one-hot coding, the recursion in `thm:tworoutes` looks like an instance. If it is, the sentence in `oscillation_section.tex` that the theorem covers what older results do not must be checked against this paper before submission.

Evidence: `audit46_sources/literature.md`, one section per source, with page numbers.

---

## 5. Table B — local results

| # | Result | Where | What it needs | Checked how | Decision for the late-split case |
|---|---|---|---|---|---|
| B1 | Update equations | `OSCILLATION_MECHANISM.md` §1–2 | — | Against the code | **Use.** Add three facts from §3 above: no unary term at the variable, Q by subtraction, readout before normalization. |
| B2 | The update is piecewise affine and need not contract | `OSCILLATION_MECHANISM.md` §3 | — | By hand | **Use.** |
| B3 | Table-difference bound: `min_w[C(v,w) − C(a,w)] ≤ R(v) − R(a) ≤ max_w[C(v,w) − C(a,w)]` | `OSCILLATION_MECHANISM.md` §4 | Exact arithmetic. Any start, any graph, any domain. For a split copy use `C/2`. | By hand | **Use.** Nobody has yet evaluated it on the 20-value tables, so it is unknown whether it rules out any value there. |
| B4 | The constructed A/B/C example | `OSCILLATION_MECHANISM.md` §5; `experiments/aamas/runs/oscillation_explanation_20260921/verify_example.py` | Two variables, three values, one constraint in two copies, a hand-set warm start `R = [0, 1, 10]` | By hand. Also checked earlier against the native `compute_Q` / `compute_R`; the saved source hashes match today's code. | **Narrow.** The assignments follow the selection rule. But the messages do **not** forward one row: with `Q = [0, 1, 10]` the minimizer is B for entry A, and A for entries B and C. So it shows feedback, not row-forwarding. It also shows the selection rule can hold with no row-forwarding at all. It switches between two values on a three-value domain. It is not a three-value case. |
| B5 | "Stable active minimizer": the same minimizer for every entry, so the message is one table row plus a constant | `oscillation_section.tex:88-100` | — | By hand | **Use** as a definition. |
| B6 | `lem:clones`, equal copies | `oscillation_section.tex:102-114` | As written: zero start. As proved: equal copies at the start. | By hand, against the code, and against A3 | **Narrow.** See §2. |
| B7 | `lem:decode`, `lem:sibling`, `cor:rule`: the selection rule follows from row-forwarding | `oscillation_section.tex:157, 177, 205` | **Every** message into and out of every variable forwards one row, at every iteration from some point on. Unique minima. Equal copies. | Read carefully; no gap found. A line-by-line check belongs to #48. | **Narrow.** Sound as conditional statements. On the fixed-time seed-0 run the premise is false (94.28% and 90.63% of 1,452 messages) while the rule itself held on 5,000 of 5,000 checks. Treat the rule as something measured on a run, not as a consequence of this chain. |
| B8 | `thm:tworoutes`: if the assignments follow the selection rule with unique minima, the run ends in period 1 or 2 | `oscillation_section.tex:251` | Only the hypothesis on the assignments. Shared pairwise tables plus unary terms. No condition on the start, the graph, the domain, splitting or damping. | By hand. Sound. | **Use**, with the hypothesis measured on each run. Read backwards it is also a check: a run whose assignments repeat with period 42 must break the rule, or have a tie, at least once in every period. Caution: "unique minima" relies on preferences below 0.01, while float64 steps reach 2e-3 at message size 1e13. |
| B9 | `cor:localmin`; `cor:bipartite` | `oscillation_section.tex:275, 325` | B8's hypothesis; a bipartite graph | By hand | `cor:localmin`: **use** with B8's hypothesis. `cor:bipartite`: **exclude**. |
| B10 | `lem:ema`: damping shrinks an alternating pair by `(1−λ)/(1+λ)` | `oscillation_section.tex:393` | The damping formula of A14 | By hand | **Narrow.** It speaks about the damped phase before the split. After the split `λ = 0`. |
| B11 | `thm:lambdastar`, `prop:kn` | `oscillation_section.tex:408, 468` | Binary values. `prop:kn`: `K_n`, zero start, exact symmetry. | By hand | **Exclude** for 20 values. Keep `K_n` as the counterexample to "0.9 always works". |
| B12 | The census numbers (276 runs, 32 audited, damping scans, row-forwarding shares) | `oscillation_section.tex:130-149, 225-233, 452-466`; `analysis_oscillation/results.jsonl` | — | Against the scripts and file dates | **Narrow.** Details in §7. They describe split-from-zero runs of a separate re-implementation. |
| B13 | Paper Section 4 (`thm:stability` … `prop:asym`) | `publish/sec4_effect_of_splitting.tex:73-336` | Binary values. One split factor. An isolated 4-edge cycle. The dynamic version adds an outside term on one variable only, bounded by `B < min(2|d|, −(τ_U + τ_L))`. | By hand | **Narrow** to a picture of "locked to one row" for two values. **Exclude** as a theorem for the target. `oscillation_section.tex:128-130` already says the bound does not hold everywhere on dense graphs. |
| B14 | Five "exact consequences" | `oscillation_explanation_sources.md` §5 | — | By hand | **Use.** The one that matters most: a finite set of visited values does not make the assignments a closed finite-state system. |

Evidence: `audit46_sources/local_derivations.md` sections A–D; `audit46_sources/code.md` Q9–Q10.

---

## 6. Table C — statements that must not be used

All of these are in `analysis_oscillation/why_two_route_oscillation.md`. The file's own banner withdraws six points. Going through the body gives eleven statements. Each one is either withdrawn, dropped without comment in the corrected `oscillation_section.tex`, or contradicted by the group's own data.

| # | Statement | Line | Why not |
|---|---|---|---|
| E1 | Damping "always" removes the oscillation | title, abstract | Withdrawn (banner 3). `K_12` counterexample; 2 of 17 scans are not monotone. |
| E2 | Lemma 4, the "half-weight field" and the "coherence" condition | 68–73 | Wrong as written (banner 1). Replaced by `lem:sibling`. |
| E3 | "The continuous system collapses onto a finite-state system" | 18, 159 | True only when every message forwards one row at every iteration. Seed 0 is a direct counterexample to the general reading. |
| E4 | "The period is always 2 and never 3, 5, or chaotic" | 87 | Conditional on the selection rule. The same census contains periods 16 and 42. |
| E5 | "MS-SCFG is never aperiodic" | 159 | Detection was limited to a window and to assignments (banner 6). |
| E6 | Theorem 7 as "if and only if"; the monotone cascade; `λ†` | 111–114 | The corrected text keeps one direction only. The flip census (l. 195) is not monotone. |
| E7 | Theorem 8.2–8.3: a frozen pattern converges to a local minimum; longer periods cannot hide | 118–119 | Not in the corrected text. Unproved. |
| E8 | "A split 2-cycle can never be frustrated"; "splitting forces commitment" | 54 | A binary argument resting on the Section 4 bound, which fails on dense graphs. A guess, not a result. |
| E9 | Proposition 2: "oscillation is the only possible failure mode" | 42 | Bounded message differences are true (that is B3). "Bounded" does not imply "periodic". |
| E10 | Poljak–Sůra covers "arbitrary finite domains" | 18, 87 | It covers any number of opinions, but only equal-opinion weights (A18). |
| E11 | "Almost always on larger problems", with the geometric-decay argument | 95 | Empirical only (banner 4). |

**One thing to keep** from that file: the note at lines 197–199 that raw messages grow and lose precision unless they are normalized every step. It agrees with what was seen on seed 0.

---

## 7. What has been measured, and on which runs

Several numbers are easy to quote for the wrong run. This table says what each study covers.

| Quantity | `analysis_oscillation` census | Fixed-time seed 0 | Best-checkpoint seed 0 | 50-seed study |
|---|---|---|---|---|
| Start of the split graph | Zero messages, split from iteration 0 | Late split at update 1000 | Late split at iteration 333 | Both late-split variants |
| Code | `osc_lab.py`, a separate re-implementation | PropFlow | PropFlow | PropFlow |
| Size | At most 10 values; dense graphs up to 40 agents | 50 agents, 20 values | 50 agents, 20 values | 50 agents, 20 values |
| Assignment period | Yes. Last 150 of 700 iterations, periods up to 64. | Period 2 from update 1013 to 2000 | Exact 42-update cycle in the run normalized every update. The native run follows it but never repeats exactly. | Labels from the last 100 assignments only: fixed-time 32 period-two, 17 fixed, 1 other; best-checkpoint 31 fixed, 18 period-two, 1 other |
| Two copies equal | Built in | Exact at all 1,000 updates | Not checked | Not checked |
| Share of messages forwarding one row | Yes, at three times per run | 94.28% and 90.63% of 1,452 | Not measured | No |
| Selection rule | Every 7th transition, period-2 runs only | 5,000 of 5,000 | Not measured. By B8 it must fail at least once per period. | No |
| At the winning entry, every factor minimizes at the neighbour's preceding value | No | 145,200 of 145,200 | Not measured | No |
| Do the messages repeat? | Not checked | An exact-arithmetic two-cycle of relative messages exists and selects the same two assignments. The native messages are not exactly period two (largest lag-two difference 0.00818). | Not checked for the messages. In the normalized run the four switching agents' scores repeat within 1.6e-6 over 23 cycles. | No |

**About the census.** `osc_lab.py` imports only numpy, and `runner.py` imports only `osc_lab.py`. So the old table-axis bug in PropFlow cannot have touched these numbers. Three gaps remain.

- 76 of the 490 rows in `results.jsonl` (68 `c4`, 8 `c5`) cannot be produced by the `runner.py` on disk. The tex quotes `c4` numbers at lines 130–137.
- The scripts were last modified 14 days after `results.jsonl` was written. The folder is not tracked in git.
- `verify_equiv.py` compares `osc_lab` with PropFlow on 8-variable instances only. It asserts nothing, saves nothing, and points to a path that does not exist on this machine.

`osc_lab` also differs from the late-split runs in ways that matter for numbers: it normalizes every step, never cuts Q to integers, and adds the unary preference straight to the belief.

**About the fixed-time seed-0 study.** I read its three JSON result files and the numbers match the note. The note says itself that it does not prove the run repeats forever under float64. One wording point: "1,502,000 paired checks" is 1,502 factor–variable pairs times 1,000 updates, so each check must cover both Q and R.

Evidence: `audit46_sources/code.md` Q9, Q11; `FIXED_SEED0_TWO_CYCLE.md`; `experiments/aamas/runs/late_split_domain20_20260921/fixed_seed0_cycle/step2_factor_observation.json`, `step3_closure.json`, `certificate_verification.json`; `experiments/aamas/runs/late_split_domain20_50seeds_20260921/RESULTS.md`; `experiments/aamas/runs/late_split_domain20_20260921/seed0_center_each_update/`.

---

## 8. What the audit leaves for the later tickets

These are questions the audit exposed. None of them is a result.

**For #45 (what exactly is being claimed).** Four choices must be written down before any claim is made.

1. The time unit: engine iterations or message hops.
2. The object: values visited by one agent, the joint assignment, or the messages up to an added constant.
3. A finite observed tail, or a statement about all later iterations.
4. Exact arithmetic, or float64 normalized every 6 iterations.

Every published periodicity result in Table A is about assignments. None is about messages.

**For #47 (the account).** The account cannot rest on "every message forwards one row". That was measured false on the one late-split run where it was measured.

**For #48 (checking the selection-rule chain).** The fixed-time study measured a weaker premise, and measured it true on every check: *at the entry that wins*, every factor minimizes at the neighbour's preceding selected value. A candidate argument from that premise to the selection rule, **not audited by anyone and not a result of this ticket**:

- For any entry `v`, each copy's message is at most `½·C(v, x̂_j) + Q_j(x̂_j)`, because a minimum is at most any one term.
- Add both copies, all neighbours and the unary halves: `belief(v) ≤ φ(v) + Σ_j C(v, x̂_j) + K`, where `K` does not depend on `v`.
- The premise makes this an equality at the winner `w`.
- So `φ(w) + Σ_j C(w, x̂_j) ≤ φ(v) + Σ_j C(v, x̂_j)` for every `v`. That is the selection rule.

If #48 confirms it, the rule rests on a premise that holds on the measured run, and B7's stronger premise is not needed.

**For #50 (what to measure next, if approved).**

- Equal copies on the best-checkpoint run.
- Where, inside the 42-update cycle, the selection rule or the winning-entry premise fails.
- The split iteration changes three things at once (§3). Any comparison "before versus after the split" mixes them.
- The damped phase runs on Q cut to integers. An earlier measurement on DMS found no significant bias (`experiments/aamas/splitting_explanation/EXPLANATION_formal.md:370-378`), but the late-split notes never mention it.

**Housekeeping, outside the map.**

- `oscillation_explanation_sources.md` has two imprecise pointers (Yedidia's readout rule; the "Backtrack Cost Tree" section name), two missing premises (equal start for eq (13); constant start for the unwrapped tree), and three passages it should cite (A4's weight range, A13 twice). I did not edit that note.
- The Poljak–Turzik check in §4.
- The three provenance gaps in `analysis_oscillation/` in §7.

---

## 9. Source notes

- [`audit46_sources/literature.md`](audit46_sources/literature.md) — the ten sources, page by page, with a line-by-line check of `oscillation_explanation_sources.md` and of the literature paragraph in `oscillation_section.tex`. Lists what could not be opened.
- [`audit46_sources/code.md`](audit46_sources/code.md) — eleven questions about the implementation, each answered with file and line. Nothing was run.
- [`audit46_sources/local_derivations.md`](audit46_sources/local_derivations.md) — the hand check of the local proofs and of the older note.

Starting points named by the ticket: [`oscillation_explanation_sources.md`](oscillation_explanation_sources.md), `analysis_oscillation/oscillation_section.tex`, `analysis_oscillation/why_two_route_oscillation.md`.
