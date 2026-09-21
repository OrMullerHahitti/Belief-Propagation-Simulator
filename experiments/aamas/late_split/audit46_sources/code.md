# Audit 46, code half: implementation conventions behind the late-split claims

Repository: `/Users/or/Projects/Belief-Propagation-Simulator`, branch `aamas`, HEAD `2fc381b`.
Everything below comes from reading files. Nothing was run (no engine, no script, no test).
Where I reason about what the code would do, I say "from reading the code, not run".
`src/propflow` has no uncommitted changes (`git diff HEAD --stat -- src/propflow` is empty), so the line numbers are the committed code.
Paths are relative to the repository root unless they start with `/`.
No file I read contained instructions addressed to an AI agent.

---

## Q1. Update order inside one iteration

The description is correct. In one call to `BPEngine.step(i)`: every variable computes its Q messages from the R messages sitting in its inbox (those were delivered at the end of the previous step; at iteration 0 they are zeros). Only after all variables have computed does any variable send. Then the variable inboxes are emptied. Then every factor computes R from the Q it just received, and only after all factors have computed does any factor send. Then the factor inboxes are emptied. So a factor sees the fresh Q of the same iteration, and one engine iteration is one Q update followed by one R update. The assignment is read and the cost recorded at the very end of `step`, after the factors have sent, from the fresh R of this iteration. Normalization is not part of `step`; it happens afterwards in `_handle_cycle_events`, so the recorded assignment is always taken before that iteration's normalization. Nothing is sequential: no agent sees another agent's output from the same phase. The only hook that runs inside the compute loop is `post_var_compute` (the damping hook), and it touches only that variable's own outbox.

Two additions to the description. The Q that is sent is what is in the outbox after the damping hook, so before the split it is the damped Q. The belief is the plain sum of the R messages in the inbox; there is no separate unary term at the variable.

Evidence
- `src/propflow/bp/engine_base.py:124-131` all variables compute (+ `post_var_compute`), `:132-133` all variables send, `:134-136` variable inboxes cleared.
- `src/propflow/bp/engine_base.py:139-147` all factors compute, `:148-149` all factors send, `:150-152` factor inboxes cleared.
- `src/propflow/bp/engine_base.py:154` `cost = self.update_global_cost()`, `:155-158` snapshot captured and cost stored.
- `src/propflow/bp/engine_base.py:261-263` cost is computed on `graph.original_factors`; `src/propflow/bp/factor_graph.py:88` reads `var.curr_assignment`; `:55` `original_factors` is a deep copy taken when the graph is built (so it includes the unary factors, and it is not changed by a later split).
- `src/propflow/bp/engine_base.py:80-82` variables and factors are processed in name-sorted order (no numerical effect because of compute-all-then-send-all).
- `src/propflow/bp/engine_base.py:265-276` and `src/propflow/core/components.py:107-124` initial inbox = zero vector from every neighbour factor (only variables get initial messages).
- Late-split runner: `experiments/aamas/late_split/core.py:99-109` `advance` = `engine.step(i)` then `engine._handle_cycle_events(i)`; `core.py:62-68` the snapshot reads `engine.assignments` inside `step`, i.e. before normalization.

---

## Q2. Q computation

Yes, the message to factor f excludes f's own incoming R. It does that by subtraction: it sums all inbox messages once, then subtracts f's message (`total - R_f`). It does not re-sum the others. In exact arithmetic this is the same thing; in floating point it is not bit-identical to summing the others. No normalization of any kind happens inside `compute_Q` (no min or mean subtraction). `compute_Q` has no unary term. Unary costs exist only as separate one-variable factors named `u<idx>`; their R message lands in the variable's inbox like any other R and is summed with the rest. A variable with a single inbox message sends a zero vector.

One detail about the unary factors, from reading the code, not run: `compute_R` for a one-variable factor returns `(c + q) - q`, not `c`. In exact arithmetic that is `c`. In float64 it is `c` rounded to the spacing of `q`. The seed-0 README reports belief magnitudes of about 2.7e13; at that size the spacing of float64 is about 2e-3, while the unary preferences are below 1e-2.

Evidence
- `src/propflow/bp/computators.py:145-146` stack and sum; `:149-152` remove own message; `:112-113` removal is `np.subtract` (`:41-44` dispatch table).
- `src/propflow/bp/computators.py:229-237` single message -> zeros.
- `src/propflow/utils/fg_utils.py:245-305` `build_with_unary_costs`: one `FactorAgent` per variable, name `u{var_name[1:]}` (`:296-303`), table cast to float (`:280`).
- `src/propflow/bp/computators.py:199-214` for a 1-D table `axes_cache` is `()`, so the reduce is over no axes and R = `(table + q) - q`.
- Magnitudes: `experiments/aamas/runs/late_split_domain20_20260921/seed0_diagnosis/README.md:34-36`.

---

## Q3. R computation and the 9c2f4c8 fix

The fix is present. `git merge-base --is-ancestor 9c2f4c8 HEAD` returned exit code 0, the commit is dated 2026-09-09 11:21:38 +0300, and the working-tree `computators.py` equals HEAD. `compute_R` now sorts the incoming Q messages by `factor.connection_number[sender.name]` before it loops, so the loop index is the real table axis. Before the fix the axis was the position in the inbox. The inbox is filled in the order variables send, which is string-sorted by name, so "x13" arrived before "x2" and the factor worked on the transposed table while the cost was still scored on the correct one.

Orientation for a pairwise factor: `edges[f] = [a, b]` gives `connection_number = {a: 0, b: 1}`, and the table is `C[x_a, x_b]`. In `FGBuilder.build_random_graph` `a` is the first variable of the edge tuple. The message to `a` is `R_a(x_a) = min over x_b of [C(x_a, x_b) + Q_b(x_b)]` (add both Q to the table, subtract the recipient's own Q, reduce over every other axis). If `connection_number` is empty the code silently falls back to inbox order.

A regression test with two-digit names exists.

Evidence
- `src/propflow/bp/computators.py:186-195` the sort by `connection_number`; `:199-203` broadcast each Q along its own axis; `:205-207` aggregate; `:210-214` subtract own Q and reduce over the other axes; `:215-221` recipient = the sender of that axis.
- `src/propflow/bp/factor_graph.py:245-255` `connection_number[var.name] = i` from the order of the edge list.
- `src/propflow/utils/fg_utils.py:46-58` factor `f{a}{b}` gets `[a, b]`.
- `git show 9c2f4c8` (adds exactly the lines at `computators.py:186-195`); `tests/test_bp_computators.py:129-156` `test_compute_r_uses_connection_number_not_inbox_order`.

---

## Q4. Belief, assignment, ties, and where the tie-breaking preferences come from

Belief = zero vector plus every R message in the inbox, added one after another in inbox order. Assignment = `np.ndarray.argmin(belief)`, which returns the first (lowest) index on an exact tie. `VariableAgent.curr_assignment` just calls the computator's `get_assignment` on `self.belief`. There is no random tie-breaking and nothing that depends on factor names or clone names.

Tie-breaking preferences in the late-split benchmark: `build_dense` in `domains.py` builds the pairwise graph with `FGBuilder.build_random_graph` (50 variables, density 0.6, factory `random_int` with low=100, high=200), then adds one unary factor per variable with costs drawn from `uniform(0, 1e-2)` per domain value, using `np.random.default_rng(seed)`. `PREF_SCALE = 1e-2`, `NUM_AGENTS = 50`. So: pairwise tables are integers in [100, 200) (upper end excluded), i.i.d. uniform, dtype numpy default int (int64 on this platform, see Q8). Unary preferences are real-valued, i.i.d. uniform on [0, 0.01), one vector per variable, total mass at most 0.5. The recorded global cost includes these unary costs (that is why costs such as 96588.2387 have a fractional part). The domain-10 benchmark `build_random_dense` uses the same recipe.

Evidence
- `src/propflow/bp/computators.py:290-307` belief; `:276-288` `get_assignment`; `:35-39` `np.min -> np.ndarray.argmin`.
- `src/propflow/core/agents.py:136-151` `belief`; `:153-160` `curr_assignment`.
- `experiments/aamas/late_split/domains.py:23-40` `build_dense`.
- `experiments/aaai/code/problems.py:44-45` constants; `:23-29` stated purpose; `:75-78` `_with_tiebreak_prefs`; `:81-100` domain-10 builders.
- `src/propflow/configs/global_config_mapping.py:331-339` `np.random.randint(low, high, size)`; `:356-360` registry.
- `experiments/aamas/runs/late_split_domain20_20260921/random_dense_0/references.json` (domain 20, agents 50, density 0.6).

---

## Q5. Normalization

`step()` never normalizes. Normalization lives in `_handle_cycle_events(i)`: if `normalize_messages` is true and `i % graph_diameter == 0`, it calls `normalize_inbox(var_nodes)`. `run()` calls `step(i)` then `_handle_cycle_events(i)`. A bare `engine.step(i)` loop therefore runs with no normalization at all. The default for `normalize_messages` is True.

The late-split runner sets `normalize_messages=True` and its `advance` function calls `step(i)` and then `_handle_cycle_events(i)` itself, swallowing the convergence stop. So it normalizes every `graph_diameter` iterations, keyed on the global iteration index (the clock is not restarted at the split). The diameter is recomputed after the split. For seed 0, domain 20, the period is 6 before and after the split. It also fires at i = 0.

What normalization does: for each variable, each R message in the inbox gets its own minimum subtracted (one offset per message). It also subtracts the minimum from each message in the variable's stored "last sent Q" list (`_history[-1]`), which is what damping uses as "old". Nothing else is touched.

Does it change an argmin in exact arithmetic? With real-valued tables, no: a constant added to one R shifts the belief by a constant, shifts every Q built from it by a constant, and every R built from those Q by a constant; the shifted "old" Q only shifts the damped Q by a constant. Two caveats in the actual code. (1) Before the split the pairwise tables are int64 and `compute_R` truncates every incoming Q to an integer (Q8). Truncation does not commute with a non-integer shift, and the shift applied to the stored old Q is the minimum of a float vector. So before the split normalization is not exactly neutral (from reading the code, not run). (2) Between two normalizations the undamped post-split messages grow to about 1e13; the seed-0 README reports that normalization changed relative scores by up to 0.024 through floating point, and that it changed no selected value for the two agents it watched.

Evidence
- `src/propflow/bp/engine_base.py:105-163` `step` has no normalization; `:189-194` `run`; `:343-357` `_handle_cycle_events`; `:372-373` the normalize call; `:374-376` convergence check.
- `src/propflow/configs/global_config_mapping.py:89` default True.
- `src/propflow/policies/normalize_cost.py:81-88` the two loops (`last_iteration` then inbox), each `data - data.min()`.
- `experiments/aamas/late_split/core.py:93` flag; `:99-109` `advance`.
- `src/propflow/bp/engines.py:156-160` diameter recomputed after the split.
- `experiments/aamas/runs/late_split_domain20_20260921/random_dense_0/prefix.json` (`normalization_period: 6`); `random_dense_0/best_result.json:13-14` (6 before, 6 after); `manifest.json:103`.
- `experiments/aamas/late_split/README.md:12-13, 21` (keep the original index, native schedule).
- Float effect: `seed0_diagnosis/README.md:33-37`, `seed0_diagnosis/summary.json:35-36`.

---

## Q6. Damping

Only Q messages (variable to factor) are damped in `DampingEngine` and in the late-split engines. R damping exists only in `RDampingEngine` and `QRDampingEngine`, which the late-split study does not use. Formula: `sent = x * old + (1 - x) * computed`, with x the damping factor. "Old" is the previously SENT, already damped message: right after `damp` rewrites the outbox, `append_last_iteration` stores a copy of that outbox, and the next iteration reads it back through `last_iteration` (`_history[-1]`). Messages are matched by recipient name. On the first iteration there is no history, so the first message goes out undamped. One wrinkle: every `graph_diameter` iterations `normalize_inbox` subtracts the minimum from that stored old message. `TD` uses the message from `diameter` iterations ago instead; none of these engines call it.

`ReleasedDampingSplitEngine`: damping factor 0.9 before the split (`Config.damping`). At the start of `step(i)`, if `i >= split_at_iter`, it sets `self.damping_factor = 0.0`. Then the parent `step` applies the split (which sets `_split_applied = True`), and the overridden `post_var_compute` does nothing once `_split_applied` is true: no damping and no history append. So the split iteration itself is already undamped, and damping never comes back. `run.py` raises an error if the run did not end with exactly one split and damping 0.

Evidence
- `src/propflow/policies/damping.py:14-22` `_apply_damping` (`msg.data = x * last_msg.data + (1 - x) * msg.data`, matched by `recipient.name`); `:56-68` `damp`; `:25-53` `TD`.
- `src/propflow/bp/engines.py:290-293` `DampingEngine.post_var_compute` (damp, then append); `src/propflow/core/agents.py:77-82, 97-104` history.
- `experiments/aaai/code/engines.py:95-105` `DampedMidRunSplitEngine` (same hook).
- `experiments/aamas/late_split/core.py:34` damping 0.9; `:71-81` `ReleasedDampingSplitEngine`; `:84-96` `make_engine`.
- `src/propflow/bp/engines.py:108-112, 130` order: split first, then the base `step`.
- `experiments/aamas/late_split/run.py:303-304` the final check.

---

## Q7. Splitting and message hand-over

### (a) How the two copies are built
Tables are `p * C` and `(1.0 - p) * C`. With p = 0.5, `1.0 - 0.5` is exactly 0.5, so the two arrays are elementwise equal. An int64 table times 0.5 becomes float64, and every entry is a multiple of 0.5, so nothing is lost. Names are `<name>'` and `<name>''`. Both copies get a deep copy of the original `connection_number`, and the graph edges are re-created with the original edge data (which holds `dim`). So both copies keep the original axis orientation. Each copy gets its own copy of the table (`FactorAgent.__init__` copies it). `make_engine` passes `split_factor=config.split = 0.5`.

Evidence: `src/propflow/policies/splitting.py:32-33` tables, `:35-36` names, `:38-39` `connection_number`, `:41-43` edges, `:45-50` graph bookkeeping; `src/propflow/core/agents.py:203, 209-226`; `experiments/aamas/late_split/core.py:35, 90`.

### (b) What each variable holds from each copy right after the split
Half of the original factor's last R, from each copy. The engine first copies every variable's inbox, then splits, then clears every mailbox and every history, then rebuilds each variable's inbox: for a message whose sender was split it writes `data * p` from the first copy and `data * (1 - p)` from the second. With p = 0.5 both are `0.5 * R_old`, as float64. "Last R" means exactly what was in the inbox at that moment, so it includes the normalization of the previous iteration if there was one. A message from a factor that was not split is copied unchanged. Missing neighbours would be filled with zeros (does not happen here). The late-split study uses `transfer_mode="transfer"`; the other mode, `"reset"`, sets everything to zero.

Evidence: `src/propflow/bp/engines.py:120-139` order of operations; `:162-166` capture; `:168-173` clear; `:198-222` rebuild (`:206` weights, `:208-214` the clone messages); `:224-238` zero fill; `experiments/aamas/late_split/core.py:92`.

### (c) What each copy holds as incoming Q right after the split
Nothing. The copies are new `FactorAgent` objects with an empty mailbox, and `_clear_agent_state` empties every mailbox anyway. This is not special to the split: in this engine every factor's inbox is emptied at the end of every step, so a factor never keeps a Q from one iteration to the next. The first Q each copy receives is computed in the same `step` call that did the split, from the transferred inbox: `Q(X -> F') = total - 0.5 * R_old(F -> X)`.

Evidence: `src/propflow/core/agents.py:36` new mail handler; `src/propflow/bp/engines.py:168-173, 194`; `src/propflow/bp/engine_base.py:150-152`; `src/propflow/bp/engines.py:108-112`.

### (d) Are the two copies of one factor in identical state at split time?
Yes, when p = 0.5.
- Same table: `splitting.py:32-33` (both are `0.5 * C`).
- Same axis map: `splitting.py:38-39`.
- Same inbox: both empty (`engines.py:168-173`, `agents.py:36`).
- Same stored history: both empty (`agents.py:34`, `engines.py:172-173`).
- Same last R in every neighbour's inbox: both `np.asarray(data, float) * 0.5` (`engines.py:206-214`).

Can anything break the equality later? From reading the code, not run: I found nothing.
- Ordering: factors compute in name order (`F'` before `F''`), but all factors compute before any factor sends (`engine_base.py:139-149`), so order has no effect.
- The two Q messages from a variable to `F'` and `F''` are `total - R'` and `total - R''`, using the same `total` array (`computators.py:146-152`). If `R'` and `R''` are bit-identical, so are the two Q.
- `compute_R` on equal tables and equal Q uses only elementwise add, elementwise subtract and a min reduction (`computators.py:199-214`). Those give the same bits for the same inputs. The Q are ordered by `connection_number`, identical for both copies.
- Damping history: none after the split (`core.py:79-81`), so no different damping history can build up. Even with damping left on, history is matched by recipient name and both copies would start with no history.
- Names enter only as dictionary keys and as the sort key. They never enter a number.
- Tie-breaking: `argmin` depends on the position inside the domain, not on which copy sent what. The min inside `compute_R` returns a value, not an index.
- Normalization subtracts each message's own minimum (`normalize_cost.py:86-88`); equal messages stay equal.
- Message pruning is not used by this engine.

So the induction step of the clone-synchronization lemma ("equal tables and equal inputs give equal outputs") holds in this code, and the base case "all messages zero" can be replaced by "at the split, `R' = R''` for every split factor", which `_transfer_messages` guarantees for p = 0.5. The state of this synchronous engine between two steps is only the R messages in the variable inboxes plus the damping history; both are equal across the two copies at the split. This does NOT hold for p != 0.5: then both the tables and the transferred messages differ.

Independent empirical support already in the repo (I did not rerun it): `experiments/aamas/late_split/FIXED_SEED0_TWO_CYCLE.md:42-45` reports that both clones' Q and R stayed exactly equal at all 1,000 replay updates of the fixed-time seed-0 run (1,502,000 paired checks; 1,502 = 726*2 + 50 factor-variable pairs per update).

### (e) Sum of the two copies' R right after the split
It equals the original R: `0.5*R + 0.5*R = R` (multiplying by 0.5 and adding two equal halves are both exact in float64). So each variable's belief is preserved at the split, up to the order in which the now longer list is summed. It is not doubled.

What this means for the first post-split Q: `Q(X -> F') = sum of all R - 0.5*R_F = [sum over G != F of R_G] + 0.5 * R_F(old)`. The pre-split computed Q to F was `sum over G != F of R_G`. So the first Q into each copy already contains the sibling's share, half of F's own last R, as in any symmetric-split graph. But that `R_F(old)` was computed on the full table C with damped Q. A from-zero split run would have produced the sibling's R from the half table. So the state right after the transfer is a valid, clone-synchronized state of the split system, but it is in general not one the split system reaches from zero, and it is not a fixed point of the split update even if the unsplit run had converged: `min_y [0.5*C(x,y) + Q_Y(y) + 0.5*R_{F->Y}(y)]` is not in general `0.5 * min_y [C(x,y) + Q_Y(y)]` plus a constant. The engine's own docstring says the same ("heuristic injection"). Also, because every factor is split in this study (see g), there are no "unaffected neighbours": every first post-split Q differs from the pre-split one. Two jumps happen in the same iteration: the sent Q stops being `0.9*old + 0.1*computed` and becomes `computed`, and it gains `+0.5*R_F(old)`.

Evidence: `src/propflow/bp/engines.py:206-214`; `src/propflow/bp/computators.py:145-152`; docstrings `engines.py:61-68` and `:180-192`.

### (f) Is any damping history carried over?
No. `_clear_agent_state` clears `_history` on every node that is in the graph after the split, which includes every variable's stored last-sent Q. The copies are new objects with empty history. After the split the damping hook does nothing, so nothing is stored afterwards either.

Evidence: `src/propflow/bp/engines.py:168-173, 194`; `experiments/aamas/late_split/core.py:79-81`; `experiments/aamas/late_split/README.md:21-23`.

### (g) All factors or only some?
All of them, and that includes the 50 unary tie-breaking factors. `make_engine` passes neither `split_targets` nor `split_fraction`, so `_select_factors_for_split` returns every factor in `fg.factors`. The recorded split event for seed 0 has `split_targets: null`, `split_fraction: null`, and its mapping contains `"u1": ["u1'", "u1''"]` through `u50`; there are 776 entries ending in `''` = 726 pairwise + 50 unary. Each unary copy holds half of the unary table and each variable gets `0.5 * R_u` from each copy. In exact arithmetic the two halves add back to the full unary cost, so the belief is unaffected.

Evidence: `experiments/aamas/late_split/core.py:84-96`; `src/propflow/policies/splitting.py:62-72`; `src/propflow/bp/engines.py:123-129`; `experiments/aamas/runs/late_split_domain20_20260921/random_dense_0/best_result.json:15-21, 2927, 3107`.

---

## Q8. Numeric types and rounding

Pairwise tables are integer arrays: `np.random.randint` returns numpy's default int (int64 on this machine; the manifest records macOS arm64, numpy 2.2.6; I did not load `input.npz` to confirm, but `experiments/aamas/splitting_explanation/EXPLANATION_formal.md:372-377` states int64). Nothing converts them to float when the graph is built. Unary tables are float64. Initial messages are float64 zeros.

There is no call to `round`, `floor`, `rint` or `astype(int)` on the late-split path (I grepped `src/propflow/bp`, `core`, `policies`, the late-split `core.py`, `run.py`, `domains.py` and `experiments/aaai/code/engines.py`). But there is one implicit rounding: `compute_R` takes `dtype = cost_table.dtype` and does `q = np.asarray(msg.data, dtype=dtype)`. For an int64 table this truncates every incoming Q toward zero before it is used, and the resulting R is int64. This applies to every pairwise factor during the whole unsplit, damped phase (where Q is fractional because of the 0.9/0.1 mix and the unary preferences). After the split the tables are `0.5 * C` = float64, so Q is no longer truncated. The transferred R is cast to float. Q is always float64 (a stack of int64 and float64 messages becomes float64). Beliefs are float64. The group already knows this: `EXPLANATION_formal.md:372-377` describes it and measured no significant bias on DMS. The late-split notes do not mention it.

The only other rounding on the path is cosmetic: `run.py:181-185` formats costs with `:.4f` to compare against old CSV baselines. The domain-20 runs return before that line because `DMS_trace.npz` exists (`run.py:166-180`).

Evidence: `src/propflow/configs/global_config_mapping.py:331-339`; `src/propflow/core/agents.py:246-256` (table stored as returned); `src/propflow/bp/computators.py:182, 200, 205-207, 214`; `src/propflow/core/components.py:119-124`; `src/propflow/policies/splitting.py:32-33`; `src/propflow/bp/engines.py:210`; `tests/test_fg_builder.py:231` (asserts int32 or int64).

---

## Q9. Provenance of `analysis_oscillation/`

**Self-contained or PropFlow?** `osc_lab.py` is a self-contained re-implementation. It imports only numpy and dataclasses. `runner.py` imports only `osc_lab`. So every number in `results.jsonl` comes from `osc_lab.FastEngine`, never from PropFlow. Its table orientation is fixed by construction: for an edge (i, j) with i < j it stores `C[k][x_i, x_j]`; the arc i->j uses `C[k]`, the arc j->i uses `C[k].T`, and `R[a][x_dst] = min over x_src of (Ct[a][x_src, x_dst] + Q[a][x_src])`. The cost function and the best-response function use the same `C[k][x_i, x_j]`. There is no inbox and no send order, so the kind of bug fixed in 9c2f4c8 cannot occur there.

**Timestamps (local time, +0300).**
- `results.jsonl`: modified 2026-07-12 16:32:17.
- `oscillation_section.tex`: 2026-07-12 16:32:58.
- `why_two_route_oscillation.md`: 2026-07-11 14:47:24.
- `osc_lab.py`, `verify_equiv.py`, `runner.py` (and `analyze.py`): all modified 2026-07-26 11:50:05, created 2026-07-11 13:07:17.
- Fix commit 9c2f4c8: 2026-09-09 11:21:38.
- `__pycache__/osc_lab…pyc`, `runner…pyc`, `verify_equiv…pyc`: created 2026-09-09 12:52-12:53.
So the results and the tex are about two months older than the fix. The three scripts were last modified 14 days AFTER `results.jsonl` was written. The three modules were imported about 90 minutes after the fix commit, but `results.jsonl` was not changed then and no output of that session is in the folder. The folder is untracked (`?? analysis_oscillation/`), so git has no history for it.

**`verify_equiv.py`.** It compares `FastEngine` with PropFlow's `BPEngine`, `SplitEngine`, `DampingSCFGEngine` and `DampingEngine` on four instances of 8 variables (`x0..x7`), domain 3, density 0.5, real-valued costs, unary preferences set to zero, 50 iterations, `normalize_messages=False`, driven by a bare `engine.step(t)` loop. With eight single-digit names, string order equals numeric order equals axis order (edges are built as `[vs[i], vs[j]]` with i < j), so the pre-fix bug could not show on these instances. The script only prints an agreement fraction; it asserts nothing, and no output of it is saved. Its import paths point to `/sessions/exciting-adoring-bohr/mnt/...`, a sandbox that does not exist on this machine, so the July work was done elsewhere against a mounted copy of the repo.

**Conclusion.** The numbers in `oscillation_section.tex` (276-run census, 32 audited runs, damping scans) could not have been produced with the transposed-table behaviour, because they never went through PropFlow's `compute_R`. They come from `osc_lab`, which orients tables consistently. The pre-fix bug touches this folder in only one place: the README's claim that `osc_lab` reproduces PropFlow exactly. That claim was made in July against pre-fix PropFlow, on 8-variable instances where pre-fix PropFlow was correct. It says nothing about graphs with 10 or more variables, and nothing in the folder checks that since the fix.

I cross-checked the tex numbers against `results.jsonl` with grep: the split census has 83 period-1, 191 period-2, one period 16 and one period 42 (= 276); the mechanism units are 34, of which 32 have period 2, and all 32 have `BRxy`, `BRyx`, `pair_local_min` true and `BR_step_acc` 1.0; of 19 damping-scanned instances 17 have period 2 at lambda 0. These match the tex (`oscillation_section.tex:144-149, 225-230, 452-453`).

What I cannot establish from timestamps and code alone:
- That the `osc_lab.py` on disk is byte-for-byte the version that produced `results.jsonl` (it was modified 14 days later; identical modification times on four files look like a bulk reformat or copy, but that is a guess).
- Who produced 76 rows of `results.jsonl`. The file has 490 rows; `runner.py` on disk can only produce the `c1`, `c2`, `c3m`, `c3` units (114 + 34 + 19 + 247 = 414, the number the README gives). The other 68 `c4|` rows (split versus unsplit commitment, which the tex quotes at lines 130-137) and 8 `c5|` rows (K44 bipartite) have no code in the folder.
- That `verify_equiv.py` ever printed full agreement. The README says per-step `engine.normalize_inbox()` was needed on the PropFlow side for undamped runs; the script on disk does not do that.

**Other requested facts about `osc_lab.py` / `runner.py`.**
- Initialization: all R and all Q are zero (`osc_lab.py:135-136`).
- Normalization: every step, both the computed Q and the R get their per-message minimum subtracted (`osc_lab.py:149, 156`). PropFlow's late-split path does this only every 6 iterations and only for R (and the stored old Q).
- Damping: `Q = lam * Q_sent_prev + (1 - lam) * Q_computed` from the second step on (`osc_lab.py:150-153`). Same "old = previously sent" rule as PropFlow.
- Unary preferences: added directly to the belief as `theta` (`osc_lab.py:139-143`); they are not a factor and are never split.
- Tables: integers in [100, 200) cast to float (`osc_lab.py:62-63`), preferences uniform [0, 0.01) (`:66`). No integer truncation. Its own generator and random stream, so these are not the AAAI/PropFlow instances.
- Period detection: `detect_period` returns the smallest p <= pmax such that the last `window` assignments equal the ones p steps earlier, else 0 (`osc_lab.py:182-194`). Census: 700 iterations, window 150, pmax 64 (`runner.py:52-53`). Mechanism: last 100 of 700, window 40, pmax 16 (`runner.py:86-93`). Damping: 2000 iterations, tail 300, window 150, pmax 8 (`runner.py:128-139`).
- "Stable active minimizer / commitment": `saturation()` = fraction of arcs where the minimizing sender value of `Ct + Q` is the same for ALL receiver values (`osc_lab.py:165-169`). Measured at t = 9, t = 49 and averaged over t >= 650 (`runner.py:88-98`).
- "Selection rule": a sample, not every transition. `BR_step_acc` checks `BR(x_t) == x_{t+1}` on every seventh transition of the 100-step tail (about 15 transitions per run), plus the final pair in both directions (`runner.py:99-108`). Only runs with detected period 2 are checked. `sync_best_response` uses the unsplit tables plus `theta` (`osc_lab.py:197-214`).
- Sizes run (`runner.py:36-41, 170-211`): dense = domain 10, density 0.6; sparse = domain 10, density 0.25; binary = domain 2, density 0.25; coloring = 3 colours, density 0.15. Census: dense n in {10, 20, 40} with 12 seeds; the other three n in {10, 20, 40, 80} with 20 seeds; split and unsplit. Mechanism: sparse 30, coloring 30, binary 50, dense 30 variables. Damping: same sizes, 13 lambda values from 0 to 0.9. Nothing with 50 variables and domain 20.
- Late or warm split: none. The split exists only as a constructor argument of `FastEngine` (`osc_lab.py:107-126`), so every split run is split from iteration 0 with zero messages. Every damping scan is on a graph split from iteration 0.

Evidence: `analysis_oscillation/osc_lab.py:11-13, 24-29, 113-127, 145-157`; `analysis_oscillation/runner.py:5-16`; `analysis_oscillation/verify_equiv.py:5-8, 18-34, 37-46, 51-91`; `analysis_oscillation/README.md:27-36`; `stat` and `git show -s --format=%ci 9c2f4c8` output quoted above.

---

## Q10. The constructed three-value teaching example

Both files exist: `experiments/aamas/runs/oscillation_explanation_20260921/verify_example.py` (3,727 bytes) and `verification.json` (1,805 bytes), plus an identical-size `verification.log`, all dated 2026-09-21 15:00.

The script calls the real `MinSumComputator` methods directly: `compute_belief`, `get_assignment`, `compute_Q`, `compute_R`. It builds two `VariableAgent`s X and Y and two `FactorAgent`s ("first_clone", "second_clone") that each hold `costs / 2` for the symmetric 3x3 table `[[4,0,20],[0,6,20],[20,20,20]]`, and sets `connection_number = {"X": 0, "Y": 1}` by hand. It does NOT use `BPEngine.step`, the mail handler, normalization, damping, `_split_factors` or `_transfer_messages`. It starts from a hand-set state: every clone-to-variable R equals `[0, 1, 10]` (not zeros, and not produced by a transfer). Because the table is symmetric, axis orientation cannot matter here.

What it asserts, over four updates, all with exact equality: each variable's belief is `[0, 2, 20]` on even steps and `[2, 0, 20]` on odd steps; both variables pick the same value; every outgoing Q equals that belief divided by 2; at step 2 and again after step 4 every R is back to `[0, 1, 10]`; the original cost sequence is `[4, 6, 4, 6]`. `verification.json` records these four steps, `"native_Q_R_checks": "passed"`, and SHA-256 hashes of `computators.py`, `agents.py`, `components.py`. I hashed the three files in the working tree; all three match, so the saved result was produced against the current code. The JSON itself labels the scope: "constructed exact-arithmetic teaching example; not a seed-0 diagnosis".

Evidence: `verify_example.py:9-10, 15-28, 30-43, 53-71, 72-75, 81-91`; `verification.json:2-3, 95-101`.

---

## Q11. Seed-0 diagnosis folder

I read only `README.md` and `summary.json` in that folder. They do not state the graph size, density, domain size, split iteration, damping values or normalization interval. What they do say:
- It restores "the same best checkpoint" and replays "all 1,000 saved native updates" without changing messages, costs, damping or split settings, and all assignments and costs equal the saved trace (`README.md:3-5`; `summary.json:2` `exact_replay_updates: 1000`).
- It records beliefs and incoming factor messages for x21 and x28 only, plus centered lag-two message differences (`README.md:6-7`).
- Findings: x21 visits values {0, 10, 17}, x28 visits {4, 6, 13}; smallest winning margins 0.267 and 0.490; belief magnitudes up to 2.7e13 and 3.1e13; normalization changed relative scores by up to 0.02417 and changed no assignment; lag-two message residuals in the last 100 updates range from 8.03 to 68.98 (`README.md:17-41`; `summary.json:10-12, 26-27, 35-40`).
- It says only passive replay and analysis were run; the three controls it lists are proposals (`README.md:44-53`).
- It does not list saved files. The folder holds `README.md`, `analyze_incoming.py`, `belief_trace.npz` (502 KB), `incoming_analysis.log`, `replay.log`, `replay.py`, `summary.json`, `visual_explanation/`. There is no input file and no checkpoint in this folder; the README points to the checkpoint of the parent run.

The missing parameters are in the parent run folder `experiments/aamas/runs/late_split_domain20_20260921/`:
- 50 agents, density 0.6, domain size 20 (`random_dense_0/references.json`).
- Best checkpoint at zero-based iteration 332; split before iteration 333; 1,000 post-split updates (iterations 333..1332). The README's "completed update 1333" is index + 1 (`random_dense_0/prefix.json`; `random_dense_0/best_result.json:6-8, 15-16`).
- Damping 0.9 before, 0 after; split 0.5; transfer mode (`manifest.json:2-8`; `best_result.json:17-18`; `run.py:303`).
- Normalization every 6 iterations before and after (`prefix.json`; `best_result.json:13-14`).
- Saved there: `input.npz` (tables and axis order), `best_checkpoint.json.gz`, `fixed_checkpoint.json.gz`, `prefix.npz`, `best_trace.npz`, `fixed_trace.npz`, `DMS_trace.npz`, `DMS_split_0.5_trace.npz`, `references.npz`, and a frozen copy of the source under `source/`.
- I compared that frozen source with the working tree: `computators.py`, `engine_base.py`, `engines.py`, `splitting.py`, `damping.py`, `normalize_cost.py`, `agents.py`, `components.py`, `experiments/aaai/code/engines.py`, `run.py`, `domains.py` are identical. Only `late_split/core.py` differs, in the signature of `Checkpoint.capture` (a new `runtime_graph` option); the engine and the hand-over are unchanged.

---

## Things that surprised me or look inconsistent between notes and code

1. **Integer truncation before the split.** The unsplit damped phase truncates every Q to an integer inside every pairwise factor (`computators.py:182, 200`) because the tables are int64. After the split the half tables are float and the truncation stops. So "DMS, then split" also switches from truncated to untruncated arithmetic at the split. The splitting-explanation notes document this (`EXPLANATION_formal.md:372-377`); the late-split notes never mention it. It also means normalization is not exactly neutral before the split (Q5).

2. **The unary tie-breaking factors are split too**, and a unary factor's message is `(c + q) - q`, not the constant `c`. With beliefs around 1e13 between normalizations, float64 spacing is about 2e-3, the same order as the 1e-2 preferences. The preferences are therefore coarsened exactly when the messages are large. The seed-0 README already reports a 0.024 floating-point effect against smallest margins of 0.27 and 0.49.

3. **Normalizing only every 6 iterations lets undamped post-split messages grow to about 1e13.** From reading the code: with about 60 incoming messages per variable after the split, magnitudes multiply by roughly 60 per iteration between normalizations. Any argument that says "exact arithmetic, so offsets do not matter" holds for the mathematics but not automatically for these float64 runs. `osc_lab` normalizes every step and does not have this issue, so the two code bases are not numerically comparable on this point.

4. **`compute_Q` removes the recipient's message by subtraction** (`total - R_f`). At 1e13 this loses low bits compared with summing the other messages. It does not break clone equality, because both clones use the same `total`.

5. **The clone-synchronization lemma as written does not cover the late split, but its induction does.** `oscillation_section.tex:102-114` states it for zero-initialized messages. The code meets the induction's real requirement (equal tables and equal R for the two copies at the starting time) through the 0.5/0.5 transfer, and only for p = 0.5. The lemma's statement would need "from any state in which the two copies' messages are equal" instead of "with zero-initialized messages". The repo already holds an exact-equality check over 1,000 updates for the fixed-time seed-0 run (`FIXED_SEED0_TWO_CYCLE.md:42-45`). I found no such check for the best-checkpoint run (split at 333), though the same code path applies.

6. **The transfer state is not a continuation of anything.** At the split three things change in one iteration: damping is released, every Q gains `+0.5 * R_F(old)`, and truncation stops. `R_F(old)` came from the full table with damped, truncated Q. The engine docstring says this openly (`engines.py:61-68`). Its sentence about Q to "unaffected neighbours" being unchanged does not apply to this study, because every factor is split.

7. **`analysis_oscillation` provenance gaps.** 76 of the 490 rows in `results.jsonl` (`c4`, `c5`) cannot be produced by the `runner.py` on disk, and the tex quotes numbers from the `c4` rows (split versus unsplit commitment, tex lines 130-137). The scripts were modified 14 days after the results were written. `verify_equiv.py` has no assertion, no saved output, hard-coded sandbox paths, uses 8-variable instances only, and does not do the per-step normalization the README says was required. The folder is untracked in git.

8. **The older study never ran a late or warm split** and never ran 50 variables with domain 20. It normalizes every step, never truncates, adds the unary preference directly to the belief, and checks the selection rule on every seventh transition only. Its census describes split-from-zero runs of `osc_lab`, not PropFlow late-split runs.

9. **The recorded global cost includes the unary preference costs** (at most 0.5 in total), because `original_factors` contains the unary factors. Costs therefore have a fractional part. This matters only when comparing with an integer optimum.

10. **Small counting ambiguity.** `FIXED_SEED0_TWO_CYCLE.md:42-45` says "Q and R messages remained exactly equal ... (1,502,000 paired checks)". 1,502 per update is the number of factor-variable pairs. If Q and R were both compared, one "paired check" must cover both. Worth one sentence of clarification if the number is quoted.

11. **`compute_R` silently falls back to inbox order** when `connection_number` is empty (`computators.py:191-195`). Every graph built through `FactorGraph` or `_split_factors` sets it, so this is not live on the audited path. A hand-built factor without it would bring the old behaviour back without any error.
