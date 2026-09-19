# Why damping changes split Min-sum dynamics

This investigation addresses the cause of oscillation and the effect of damping,
not controller training or benchmark cost improvements. It preserves the current
paper, previous evidence, and unrelated work.

## Required evidence

1. Give explicit small original graphs and cost tables, with identical inputs
   for unsplit Min-sum, symmetric splitting without damping, and symmetric
   splitting with old-Q damping. Reconstruct every original-objective cost.
2. Verify the decisive examples through the native AAAI update path and saved
   snapshots. Separate gauge-normalized messages, beliefs, decoded assignments,
   and undamped fixed-point defects. Constant cost alone is insufficient.
3. Derive the feedback equations and certify at least one oscillation and one
   damped fixed point with exact arithmetic. Explain which statements are local,
   conditional, or restricted to a symmetric subspace.
4. Intervene on the same reached message state: change damping while preserving
   messages, and separately remove the sibling return or external amplification.
   Label the latter as modified diagnostic operators, not competing solvers.
5. Include boundary cases: convergence without damping, persistent oscillation
   under insufficient damping, and message convergence with ambiguous decoding.
   Do not assert that damping is universally necessary or sufficient.
6. Produce compact side-by-side figures without titles or subtitles, a technical
   account, and an experiment-section paragraph with the precise supported claim.

## Execution conventions

Use old-Q coefficient lambda in [0,1): Q_new=lambda*Q_old+(1-lambda)*Q_raw.
R updates remain undamped. Start native Q/R at zero, introduce unary evidence
through the native factor phase, and execute the full horizon including native
cycle normalization. The two clones sum to the original factor; score only on
the original objective. Do not use best-so-far cost or early stopping.

Small fixtures may be selected for interpretability. They establish mechanisms
and counterexamples, not population frequency. Record that selection explicitly.
Use exact rational calculations where practical; use floating-point replay only
with explicit tolerances and report near-tie decoding separately.

## Planned causal decomposition

For equal clones, express one full-scale incoming message per original directed
edge. Its outgoing cavity field is twice the external field plus one sibling
return. Compare external/sibling gains (1,0), (2,0), (1,1), and (2,1), retaining
the same cost tables, initialization, and update schedule. Check the (1,0) and
(2,1) trajectories against native unsplit and split runs before interpretation.
Amplitude controls must distinguish the direction of returned evidence from
its mere size. Any mechanism claim is restricted to the tested fixtures.
