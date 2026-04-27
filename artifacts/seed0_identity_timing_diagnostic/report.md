# Seed-0 Identity/Timing Diagnostic Report

This report describes observed behavior in this tested instance. It reports diagnostic signatures and candidate conditions only; it does not claim a theorem.

## Q1. Nested 60/70 identity control
Across 20 nested subset pairs, 11 pairs landed in the same outcome bucket and 9 pairs differed.
Answer: the observed behavior in this tested instance is not a simple monotone percentage effect. The 9 differing nested pairs are a diagnostic signature that subset identity matters strongly.

## Q2. High-switch vs low-switch subsets
Structured subset outcomes: `split_pct60_topswitch_pre40_at_0_transfer` -> `transient_then_oscillation`; `split_pct70_topswitch_pre40_at_0_transfer` -> `converged`; `split_pct60_lowswitch_pre40_at_0_transfer` -> `no_clear_classification`; `split_pct70_lowswitch_pre40_at_0_transfer` -> `no_clear_classification`.

## Q3. Split-all timing at 20 vs 40
Splitting at iteration 20: pre_split_switch_mean5=1756.4, post_split_switch_mean10=3738.2, pre_split_residual2_mean5=15.4, post_split_residual2_mean10=6.55, minimizer_freeze_start=32, minimizer_period2_start=33, assignment_freeze_start=30, assignment_period2_start=31, min_margin_after_split_10=0.0, mean_margin_last20=0.0.
Splitting at iteration 40: pre_split_switch_mean5=2514.6, post_split_switch_mean10=7540.8, pre_split_residual2_mean5=15.2, post_split_residual2_mean10=8.1, minimizer_freeze_start=None, minimizer_period2_start=79, assignment_freeze_start=None, assignment_period2_start=77, min_margin_after_split_10=0.0, mean_margin_last20=1.5.
Answer: active-minimizer switching changes first and most sharply after the split. The iteration-40 run has the larger pre-split and post-split switching signature, then settles into an even/odd pattern rather than one-step minimizer freeze.

## Q4. One-step minimizer switching vs even/odd residual
Historical convergent runs have mean one-step minimizer switching [0.0, 0.0]; period-2 bucket runs have [2900.0, 1058.0]. The corresponding even/odd minimizer-change means are [0.0, 0.0] for convergent runs and [0.0, 0.0] for period-2 bucket runs.
Answer: yes, in the historical comparison this is an observed diagnostic signature. Convergent split runs collapse one-step minimizer switching to zero, while period-2 bucket runs retain persistent one-step switching with zero even/odd minimizer change.

## Q5. Original 60% vs 70% anomaly
In the historical rerun, `split_pct_60_at_0_transfer_instrumented` was classified as `transient_then_oscillation` and `split_pct_70_at_0_transfer_instrumented` was classified as `converged`. The candidate condition in this tested instance is the identity of the selected original factors, not split percentage alone. The nested and structured subset tables provide the diagnostic signature for that interpretation.

## Q6. Original split-at-20 vs split-at-40 timing anomaly
In the historical rerun, `split_all_at_20_transfer_instrumented` was classified as `converged` and `split_all_at_40_transfer_instrumented` was classified as `transient_then_oscillation`. The observed behavior in this tested instance points to the pre-split message and active minimizer state at the split time as the relevant diagnostic signature; see the pre/post switch, residual, and margin summaries in `historical_comparison.csv` and the split-centered plots.

## Artifacts
- `historical_comparison.csv`
- `nested_pair_outcomes.csv`
- `nested_pair_contingency.md`
- `structured_subset_comparison.csv`
- `factor_ranking_pre40.csv`
- `split_time_state_diagnostic.csv`
- `plots/`
