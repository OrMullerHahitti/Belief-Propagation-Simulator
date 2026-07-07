#!/usr/bin/env bash
# Full ternary (arity-3) AAAI suite: 50 problems x the full algorithm family
# MINUS DABP (its integration is unary/binary only), per benchmark.
#
# Runs the THREE FAITHFUL ternary analogs only -- the unambiguous arity-3
# versions of the binary benchmarks:
#   random_sparse_ternary       (random arity-3 factors at the sparse degree)
#   random_dense_ternary        (same at the dense degree; heaviest benchmark)
#   meeting_scheduling_ternary  (each agent in three meetings -> ternary constraint)
# The two CONSTRUCTED families (graph_coloring_ternary, scale_free_ternary) have
# no canonical ternary form and are intentionally excluded here; they remain
# available by name or via `--benchmarks all_ternary` if wanted later.
#
# Writes to experiments/aaai/ternary_data and experiments/aaai/ternary_plots so
# the binary suite under data/ and plots/ is untouched. The goal is to see
# whether the binary phenomena (damping, the 0.5 split, mid-run split timing,
# the MGM/optimal split merges) reappear when every factor is genuinely ternary
# -- NOT to compare ternary against binary.
#
# Optimal (exact branch and bound) is only attempted on meeting (20 variables).
# The domain-10 random ternary benchmarks are far beyond exact search (10^50
# states), so Optimal is omitted there.
set -euo pipefail
cd "$(dirname "$0")/../../.."

TERNARY_DATA="experiments/aaai/ternary_data"
TERNARY_PLOTS="experiments/aaai/ternary_plots"

# domain-10 ternary benchmarks: no Optimal (exact search infeasible), no DABP.
NO_OPT="MS DMS DMS_split_0.5 DMS_split_0.4_0.6 DMS_split_at_50 DMS_split_at_100 \
DMS_split_at_300 DMS_split_at_500 DMS_split_at_1000 \
MS_split_0.5 MS_split_MGM_200 MS_split_MGM_inverted_200 MS_split_opt_200"

# sparse ternary: standard split points
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks random_sparse_ternary \
    --algorithms $NO_OPT \
    --out-dir "$TERNARY_DATA" \
    "$@"

# dense ternary: standard split points plus the opt-in late split@1500 (mirrors
# the binary dense run); this is the heaviest benchmark (~490 arity-3 factors)
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks random_dense_ternary \
    --algorithms $NO_OPT DMS_split_at_1500 \
    --out-dir "$TERNARY_DATA" \
    "$@"

# meeting ternary: full set including Optimal (branch and bound may complete on
# 20 variables, time-limited); still no DABP (resolved by --algorithms all for
# the ternary suite)
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks meeting_scheduling_ternary \
    --algorithms all \
    --opt-time-limit 300 \
    --out-dir "$TERNARY_DATA" \
    "$@"

uv run python experiments/aaai/code/analyze_results.py --data-dir "$TERNARY_DATA"
# no time_dabp.py: DABP is excluded from the ternary suite, so there is no curve
# to stretch onto the wall-clock axis.
uv run python experiments/aaai/code/plot_results.py \
    --data-dir "$TERNARY_DATA" --plots-dir "$TERNARY_PLOTS"
