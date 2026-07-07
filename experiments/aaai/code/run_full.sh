#!/usr/bin/env bash
# full AAAI experiment suite: 50 problems x all algorithms per benchmark.
# optimal (branch and bound) is only attempted where it can complete:
# graph_coloring and meeting_scheduling; the domain-10 benchmarks are far
# beyond exact search (10^50 states), so it is skipped there.
set -euo pipefail
cd "$(dirname "$0")/../../.."

# domain-10 benchmarks: no Optimal (exact search infeasible). Attentive (DABP)
# is included so it is tracked everywhere.
NO_OPT="MS DMS DMS_split_0.5 DMS_split_0.4_0.6 DMS_split_at_50 DMS_split_at_100 \
DMS_split_at_300 DMS_split_at_500 DMS_split_at_1000 Attentive \
MS_split_0.5 MS_split_MGM_200 MS_split_MGM_inverted_200 MS_split_opt_200"

# sparse / scale-free: standard split points
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks random_sparse scale_free \
    --algorithms $NO_OPT \
    "$@"

# dense: standard split points plus the opt-in late split@1500
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks random_dense \
    --algorithms $NO_OPT DMS_split_at_1500 \
    "$@"

# true arity-3 benchmark: targeted DMS + split 0.5 only
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks random_ternary \
    --algorithms DMS_split_0.5 \
    "$@"

# coloring / meeting: full set including Optimal (branch and bound completes)
uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks graph_coloring meeting_scheduling \
    --algorithms all \
    --opt-time-limit 300 \
    "$@"

uv run python experiments/aaai/code/analyze_results.py
# one timed instance per benchmark -> data/dabp_timing.csv (DABP/DMS ratio),
# consumed by plot_results.py to stretch the DABP curve onto the time axis
uv run python experiments/aaai/code/time_dabp.py
uv run python experiments/aaai/code/plot_results.py
