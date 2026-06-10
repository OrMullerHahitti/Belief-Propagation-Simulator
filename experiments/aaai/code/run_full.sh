#!/usr/bin/env bash
# full AAAI experiment suite: 50 problems x all algorithms per benchmark.
# optimal (branch and bound) is only attempted where it can complete:
# graph_coloring and meeting_scheduling; the domain-10 benchmarks are far
# beyond exact search (10^50 states), so it is skipped there.
set -euo pipefail
cd "$(dirname "$0")/../../.."

NO_OPT="DMS DMS_split_0.5 DMS_split_0.4_0.6 DMS_split_at_50 DMS_split_at_100 \
DMS_split_at_300 DMS_split_at_500 DMS_split_at_1000 \
MS_split_0.5 MS_split_MGM_200 MS_split_opt_200"

uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks random_sparse random_dense scale_free \
    --algorithms $NO_OPT \
    "$@"

uv run python experiments/aaai/code/run_experiments.py \
    --benchmarks graph_coloring meeting_scheduling \
    --algorithms all \
    --opt-time-limit 300 \
    "$@"

uv run python experiments/aaai/code/analyze_results.py
uv run python experiments/aaai/code/plot_results.py
