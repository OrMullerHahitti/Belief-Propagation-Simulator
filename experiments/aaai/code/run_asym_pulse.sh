#!/usr/bin/env bash
# append two opt-in DMS-SCFG variants to the completed binary benchmarks
# (50 problems each, same seeds and horizon as run_full.sh):
#   DMS_split_0.95   fixed asymmetric split 0.95/0.05 (DABP's ratio, no network)
#   DMS_split_pulse  symmetric split with a temporary 0.95/0.05 pulse over
#                    iterations 64-255, messages kept at both changes
# then rebuild the summary / significance tables.
set -euo pipefail
cd "$(dirname "$0")/../../.."

uv run --no-sync python experiments/aaai/code/run_experiments.py \
    --benchmarks random_sparse scale_free random_dense graph_coloring meeting_scheduling \
    --algorithms DMS_split_0.95 DMS_split_pulse \
    --append \
    "$@"

uv run --no-sync python experiments/aaai/code/analyze_results.py
