#!/usr/bin/env bash
# DABP-SymSplit edge-weight suite: record 50 seeds, analyze, plot.
set -euo pipefail
cd "$(dirname "$0")/../../.."

uv run python experiments/dabp_weights/code/run_weights.py "$@"
uv run python experiments/dabp_weights/code/analyze_weights.py
uv run python experiments/dabp_weights/code/plot_weights.py
