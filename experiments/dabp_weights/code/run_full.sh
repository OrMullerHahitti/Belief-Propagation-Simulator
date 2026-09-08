#!/usr/bin/env bash
# DABP edge-weight suite: record 50 seeds, analyze, plot.
# pass --engine asym for the 0.95/0.05 split; outputs then land in data_asym/.
# figures for whichever splits have data go to experiments/dabp_plots/small_10agents_50seeds/
set -euo pipefail
cd "$(dirname "$0")/../../.."

data=experiments/dabp_weights/data
if [[ " $* " == *" --engine asym "* ]]; then
    data=experiments/dabp_weights/data_asym
fi

uv run python experiments/dabp_weights/code/run_weights.py "$@"
uv run python experiments/dabp_weights/code/analyze_weights.py --data-dir "$data"
uv run python experiments/dabp_plots/code/plot_small.py
