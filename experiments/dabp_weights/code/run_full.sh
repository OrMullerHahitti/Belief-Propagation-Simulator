#!/usr/bin/env bash
# DABP edge-weight suite: record 50 seeds, analyze, plot.
# pass --engine asym for the 0.95/0.05 split; outputs then land in data_asym/ + plots_asym/.
set -euo pipefail
cd "$(dirname "$0")/../../.."

data=experiments/dabp_weights/data
plots=experiments/dabp_weights/plots
if [[ " $* " == *" --engine asym "* ]]; then
    data=experiments/dabp_weights/data_asym
    plots=experiments/dabp_weights/plots_asym
fi

uv run python experiments/dabp_weights/code/run_weights.py "$@"
uv run python experiments/dabp_weights/code/analyze_weights.py --data-dir "$data"
uv run python experiments/dabp_weights/code/plot_weights.py --data-dir "$data" --plots-dir "$plots"
