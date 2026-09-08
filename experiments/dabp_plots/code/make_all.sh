#!/usr/bin/env bash
# regenerate every figure under experiments/dabp_plots/
# extra arguments go to plot_audit.py (e.g. --results <path to the audit's results dir>)
set -euo pipefail
cd "$(dirname "$0")/../../.."

uv run python experiments/dabp_plots/code/plot_small.py
uv run python experiments/dabp_plots/code/plot_bigger.py
uv run python experiments/dabp_plots/code/plot_audit.py "$@"
