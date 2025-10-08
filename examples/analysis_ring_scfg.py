"""Demonstration of the reporting analyzer on a small ring graph."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyzer.reporting import AnalysisReport, SnapshotAnalyzer, parse_snapshots

RAW_RING = [
    {
        "step": 0,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f12", "values": [0.0, 0.5]},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f12", "values": [0.4, 0.1]},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f23", "values": [0.2, 0.3]},
            {"flow": "variable_to_factor", "sender": "x3", "recipient": "f23", "values": [0.7, 0.1]},
            {"flow": "variable_to_factor", "sender": "x3", "recipient": "f31", "values": [0.1, 0.4]},
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f31", "values": [0.3, 0.0]},
            {"flow": "factor_to_variable", "sender": "f12", "recipient": "x1", "values": [0.2, 0.1]},
            {"flow": "factor_to_variable", "sender": "f12", "recipient": "x2", "values": [0.3, 0.5]},
            {"flow": "factor_to_variable", "sender": "f23", "recipient": "x2", "values": [0.4, 0.3]},
            {"flow": "factor_to_variable", "sender": "f23", "recipient": "x3", "values": [0.2, 0.2], "neutral": True},
            {"flow": "factor_to_variable", "sender": "f31", "recipient": "x3", "values": [0.3, 0.1]},
            {"flow": "factor_to_variable", "sender": "f31", "recipient": "x1", "values": [0.4, 0.2]},
        ],
        "assignments": {"x1": 0, "x2": 1, "x3": 0},
        "cost": None,
        "neutral_messages": 1,
        "step_neutral": False,
    }
]

COSTS = {
    "f12": np.array([[0.0, 1.0], [2.0, 0.4]]),
    "f23": np.array([[0.0, 0.5], [0.5, 0.0]]),
    "f31": np.array([[0.0, 0.2], [0.8, 0.1]]),
}


def main() -> None:
    records = parse_snapshots(RAW_RING)
    analyzer = SnapshotAnalyzer(records, domain={"x1": 2, "x2": 2, "x3": 2}, max_cycle_len=6)
    for factor, table in COSTS.items():
        analyzer.register_factor_cost(factor, table)

    report = AnalysisReport(analyzer)
    out_dir = Path("results/analysis_ring")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = report.to_json(0)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report.to_csv(out_dir, step_idx=0)
    report.plots(out_dir, step_idx=0, include_graph=True)

    cover, _ = analyzer.scc_greedy_neutral_cover(0, alpha={})
    print("Neutral cover:", cover)
    print("Nilpotent index:", analyzer.nilpotent_index(0))


if __name__ == "__main__":  # pragma: no cover
    main()
