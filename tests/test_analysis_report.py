"""Tests covering the AnalysisReport helper class."""
from __future__ import annotations

from analyzer.reporting import AnalysisReport, SnapshotAnalyzer, parse_snapshots

RAW = [
    {
        "step": 0,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f1", "values": [0.0, 0.4]},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f1", "values": [0.2, 0.3]},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x1", "values": [0.1, 0.5]},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x2", "values": [0.4, 0.2]},
        ],
        "assignments": {"x1": 0, "x2": 1},
        "cost": None,
        "neutral_messages": 0,
        "step_neutral": False,
    }
]


def test_report_exports(tmp_path) -> None:
    records = parse_snapshots(RAW)
    analyzer = SnapshotAnalyzer(records, domain={"x1": 2, "x2": 2})
    analyzer.register_factor_cost("f1", [[0.0, 1.0], [1.5, 0.2]])

    report = AnalysisReport(analyzer)
    summary = report.to_json(0)
    assert "spectral_radius" in summary

    report.to_csv(tmp_path, step_idx=0)
    assert (tmp_path / "beliefs.csv").exists()
    report.plots(tmp_path, step_idx=0)
    assert (tmp_path / "beliefs.png").exists()
