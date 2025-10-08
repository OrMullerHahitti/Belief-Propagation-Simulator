"""Tests for reconstructing belief argmin series from snapshots."""
from __future__ import annotations

from analyzer.reporting import SnapshotAnalyzer, parse_snapshots
from analyzer.snapshot_visualizer import SnapshotVisualizer


RAW_SNAPSHOTS = [
    {
        "step": 0,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f1", "values": [0.0, 0.4]},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f1", "values": [0.1, 0.3]},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x1", "values": [0.2, 0.1]},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x2", "values": [0.0, 0.5]},
        ],
        "assignments": {"x1": 1, "x2": 0},
        "cost": None,
        "neutral_messages": 0,
        "step_neutral": False,
    },
    {
        "step": 1,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f1", "values": [0.2, 0.0]},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f1", "values": [0.0, 0.6]},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x1", "values": [0.3, 0.4]},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x2", "values": [0.2, 0.1]},
        ],
        "assignments": {"x1": 0, "x2": 1},
        "cost": None,
        "neutral_messages": 0,
        "step_neutral": False,
    },
]


def test_beliefs_match_visualizer() -> None:
    records = parse_snapshots(RAW_SNAPSHOTS)
    analyzer = SnapshotAnalyzer(records)
    beliefs = analyzer.beliefs_per_variable()

    visualizer = SnapshotVisualizer.from_object(RAW_SNAPSHOTS)
    expected = visualizer.argmin_series()

    assert beliefs == expected
