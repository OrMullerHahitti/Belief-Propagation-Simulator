"""Ring-style example expectations for the reporting analyzer."""
from __future__ import annotations

import numpy as np

from analyzer.reporting import SnapshotAnalyzer, parse_snapshots

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


def test_ring_nilpotent_bound_and_cover() -> None:
    records = parse_snapshots(RAW_RING)
    analyzer = SnapshotAnalyzer(records, domain={"x1": 2, "x2": 2, "x3": 2}, max_cycle_len=6)
    for factor, table in COSTS.items():
        analyzer.register_factor_cost(factor, table)

    cover, residual = analyzer.scc_greedy_neutral_cover(0, alpha={})
    assert cover

    nilpotent = analyzer.nilpotent_index(0)
    dag_bound = analyzer._dag_bound_cache.get(0)
    if nilpotent is not None and dag_bound is not None:
        assert nilpotent <= dag_bound + 1

    ratios = analyzer.recommend_split_ratios(0)
    assert all(0.0 <= value <= 1.0 for value in ratios.values())
