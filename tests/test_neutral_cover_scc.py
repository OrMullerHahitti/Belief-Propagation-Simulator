"""Tests for the SCC-based neutral cover routine and related metrics."""
from __future__ import annotations

import networkx as nx
import numpy as np

from analyzer.reporting import SnapshotAnalyzer, parse_snapshots

RAW_SNAPSHOT = [
    {
        "step": 0,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f1", "values": [0.0, 0.5], "argmin_index": 0},
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "g1", "values": [0.4, 0.2], "argmin_index": 1},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "g1", "values": [0.3, 0.4], "argmin_index": 0},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f1", "values": [0.1, 0.3], "argmin_index": 0},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x1", "values": [0.2, 0.1], "argmin_index": 1},
            {"flow": "factor_to_variable", "sender": "g1", "recipient": "x1", "values": [0.5, 0.5], "argmin_index": 0, "neutral": True},
            {"flow": "factor_to_variable", "sender": "g1", "recipient": "x2", "values": [0.2, 0.4], "argmin_index": 0},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x2", "values": [0.0, 0.4], "argmin_index": 0},
        ],
        "assignments": {"x1": 0, "x2": 1},
        "cost": None,
        "neutral_messages": 1,
        "step_neutral": False,
    }
]


def test_scc_cover_yields_dag_and_bounds() -> None:
    records = parse_snapshots(RAW_SNAPSHOT)
    analyzer = SnapshotAnalyzer(records, domain={"x1": 2, "x2": 2})
    analyzer.register_factor_cost("f1", np.array([[0.0, 1.0], [2.0, 0.5]]))
    analyzer.register_factor_cost("g1", np.array([[0.0, 0.0], [0.0, 0.0]]))

    cover, residual = analyzer.scc_greedy_neutral_cover(0, alpha={})
    assert cover
    assert nx.is_directed_acyclic_graph(residual)

    nilpotent = analyzer.nilpotent_index(0)
    dag_bound = analyzer._dag_bound_cache.get(0)
    if nilpotent is not None and dag_bound is not None:
        assert nilpotent <= dag_bound + 1

    norms = analyzer.block_norms(0)
    assert set(norms.keys()) == {"A", "B", "P"}

    metrics = analyzer.cycle_metrics(0)
    assert metrics["num_cycles"] >= 1

    ratios = analyzer.recommend_split_ratios(0)
    assert all(0.0 <= value <= 1.0 for value in ratios.values())
