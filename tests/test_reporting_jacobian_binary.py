"""Binary-domain Jacobian structure smoke tests."""
from __future__ import annotations

import numpy as np

from analyzer.reporting import SnapshotAnalyzer, parse_snapshots

RAW_SNAPSHOT = [
    {
        "step": 0,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f1", "values": [0.0, 0.5], "argmin_index": 0},
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "g1", "values": [0.4, 0.2], "argmin_index": 1},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f1", "values": [0.3, 0.1], "argmin_index": 1},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x1", "values": [0.1, 0.4]},
            {"flow": "factor_to_variable", "sender": "g1", "recipient": "x1", "values": [0.6, 0.6], "argmin_index": 0, "neutral": True},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x2", "values": [0.2, 0.1]},
        ],
        "assignments": {"x1": 0, "x2": 1},
        "cost": None,
        "neutral_messages": 1,
        "step_neutral": False,
    }
]


def test_variable_rows_collect_incoming_r_messages() -> None:
    records = parse_snapshots(RAW_SNAPSHOT)
    analyzer = SnapshotAnalyzer(records, domain={"x1": 2, "x2": 2})
    analyzer.register_factor_cost("f1", np.array([[0.0, 1.0], [2.0, 0.5]]))
    matrix = analyzer.jacobian(0)
    assert matrix.shape == (6, 6)
    # Row for ΔQ_{x1→f1}
    assert np.isclose(matrix[0, 4], 1.0)
    # Factor rows are restricted to {-1, 0, 1}
    factor_rows = matrix[3:]
    assert set(np.unique(np.round(factor_rows))).issubset({-1.0, 0.0, 1.0})
    neutral, label = analyzer.neutral_step_test(0, "f1", "x2", "x1")
    assert isinstance(neutral, bool)
    assert label in (0, 1, None)
