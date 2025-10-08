"""Multi-label Jacobian and neutrality helper tests."""
from __future__ import annotations

import numpy as np

from analyzer.reporting import SnapshotAnalyzer, parse_snapshots

RAW_SNAPSHOT = [
    {
        "step": 0,
        "messages": [
            {"flow": "variable_to_factor", "sender": "x1", "recipient": "f1", "values": [1.5, 0.5, 2.0], "argmin_index": 1},
            {"flow": "variable_to_factor", "sender": "x2", "recipient": "f1", "values": [0.0, 0.8, 0.6], "argmin_index": 0},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x1", "values": [0.2, 0.1, 0.15], "argmin_index": 1},
            {"flow": "factor_to_variable", "sender": "f1", "recipient": "x2", "values": [0.05, 0.3, 0.4], "argmin_index": 0},
        ],
        "assignments": {"x1": 1, "x2": 0},
        "cost": None,
        "neutral_messages": 0,
        "step_neutral": False,
    }
]

COST = np.array(
    [
        [0.0, 1.5, 2.0],
        [0.5, 0.0, 1.0],
        [1.5, 0.2, 0.0],
    ]
)


def test_selector_block_rows_sum_to_zero() -> None:
    records = parse_snapshots(RAW_SNAPSHOT)
    analyzer = SnapshotAnalyzer(records, domain={"x1": 3, "x2": 3})
    analyzer.register_factor_cost("f1", COST)
    matrix = analyzer.jacobian(0)
    # There are 6 Q coords and 6 R coords -> 12x12 matrix
    assert matrix.shape == (12, 12)
    # Row for ΔR_{f1→x1}(label 0)
    row_index = 6  # first R coordinate row
    assert np.isclose(matrix[row_index, :6].sum(), 0.0)
    neutral, winner = analyzer.neutral_step_test(0, "f1", "x1", "x2")
    assert neutral is True and winner == 1
