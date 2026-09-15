"""Tests for verdict, gauge, cavity, and component interpretation."""

import networkx as nx
import numpy as np
import pytest

from propflow.snapshots import EngineSnapshot, VariableDynamicsAnalyzer
from propflow.snapshots.variable_dynamics import (
    component_profile,
    cost_table_profile,
    observed_period,
)


def snapshots():
    result = []
    for t in range(8):
        messages = {
            ("f'", "x"): np.array([0.0, 3.0 if t % 2 else -3.0]) + 100 * t,
            ("f''", "x"): np.array([0.0, 3.0 if t % 2 else -3.0]) - 20 * t,
            ("g", "x"): np.array([0.0, 2.0]),
        }
        b = sum(messages.values())
        result.append(
            EngineSnapshot(
                step=t + 1,
                lambda_=0.0,
                dom={"x": ["0", "1"]},
                N_var={"x": ["f'", "f''", "g"]},
                N_fac={},
                Q={},
                R=messages,
                beliefs={"x": b},
                assignments={"x": int(b.argmin())},
            )
        )
    return result


def test_verdict_period_and_gauge_independence():
    a = VariableDynamicsAnalyzer(snapshots())
    row = a.verdicts(window=8)[0]
    assert row["unsettled"] and row["tail_switches"] == 7
    assert row["observed_period"] == 2
    assert row["belief_span"] == 12
    assert row["last_switch_step"] == 8
    assert observed_period(np.array([0, 1, 0, 1])) is None


def test_stable_verdict_does_not_mean_stable_beliefs():
    records = snapshots()
    for s in records:
        s.beliefs["x"][1] += 20
        s.assignments["x"] = 0
    row = VariableDynamicsAnalyzer(records).verdicts(window=8)[0]
    assert not row["unsettled"] and not row["belief_fixed"]


def test_clone_cavity_retains_sibling_and_external_removes_both():
    result = VariableDynamicsAnalyzer(snapshots()).cavities("x", ["f'", "f''"])
    assert result["external_cavity"][0].tolist() == [0, 2]
    assert result["clone_cavities"][0, 0].tolist() == [0, -1]
    np.testing.assert_allclose(
        result["incoming"].sum(axis=1) + result["external_cavity"], result["belief"]
    )


def test_reject_missing_messages_or_gapped_tail():
    records = snapshots()
    records[0].R.pop(("g", "x"))
    with pytest.raises(KeyError):
        VariableDynamicsAnalyzer(records).cavities("x", ["f'"])
    records[3].step = 50
    with pytest.raises(ValueError, match="consecutive"):
        VariableDynamicsAnalyzer(records).verdicts(window=8)


def test_component_boundary_and_articulation_are_original_graph_measures():
    graph = nx.path_graph(["x1", "x2", "x3", "x4"])
    profile = component_profile(graph, {"x2", "x3"})
    c = profile["components"][0]
    assert c["edges"] == 1 and c["cut_edges"] == 2
    assert c["boundary_variables"] == ["x1", "x4"]
    assert profile["nodes"]["x2"]["degree"] == 2
    assert profile["nodes"]["x2"]["articulation"]


def test_interaction_ignores_additive_row_column_effects_and_axis_order():
    c = np.array([[4.0, 1, 8], [0, 5, 3]])
    base = cost_table_profile(c)
    shifted = cost_table_profile(
        c + np.array([100, -20])[:, None] + np.array([5, 9, -7])
    )
    assert shifted["interaction_rms"] == pytest.approx(base["interaction_rms"])
    assert cost_table_profile(c.T)["interaction_rms"] == pytest.approx(
        base["interaction_rms"]
    )
    assert cost_table_profile(np.ones((2, 2)))["effective_rank"] == 0
