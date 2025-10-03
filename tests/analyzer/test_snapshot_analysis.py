import numpy as np
import pytest

from analyzer import EngineSnapshotRecorder
from analyzer.reporting import SnapshotAnalyzer
from propflow.bp.engine_components import Step
from propflow.core.components import Message


class _DemoEngine:
    """Small deterministic engine for snapshot analysis tests."""

    def __init__(self):
        var_a = type("Node", (), {"name": "A"})()
        var_b = type("Node", (), {"name": "B"})()
        factor_f = type("Node", (), {"name": "F"})()

        step0 = Step(num=0)
        step0.q_messages["A"] = [Message(np.array([1.0, 2.0]), var_a, factor_f)]
        step0.r_messages["F"] = [Message(np.array([3.0, 1.0]), factor_f, var_b)]

        step1 = Step(num=1)
        step1.q_messages["A"] = [Message(np.array([0.8, 1.4]), var_a, factor_f)]
        step1.r_messages["F"] = [Message(np.array([2.5, 1.2]), factor_f, var_b)]

        self._steps = [step0, step1]
        self._assignments = [
            {"A": 0, "B": 1},
            {"A": 0, "B": 0},
        ]
        self._costs = [12.0, 9.5]
        self._cursor = -1

    def step(self, iteration):
        self._cursor = iteration
        return self._steps[iteration]

    @property
    def assignments(self):
        if self._cursor < 0:
            return {}
        return self._assignments[self._cursor]

    def calculate_global_cost(self):
        if self._cursor < 0:
            return None
        return self._costs[self._cursor]


@pytest.fixture
def sample_snapshots():
    engine = _DemoEngine()
    recorder = EngineSnapshotRecorder(engine)
    recorder.record_run(max_steps=2)
    return recorder.snapshots


def test_snapshot_analyzer_produces_metrics(sample_snapshots):
    analyzer = SnapshotAnalyzer.from_snapshots(sample_snapshots)
    report = analyzer.build_report()

    assert report.metadata["num_steps"] == 2
    assert set(report.node_metrics.index) == {"A", "B", "F"}
    assert "impact_score" in report.node_metrics.columns

    expected_edges = {("A", "F"), ("F", "B")}
    assert set(report.edge_metrics.index) == expected_edges

    assert not report.component_metrics.empty
    component = report.component_metrics.iloc[0]
    assert component["size"] == 3

    final_step = report.step_metrics.iloc[-1]
    assert final_step["assignment_change_count"] == 1
    assert "B" in final_step["assignment_changes"]


def test_report_markdown_and_figures(sample_snapshots):
    analyzer = SnapshotAnalyzer.from_snapshots(sample_snapshots)
    report = analyzer.build_report()

    summary = report.to_markdown()
    assert "Snapshot Analysis Report" in summary
    assert "High-Impact Nodes" in summary

    figures = report.build_figures(show=False)
    assert {"cost_trajectory", "message_volume", "message_graph"}.issubset(figures.keys())
