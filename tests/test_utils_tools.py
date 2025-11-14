import numpy as np
from typing import Mapping

from propflow.configs import create_random_int_table
from propflow.utils.fg_utils import FGBuilder
from propflow.utils.tools import convex_hull as ch
from propflow.utils.tools import jacobian_analysis as ja
from propflow.snapshots.types import EngineSnapshot
from propflow.utils.tools.bct import (
    BCTCreator,
    CostKey,
    MessageKey,
    SnapshotBCTBuilder,
)
from propflow.utils.tools.draw import draw_factor_graph
from propflow.utils.tools.performance import PerformanceMonitor

np.random.seed(42)


def test_convex_hull_basic_operations():
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    q = np.array([0.5, 1.0])
    lines = ch.create_lines_from_cost_table(cost, q, 0.0, 1.0)
    assert len(lines) == 4
    hull = ch.compute_convex_hull_from_lines(lines)
    assert hull.hull_lines
    line_a, line_b = lines[0], lines[1]
    assert ch.find_line_intersection(line_a, line_b) is not None
    lower = ch.convex_hull_from_cost_table(cost, q)
    upper = ch.convex_hull_from_cost_table(cost, q, hull_type="upper")
    assert lower.hull_vertices.shape[1] == 2
    assert upper.hull_vertices.shape[1] == 2
    meta = ch.compute_hierarchical_envelopes([lower, upper], envelope_type="lower")
    assert meta.individual_envelopes[0].envelope_id == 0


def test_message_coordinate_and_jacobian():
    coord = ja.MessageCoordinate(ja.MessageType.Q_MESSAGE, "x1", "f1", 0, 1)
    assert "ΔQ" in repr(coord)
    derivative = ja.FactorStepDerivative(
        factor="f1",
        from_var="x1",
        to_var="x2",
        value=0,
        domain_size=2,
        iteration=0,
        is_binary=True,
    )
    assert derivative.is_neutral
    matrix = np.ones((2, 2))
    derivative_nb = ja.FactorStepDerivative(
        factor="f1",
        from_var="x1",
        to_var="x2",
        value=matrix,
        domain_size=2,
        iteration=0,
        is_binary=False,
    )
    assert derivative_nb.get_derivative(0, 1) == 1.0
    thresholds = ja.BinaryThresholds(theta_0=0.5, theta_1=0.4)
    neutral, label = thresholds.check_neutrality(0.6)
    assert neutral and label == 0
    coords = [
        ja.MessageCoordinate(ja.MessageType.Q_MESSAGE, "x1", "f1"),
        ja.MessageCoordinate(ja.MessageType.R_MESSAGE, "f1", "x1"),
        ja.MessageCoordinate(ja.MessageType.Q_MESSAGE, "x2", "f1"),
        ja.MessageCoordinate(ja.MessageType.R_MESSAGE, "f1", "x2"),
    ]
    jac = ja.Jacobian(coords, domain_sizes={"x1": 2, "x2": 2})
    jac.set_entry(0, 1, 0.5)
    assert jac.matrix[0, 1] == 0.5
    jac.update_factor_derivative(
        "f1",
        ja.FactorStepDerivative(
            factor="f1",
            from_var="x1",
            to_var="x2",
            value=1,
            domain_size=2,
            iteration=0,
            is_binary=True,
        ),
    )
    assert "f1" in jac.factor_derivatives


class DummyMessage:
    def __init__(self, data):
        self.data = data


class DummyAgent:
    def __init__(self, messages):
        self.mailer = type("Mail", (), {"inbox": messages})()


def test_performance_monitor_tracks_metrics():
    monitor = PerformanceMonitor(track_memory=False, track_cpu=False)
    start = monitor.start_step()
    metrics = monitor.end_step(start, 0, [DummyMessage(np.ones(3)) for _ in range(2)])
    assert metrics.message_count == 2
    msg_metrics = monitor.track_message_metrics(
        0,
        [DummyAgent([DummyMessage(np.ones(2)), DummyMessage(np.ones(2))])],
        {"pruned_messages": 1, "pruning_rate": 0.25},
    )
    assert msg_metrics.total_messages == 2
    monitor.start_cycle(0)
    monitor.end_cycle(0, belief_change=0.1, cost=5.0)
    summary = monitor.get_summary()
    assert summary["total_steps"] >= 1


def _make_snapshot(
    step: int,
    *,
    lambda_: float,
    q_messages: Mapping[tuple[str, str], np.ndarray],
    r_messages: Mapping[tuple[str, str], np.ndarray],
    assignments: Mapping[str, int],
    cost_table: np.ndarray,
) -> EngineSnapshot:
    return EngineSnapshot(
        step=step,
        lambda_=lambda_,
        dom={"x1": ["0", "1"], "x2": ["0", "1"]},
        N_var={"x1": ["f"], "x2": ["f"]},
        N_fac={"f": ["x1", "x2"]},
        Q={key: np.asarray(val, dtype=float) for key, val in q_messages.items()},
        R={key: np.asarray(val, dtype=float) for key, val in r_messages.items()},
        unary={},
        beliefs={"x1": np.array([0.0, 1.0]), "x2": np.array([0.0, 1.0])},
        assignments=dict(assignments),
        metadata={},
        cost_tables={"f": np.asarray(cost_table, dtype=float)},
        cost_labels={"f": ["x1", "x2"]},
    )


def test_snapshot_bct_builder_creates_cost_leaves():
    snapshot = _make_snapshot(
        0,
        lambda_=0.0,
        q_messages={("x1", "f"): np.array([0.0, 0.5]), ("x2", "f"): np.array([0.0, 0.2])},
        r_messages={("f", "x1"): np.array([1.0, 2.0]), ("f", "x2"): np.array([1.0, 0.5])},
        assignments={"x1": 0, "x2": 1},
        cost_table=np.array([[1.0, 2.0], [3.0, 0.5]]),
    )
    builder = SnapshotBCTBuilder([snapshot])
    root = builder.belief_root("x1", 0, 0)
    creator = BCTCreator(builder.graph, root)
    contributions = creator.cost_contributions()
    assert contributions
    assert any(isinstance(key, CostKey) for key in contributions)


def test_snapshot_bct_builder_handles_damping():
    base = _make_snapshot(
        0,
        lambda_=0.0,
        q_messages={("x1", "f"): np.array([0.0, 0.0]), ("x2", "f"): np.array([0.0, 0.0])},
        r_messages={("f", "x1"): np.array([0.0, 0.0]), ("f", "x2"): np.array([0.0, 0.0])},
        assignments={"x1": 0, "x2": 0},
        cost_table=np.array([[1.0, 2.0], [3.0, 0.5]]),
    )
    damped = _make_snapshot(
        1,
        lambda_=0.5,
        q_messages={("x1", "f"): np.array([1.0, 0.0]), ("x2", "f"): np.array([0.5, 0.0])},
        r_messages={("f", "x1"): np.array([0.2, 0.1]), ("f", "x2"): np.array([0.3, 0.7])},
        assignments={"x1": 0, "x2": 0},
        cost_table=np.array([[1.0, 2.0], [3.0, 0.5]]),
    )
    builder = SnapshotBCTBuilder([base, damped])
    q_key = MessageKey("Q", "x1", "f", 1, 0)
    prev_key = MessageKey("Q", "x1", "f", 0, 0)
    children = builder.graph.children(q_key)
    assert abs(children.get(prev_key, 0.0) - 0.5) < 1e-9


def test_bct_creator_visualizes_graph(monkeypatch):
    snapshot = _make_snapshot(
        0,
        lambda_=0.0,
        q_messages={("x1", "f"): np.array([0.0, 0.0]), ("x2", "f"): np.array([0.0, 0.0])},
        r_messages={("f", "x1"): np.array([0.0, 0.0]), ("f", "x2"): np.array([0.0, 0.0])},
        assignments={"x1": 0, "x2": 0},
        cost_table=np.array([[1.0, 2.0], [3.0, 0.5]]),
    )
    builder = SnapshotBCTBuilder([snapshot])
    root = builder.belief_root("x1", 0, 0)
    creator = BCTCreator(builder.graph, root)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    fig = creator.visualize_bct(show=False)
    assert fig is not None


def test_draw_factor_graph(monkeypatch):
    graph = FGBuilder.build_cycle_graph(
        num_vars=3,
        domain_size=2,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 2},
    )
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    draw_factor_graph(graph, with_labels=False)
