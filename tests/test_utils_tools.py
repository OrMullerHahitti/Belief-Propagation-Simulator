import numpy as np

from propflow.configs import create_random_int_table
from propflow.utils.fg_utils import FGBuilder
from propflow.utils.tools import convex_hull as ch
from propflow.utils.tools import jacobian_analysis as ja
from propflow.utils.tools.bct import BCTCreator
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


class DummyFactorGraph:
    pass


class DummyHistory:
    use_bct_history = True

    def get_bct_data(self):
        return {
            "beliefs": {"x1": [0.1, 0.05, 0.02]},
            "messages": {"f1->x1": [0.2, 0.1], "x1->f1": [0.05]},
            "assignments": {"x1": [0, 0, 0]},
            "metadata": {"total_steps": 3},
        }


def test_bct_creator_builds_tree():
    history = DummyHistory()
    creator = BCTCreator(DummyFactorGraph(), history, damping_factor=0.1)
    root = creator.create_bct("x1")
    assert root.name.startswith("x1")
    analysis = creator.analyze_convergence("x1")
    assert analysis["total_iterations"] == 3
    coeff = creator._get_damping_coefficient(2)
    assert coeff < 1.0


def test_draw_factor_graph(monkeypatch):
    graph = FGBuilder.build_cycle_graph(
        num_vars=3,
        domain_size=2,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 2},
    )
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    draw_factor_graph(graph, with_labels=False)
