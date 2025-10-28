import csv
import json
from types import SimpleNamespace
import numpy as np
import pytest

from propflow.bp.engine_components import Step
from propflow.configs import create_random_int_table
from propflow.core.components import Message
from propflow.snapshots.builder import (
    _labels_for_domain,
    _normalize_min_zero,
    build_snapshot_from_engine,
    extract_qr_from_step,
)
from propflow.snapshots.manager import SnapshotManager
from propflow.snapshots.types import SnapshotData, SnapshotsConfig
from propflow.utils.fg_utils import FGBuilder
from propflow.bp.engine_base import BPEngine
from propflow.core.agents import VariableAgent, FactorAgent

np.random.seed(42)


@pytest.fixture
def sample_factor_graph():
    return FGBuilder.build_cycle_graph(
        num_vars=3,
        domain_size=3,
        ct_factory=create_random_int_table,
        ct_params={"low": 1, "high": 5},
    )


@pytest.fixture
def sample_engine(sample_factor_graph):
    class DummyEngine:
        def __init__(self, graph):
            self.graph = graph
            self.var_nodes = graph.variables
            self.factor_nodes = graph.factors
            self.damping_factor = 0.25
            self.assignments = {var.name: 0 for var in graph.variables}

    return DummyEngine(sample_factor_graph)


@pytest.fixture
def sample_step(sample_engine):
    graph = sample_engine.graph
    step = Step(num=0)
    variable = graph.variables[0]
    factor = graph.factors[0]
    q_message = Message(data=np.array([1.0, 3.0, 2.0]), sender=variable, recipient=factor)
    r_message = Message(data=np.array([0.5, 0.2, 1.1]), sender=factor, recipient=variable)
    step.add_q(variable.name, [q_message])
    step.add_r(factor.name, [r_message])
    return step


def test_labels_and_normalization():
    assert _labels_for_domain(3) == ["0", "1", "2"]
    arr = np.array([3.0, 5.0, 4.0])
    np.testing.assert_allclose(_normalize_min_zero(arr), np.array([0.0, 2.0, 1.0]))


def test_extract_qr_from_step(sample_step):
    q, r = extract_qr_from_step(sample_step)
    assert ("x1", "f12") in q
    np.testing.assert_allclose(q[("x1", "f12")], np.array([0.0, 2.0, 1.0]))
    assert ("f12", "x1") in r
    np.testing.assert_allclose(r[("f12", "x1")], np.array([0.5, 0.2, 1.1]))


def test_build_snapshot_from_engine(sample_engine, sample_step):
    snapshot = build_snapshot_from_engine(0, sample_step, sample_engine)
    var_name = sample_engine.var_nodes[0].name
    factor_name = sample_engine.factor_nodes[0].name
    assert snapshot.dom[var_name] == ["0", "1", "2"]
    assert factor_name in snapshot.N_fac
    assert (var_name, factor_name) in snapshot.Q
    assert snapshot.lambda_ == pytest.approx(0.25)
    assert snapshot.beliefs == {}
    assert snapshot.assignments == sample_engine.assignments
    assert snapshot.global_cost is None
    assert snapshot.metadata["engine"] == "DummyEngine"


def test_snapshot_manager_capture_and_retain(tmp_path, sample_engine, sample_step):
    config = SnapshotsConfig(
        compute_jacobians=False,
        compute_block_norms=False,
        compute_cycles=False,
        retain_last=2,
        save_each_step=True,
        save_dir=str(tmp_path),
    )
    manager = SnapshotManager(config)
    _ = manager.capture_step(0, sample_step, sample_engine)
    manager.capture_step(1, sample_step, sample_engine)
    latest = manager.capture_step(2, sample_step, sample_engine)
    assert manager.get(0) is None
    assert manager.latest() is latest
    saved_dir = manager.save_step(2, tmp_path, save=True)
    meta_path = saved_dir / "meta.json"
    assert meta_path.exists()
    with meta_path.open() as fh:
        payload = json.load(fh)
    assert payload["context"]["step"] == 2
    assert payload["graph"]["dom"][sample_engine.var_nodes[0].name] == ["0", "1", "2"]
    q_section = payload["messages"]["Q"]
    assert q_section["file"] == "messages_q.npz"
    with np.load(saved_dir / q_section["file"]) as q_npz:
        q_key = next(iter(q_section["index"].values()))
        np.testing.assert_allclose(q_npz[q_key], np.array([0.0, 2.0, 1.0]))
    r_section = payload["messages"]["R"]
    assert r_section["file"] == "messages_r.npz"
    with np.load(saved_dir / r_section["file"]) as r_npz:
        r_key = next(iter(r_section["index"].values()))
        np.testing.assert_allclose(r_npz[r_key], np.array([0.5, 0.2, 1.1]))
    unary_section = payload["messages"]["unary"]
    assert unary_section["file"] == "unary.npz"
    assert (saved_dir / unary_section["file"]).exists()
    assert payload["analysis"]["min_idx"] == {}

    index_path = tmp_path / "index.json"
    assert index_path.exists()
    with index_path.open() as fh:
        manifest = json.load(fh)
    step_entry = next(entry for entry in manifest["steps"] if entry["step"] == 2)
    assert step_entry["messages"]["Q"] == 1
    assert step_entry["messages"]["R"] == 1
    assert step_entry["has_jacobians"] is False


def test_snapshot_manager_serializes_winners(tmp_path, sample_engine, sample_step):
    config = SnapshotsConfig(
        compute_jacobians=True,
        compute_block_norms=False,
        compute_cycles=False,
        retain_last=1,
        save_each_step=True,
        save_dir=str(tmp_path),
    )
    manager = SnapshotManager(config)
    manager.capture_step(0, sample_step, sample_engine)
    saved_dir = manager.save_step(0, tmp_path, save=True)
    meta_path = saved_dir / "meta.json"
    with meta_path.open() as fh:
        payload = json.load(fh)
    winners = payload["analysis"]["winners"]
    assert winners is not None
    edge_key = f"{sample_engine.factor_nodes[0].name}->{sample_engine.var_nodes[0].name}"
    assert edge_key in winners
    assert isinstance(winners[edge_key], dict)
    assert winners[edge_key]

    index_path = tmp_path / "index.json"
    with index_path.open() as fh:
        manifest = json.load(fh)
    step_entry = manifest["steps"][0]
    assert step_entry["has_jacobians"] is True


def test_snapshot_manager_helpers():
    manager = SnapshotManager(SnapshotsConfig(compute_jacobians=False, compute_block_norms=False, compute_cycles=False))
    dom = {"x1": ["0", "1"], "x2": ["0", "1"]}
    data = SnapshotData(
        step=0,
        lambda_=0.0,
        dom=dom,
        N_var={"x1": ["f"], "x2": ["f"]},
        N_fac={"f": ["x1", "x2"]},
        Q={
            ("x1", "f"): np.array([0.1, 0.0]),
            ("x2", "f"): np.array([0.0, 0.2]),
        },
        R={},
        cost={
            "f": lambda assignment: 0.0 if assignment.get("x1") == assignment.get("x2") else 1.0
        },
    )
    min_idx = manager._compute_min_idx(data)
    assert min_idx[("x1", "f")] == 1
    winners = manager._compute_winners(data)
    assert winners[("f", "x1", "0")]["x2"] == "0"
    assert winners[("f", "x2", "1")]["x1"] == "1"


def test_snapshot_builder_includes_history_context(sample_factor_graph):
    class BCTEngine:
        def __init__(self, graph):
            self.graph = graph
            self.var_nodes = graph.variables
            self.factor_nodes = graph.factors
            self.damping_factor = 0.4
            self.assignments = {var.name: 1 for var in graph.variables}
            belief_series = {var.name: 0.5 for var in graph.variables}
            assignment_series = {var.name: 1 for var in graph.variables}
            self.history = SimpleNamespace(
                config={"mode": "bct"},
                use_bct_history=True,
                step_beliefs={0: belief_series},
                step_assignments={0: assignment_series},
                step_costs=[10.0],
                name="history",
            )
            self.convergence_monitor = SimpleNamespace(
                get_convergence_summary=lambda: {"total_iterations": 1, "converged": False}
            )
            self.performance_monitor = SimpleNamespace(
                get_summary=lambda: {"total_steps": 1, "total_time": 0.1}
            )

        def get_beliefs(self):
            return {var.name: np.array([0.1, 0.2, 0.3]) for var in self.var_nodes}

        def calculate_global_cost(self):
            return 10.0

    engine = BCTEngine(sample_factor_graph)
    step = Step(num=0)
    var = sample_factor_graph.variables[0]
    fac = sample_factor_graph.factors[0]
    step.add_q(var.name, [Message(data=np.array([1.0, 2.0, 3.0]), sender=var, recipient=fac)])
    step.add_r(fac.name, [Message(data=np.array([0.3, 0.1, 0.2]), sender=fac, recipient=var)])

    snapshot = build_snapshot_from_engine(0, step, engine)
    assert snapshot.beliefs == engine.history.step_beliefs[0]
    assert snapshot.assignments == engine.history.step_assignments[0]
    assert snapshot.global_cost == pytest.approx(10.0)
    assert snapshot.metadata["convergence_summary"]["total_iterations"] == 1
    assert snapshot.metadata["performance_summary"]["total_steps"] == 1


def test_engine_step_messages_include_all_directions(sample_factor_graph):
    engine = BPEngine(factor_graph=sample_factor_graph)
    step = engine.step(0)
    factor_names = {f.name for f in sample_factor_graph.factors}
    variable_names = {v.name for v in sample_factor_graph.variables}
    factor_msg_groups = [
        step.messages[name] for name in step.messages if name in factor_names
    ]
    variable_msg_groups = [
        step.messages[name] for name in step.messages if name in variable_names
    ]
    assert factor_msg_groups, "expected factor recipients in step messages"
    assert variable_msg_groups, "expected variable recipients in step messages"
    assert any(
        isinstance(msg.sender, VariableAgent)
        for group in factor_msg_groups
        for msg in group
    )
    assert any(
        isinstance(msg.sender, FactorAgent)
        for group in variable_msg_groups
        for msg in group
    )


def test_engine_snapshot_sequence_and_saver(tmp_path, sample_factor_graph):
    config = SnapshotsConfig(
        compute_jacobians=False,
        compute_block_norms=False,
        compute_cycles=False,
        retain_last=2,
        save_each_step=False,
    )
    engine = BPEngine(
        factor_graph=sample_factor_graph,
        snapshots_config=config,
        use_bct_history=True,
    )
    engine.run(max_iter=3)

    snapshots = engine.snapshots
    assert len(snapshots) == 2
    assert snapshots[-1] is engine.latest_snapshot()
    steps = [rec.data.step for rec in snapshots]
    assert steps == sorted(steps)

    latest_step = steps[-1]
    json_path = engine.save_snapshot.save_json(
        tmp_path / "latest_snapshot.json",
        step=latest_step,
    )
    payload = json.loads(json_path.read_text())
    assert payload["step"] == latest_step
    assert payload["graph"]["dom"]
    assert payload["messages"]["Q"]

    csv_path = engine.save_snapshot.save_csv(tmp_path / "snapshots.csv")
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == len(snapshots)
    assert int(rows[-1]["step"]) == latest_step
