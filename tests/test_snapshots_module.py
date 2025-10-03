import json
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
    saved_dir = manager.save_step(2, tmp_path)
    meta_path = saved_dir / "meta.json"
    assert meta_path.exists()
    with meta_path.open() as fh:
        payload = json.load(fh)
    assert payload["step"] == 2


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
