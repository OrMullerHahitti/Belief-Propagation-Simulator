"""Native state restoration, damping release, and independent menu-optimum checks."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from experiments.aamas.late_split import core, run
from experiments.aaai.code.problems import capture_original
from experiments.aaai.code.merge import score_assignment
from propflow import DampingEngine, FGBuilder, MidRunSplitEngine, MinSumComputator
from propflow.configs import create_random_int_table


def graph(seed=4):
    np.random.seed(seed)
    return FGBuilder.build_random_graph(
        num_vars=5,
        domain_size=3,
        density=0.6,
        seed=seed,
        ct_factory=create_random_int_table,
        ct_params={"low": 1, "high": 10},
    )


def assert_messages_equal(left, right):
    for a, b in zip(
        left.var_nodes + left.factor_nodes, right.var_nodes + right.factor_nodes
    ):
        assert a.name == b.name
        for xs, ys in [
            (a.mailer.inbox, b.mailer.inbox),
            (a.last_iteration, b.last_iteration),
        ]:
            assert len(xs) == len(ys)
            for x, y in zip(xs, ys):
                assert (x.sender.name, x.recipient.name) == (
                    y.sender.name,
                    y.recipient.name,
                )
                np.testing.assert_array_equal(x.data, y.data)


def test_saved_input_retains_original_dtype_order_and_cost(tmp_path):
    original = graph()
    core.save_input(original, tmp_path / "input.npz")
    loaded = core.load_input(tmp_path / "input.npz")
    assert core.input_fingerprint(original) == core.input_fingerprint(loaded)
    for x, y in zip(original.factors, loaded.factors):
        assert x.cost_table.dtype == y.cost_table.dtype
        assert x.connection_number == y.connection_number


def test_prefix_matches_dms_and_release_matches_native_undamped_split(tmp_path):
    original = graph()
    cfg = core.Config(prefix_steps=9, post_steps=8, tail_steps=4)
    actual = core.make_engine(deepcopy(original), cfg, 9)
    control = DampingEngine(
        deepcopy(original),
        damping_factor=0.9,
        computator=MinSumComputator(),
        normalize_messages=True,
        anytime=False,
        snapshot_manager=core.TraceSnapshots(),
    )
    for i in range(9):
        a, b = core.advance(actual, i), core.advance(control, i)
        assert a.assignments == b.assignments and a.global_cost == b.global_cost
        assert_messages_equal(actual, control)
    checkpoint = core.Checkpoint.capture(actual, 9, core.input_fingerprint(original))
    undamped = MidRunSplitEngine(
        deepcopy(original),
        split_at_iter=9,
        split_factor=0.5,
        transfer_mode="transfer",
        computator=MinSumComputator(),
        normalize_messages=True,
        anytime=False,
        snapshot_manager=core.TraceSnapshots(),
    )
    checkpoint.restore(undamped)
    for i in range(9, 17):
        a, b = core.advance(actual, i), core.advance(undamped, i)
        assert a.assignments == b.assignments and a.global_cost == b.global_cost
        assert_messages_equal(actual, undamped)
    assert actual.damping_factor == 0
    assert actual.split_events[0]["iteration"] == 9


@pytest.mark.parametrize("next_iteration", [5, 6, 9])
def test_portable_checkpoint_preserves_messages_and_normalization_phase(
    tmp_path, next_iteration
):
    original = graph()
    cfg = core.Config(prefix_steps=20, post_steps=8, tail_steps=4)
    engine = core.make_engine(deepcopy(original), cfg, 20)
    for i in range(next_iteration):
        core.advance(engine, i)
    cp = core.Checkpoint.capture(
        engine, next_iteration, core.input_fingerprint(original)
    )
    cp.save(tmp_path / "state.json.gz")
    restored = core.make_engine(deepcopy(original), cfg, 20)
    core.Checkpoint.load(tmp_path / "state.json.gz").restore(restored)
    assert_messages_equal(engine, restored)
    for i in range(next_iteration, 28):
        a, b = core.advance(engine, i), core.advance(restored, i)
        assert a.assignments == b.assignments and a.global_cost == b.global_cost
        assert_messages_equal(engine, restored)


def test_checkpoint_rejects_different_input():
    g = graph()
    cfg = core.Config(prefix_steps=4, post_steps=4, tail_steps=4)
    engine = core.make_engine(g, cfg, 4)
    core.advance(engine, 0)
    cp = core.Checkpoint.capture(engine, 1, core.input_fingerprint(g))
    with pytest.raises(ValueError, match="graph differs"):
        cp.restore(core.make_engine(graph(5), cfg, 4))


def test_menu_bb_matches_independent_exhaustive_optimum():
    g = graph()
    names, axes, tables = capture_original(g)
    # nonconsecutive labels exercise the reduced-table index-to-label mapping
    a = {v: 0 for v in names}
    b = {v: 2 for v in names}
    assignments = np.array([[a[v] for v in names], [b[v] for v in names]] * 3)
    costs = np.array(
        [score_assignment(dict(zip(names, row)), tables, axes) for row in assignments]
    )
    trace = {
        "assignments": assignments,
        "costs": costs,
        "variable_names": np.array(names),
    }
    cfg = core.Config(prefix_steps=4, post_steps=6, tail_steps=6, bb_seconds=2)
    result = core.merge_tail(g, trace, cfg)
    optimum = min(
        score_assignment(dict(zip(names, row)), tables, axes)
        for row in product([0, 2], repeat=len(names))
    )
    assert result["tail_kind"] == "period_two"
    assert result["bb"]["complete"]
    assert result["bb"]["cost"] == optimum
    assert result["bb"]["cost"] <= result["mgm"]["cost"] <= min(costs)
    assert all(x in [0, 2] for x in result["bb"]["assignment"].values())


def test_tail_requires_repeated_assignments_not_only_equal_costs():
    assert core.tail_kind(np.array([[0], [1]] * 3), 6) == "period_two"
    assert core.tail_kind(np.zeros((6, 2)), 6) == "fixed"
    assert core.tail_kind(np.array([[0], [1], [2]] * 2), 6) == "other"


def test_case_pipeline_selects_earliest_minimum_and_replays(tmp_path, monkeypatch):
    cfg = core.Config(prefix_steps=12, post_steps=8, tail_steps=4, bb_seconds=0.2)
    monkeypatch.setitem(run.BENCHMARKS, "tiny", graph)
    case = tmp_path / "tiny_4"
    case.mkdir()
    reference = DampingEngine(
        graph(),
        damping_factor=0.9,
        normalize_messages=True,
        anytime=False,
        snapshot_manager=core.TraceSnapshots(),
    )
    costs = [core.advance(reference, i).global_cost for i in range(12)]
    np.savez_compressed(
        case / "references.npz", DMS=np.round(costs, 4), DMS_iterations=np.arange(12)
    )
    fixed = run.run_case((str(tmp_path), "tiny", 4, "fixed", cfg, False))
    best = run.run_case((str(tmp_path), "tiny", 4, "best", cfg, False))
    assert fixed["split_before_iteration"] == 12
    assert best["split_before_iteration"] == int(np.argmin(costs)) + 1
    assert best["checkpoint_replay_max_error"] == 0
    assert fixed["cost_verification_max_error"] == 0
    with pytest.raises(FileExistsError):
        run.run_case((str(tmp_path), "tiny", 4, "fixed", cfg, False))
    assert run.run_case((str(tmp_path), "tiny", 4, "best", cfg, True)) == best


def test_cost_verifier_detects_corruption():
    g = graph()
    engine = core.make_engine(deepcopy(g), core.Config(), 1000)
    trace = core.trace_arrays(
        [core.advance(engine, 0)], [v.name for v in engine.var_nodes]
    )
    core.verify_costs(g, trace)
    trace["costs"][0] += 1
    with pytest.raises(RuntimeError, match="reconstruction"):
        core.verify_costs(g, trace)


def test_baseline_validation_matches_actual_csv_precision(tmp_path):
    np.savez_compressed(tmp_path / "references.npz", DMS=[100.1235], DMS_iterations=[0])
    trace = {"costs": np.array([100.123456789]), "iterations": np.array([0])}
    assert 0 < run.validate_prefix(tmp_path, trace) < 0.00005
    trace["costs"][0] += 0.001
    with pytest.raises(RuntimeError, match="prefix mismatch"):
        run.validate_prefix(tmp_path, trace)
