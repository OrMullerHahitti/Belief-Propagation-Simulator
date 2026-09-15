"""End-to-end native gates for the causal damping fixtures."""

import numpy as np
import pytest

from experiments.other.damping_causality.code.native_experiments import (
    fixtures,
    run_native,
)


def test_splitting_preserves_objective_but_introduces_strict_two_cycle():
    problem = fixtures()["robust_path"]
    unsplit = run_native(problem, split=False, schedule={0: 0.0}, steps=64)
    split = run_native(problem, split=True, schedule={0: 0.0}, steps=64)
    np.testing.assert_array_equal(
        unsplit.assignments[-20:], np.tile([1, 0, 1], (20, 1))
    )
    np.testing.assert_array_equal(unsplit.costs[-20:], 13)
    np.testing.assert_array_equal(split.messages[-20:], split.messages[-22:-2])
    assert set(split.costs[-20:]) == {32, 61}
    assert split.summary(tail=20)["final_min_belief_margin"] >= 1
    assert split.undamped_defect[-1].max() == 32
    assert problem.exact()[0] == 13


@pytest.mark.parametrize("damping", [0.5, 0.9])
def test_damping_reaches_strict_message_fixed_point_of_undamped_map(damping):
    run = run_native(
        fixtures()["robust_path"], split=True, schedule={0: damping}, steps=512
    )
    np.testing.assert_array_equal(run.assignments[-50:], np.tile([1, 0, 1], (50, 1)))
    assert np.max(run.undamped_defect[-50:]) < 1e-10
    assert np.max(np.abs(np.diff(run.messages[-50:], axis=0))) < 1e-10
    assert run.summary()["final_min_belief_margin"] > 4.9


def test_same_state_damping_intervention_then_undamped_continuation():
    problem = fixtures()["robust_path"]
    baseline = run_native(problem, split=True, schedule={0: 0.0}, steps=384)
    switched = run_native(problem, split=True, schedule={0: 0.0, 32: 0.5}, steps=384)
    resumed = run_native(
        problem, split=True, schedule={0: 0.0, 32: 0.5, 256: 0.0}, steps=384
    )
    np.testing.assert_array_equal(baseline.messages[:32], switched.messages[:32])
    np.testing.assert_array_equal(switched.messages[:256], resumed.messages[:256])
    assert set(baseline.costs[-20:]) == {32, 61}
    np.testing.assert_array_equal(switched.costs[-20:], 13)
    np.testing.assert_array_equal(resumed.costs[-20:], 13)
    assert resumed.undamped_defect[-20:].max() == 0


def test_high_damping_counterexample_is_message_motion_not_only_decoder_ties():
    run = run_native(
        fixtures()["frustrated_triangle"], split=True, schedule={0: 0.9}, steps=2000
    )
    summary = run.summary()
    assert summary["tail_assignment_switches"] >= 20
    assert summary["max_tail_message_step_delta"] > 0.5
    assert summary["max_tail_undamped_map_defect"] > 1


def test_tiny_nonzero_damping_can_preserve_the_same_two_cycle():
    problem = fixtures()["robust_path"]
    insufficient = run_native(problem, split=True, schedule={0: 0.01}, steps=2000)
    sufficient = run_native(problem, split=True, schedule={0: 0.02}, steps=2000)
    np.testing.assert_allclose(
        insufficient.messages[-20:], insufficient.messages[-22:-2]
    )
    assert set(insufficient.costs[-20:]) == {32, 61}
    np.testing.assert_array_equal(sufficient.costs[-20:], 13)
    assert sufficient.undamped_defect[-20:].max() < 1e-10


def test_independent_map_defect_matches_next_native_undamped_step():
    run = run_native(fixtures()["robust_path"], split=True, schedule={0: 0.0}, steps=32)
    native_delta = np.stack(
        [
            np.max(np.abs(np.diff(values, axis=0)), axis=(1, 2))
            for values in (run.q, run.r)
        ],
        axis=1,
    )
    np.testing.assert_allclose(run.undamped_defect[:-1], native_delta)


def test_split_single_edge_converges_without_damping():
    run = run_native(fixtures()["single_edge"], split=True, schedule={0: 0.0}, steps=64)
    np.testing.assert_array_equal(run.costs[-20:], 0.5)
    np.testing.assert_array_equal(run.assignments[-20:], 0)
    assert run.undamped_defect[-20:].max() < 1e-12
    assert run.summary()["final_min_belief_margin"] > 2


def test_invalid_schedule_rejected_before_execution():
    with pytest.raises(ValueError, match="old-Q retention"):
        run_native(fixtures()["robust_path"], split=True, schedule={0: 1.0})
