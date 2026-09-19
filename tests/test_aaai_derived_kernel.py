"""Native execution and objective gates for the paper-derived control kernel."""

import numpy as np
import pytest

from experiments.aamas.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    extract_problem,
    make_problem,
    make_small_problem,
    native_parity,
)


@pytest.mark.parametrize("topology", ["bowtie", "k4"])
@pytest.mark.parametrize("damping", [0.0, 0.5, 0.9])
def test_actual_native_dms_assignment_cost_and_message_parity(topology, damping):
    report = native_parity(make_problem(topology, 13), damping, intervention=True)
    assert report["assignment_mismatch_steps"] == []
    assert report["max_cost_error"] < 1e-8
    assert report["max_gauge_message_error"] < 1e-8
    assert report["graph_diameter"] == report["native_graph_diameter"]


def test_unary_first_enters_after_initial_q_update():
    problem = make_problem("k4", 4)
    kernel = PairwiseKernel(problem)
    np.testing.assert_array_equal(kernel.beliefs(), 0)
    kernel.step()
    np.testing.assert_array_equal(kernel.q, 0)
    np.testing.assert_array_equal(kernel.unary_q, 0)
    np.testing.assert_allclose(
        kernel.unary_r[:, 0],
        (problem.unary - problem.unary.min(axis=1, keepdims=True)) / 2,
    )


def test_frustrated_case_uses_native_raw_argmin_and_periodic_normalization():
    problem = make_small_problem("bowtie", "frustrated", 2007)
    report = native_parity(problem, damping=0.0, steps=128, intervention=True)
    assert report["assignment_mismatch_steps"] == []
    assert report["max_raw_message_error"] == 0


def test_sub_tolerance_unary_preference_is_not_treated_as_a_tie():
    problem = make_problem("edge", 0, d=2)
    problem.costs[:] = 0
    problem.unary[:] = 0
    problem.unary[0, 1] = -3.47e-18
    kernel = PairwiseKernel(problem).step()
    assert kernel.assignment[0] == 1
    report = native_parity(problem, steps=8)
    assert report["assignment_mismatch_steps"] == []


def test_weight_changes_preserve_messages_and_original_objective():
    kernel = PairwiseKernel(make_problem("bowtie", 2)).advance(11)
    before = kernel.clone()
    kernel.weights[:] = np.linspace(0.05, 0.95, len(kernel.weights))
    kernel.damping = 0.7
    np.testing.assert_array_equal(kernel.q, before.q)
    np.testing.assert_array_equal(kernel.r, before.r)
    np.testing.assert_allclose(
        kernel.tables.reshape(-1, 2, 3, 3).sum(axis=1), kernel.problem.costs
    )
    assert kernel.cost == before.cost


def test_native_round_trip_preserves_axes_and_unary_costs():
    problem = make_problem("bowtie", 9)
    rebuilt = extract_problem(problem.to_native(), problem.family, problem.seed)
    np.testing.assert_array_equal(problem.edges, rebuilt.edges)
    np.testing.assert_array_equal(problem.costs, rebuilt.costs)
    np.testing.assert_array_equal(problem.unary, rebuilt.unary)
    assert problem.exact()[0] == rebuilt.exact()[0]


def test_invalid_controls_fail_before_message_mutation():
    kernel = PairwiseKernel(make_problem("k4", 6)).advance(3)
    old_q, old_r = kernel.q.copy(), kernel.r.copy()
    kernel.weights[0] = 0
    with pytest.raises(ValueError):
        kernel.step()
    np.testing.assert_array_equal(kernel.q, old_q)
    np.testing.assert_array_equal(kernel.r, old_r)
