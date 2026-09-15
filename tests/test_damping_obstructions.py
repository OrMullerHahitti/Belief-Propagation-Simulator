"""Exact global obstructions and native-arithmetic fixed-point checks."""

from fractions import Fraction as F

import numpy as np
import pytest

from experiments.other.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    PairwiseProblem,
    gauge,
)
from experiments.other.damping_generalization.code.obstructions import (
    anti_equality_cycle,
    biased_cycle_bad_fixed_point,
    biased_cycle_positive_subsolution,
    certify_assignment,
    enumerate_certificates,
)


@pytest.mark.parametrize("size", [3, 4, 5, 6])
def test_cycle_commitment_requires_and_suffices_for_bipartition(size):
    problem = anti_equality_cycle(size, [-1] + [0] * (size - 1))
    certificates = enumerate_certificates(problem)
    strict = [c for c in certificates if c.strict]
    assert len(strict) == (2 if size % 2 == 0 else 0)
    for certificate in strict:
        x = certificate.assignment
        assert all(x[u] != x[v] for u, v in problem.edges)
        assert certificate.commitment_margin == (3 if x[0] == 0 else 4)


def test_strict_best_responses_need_not_have_a_committed_fixed_point():
    problem = anti_equality_cycle(3, [-1, -1, -1])
    certificates = enumerate_certificates(problem)
    assert any(c.local_margin > 0 for c in certificates)
    assert not any(c.strict for c in certificates)


@pytest.mark.parametrize("damping", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("assignment,cost", [([1, 0, 1, 0], 0), ([0, 1, 0, 1], 1)])
def test_good_and_suboptimal_attractors_exist_for_every_damping(
    damping, assignment, cost
):
    problem = anti_equality_cycle(4, [-1, 0, 0, 0])
    certificate = certify_assignment(problem, assignment)
    assert certificate.strict
    optimum, _ = problem.exact()
    assert optimum == 0
    assert problem.cost(np.array(assignment)) == cost
    kernel = PairwiseKernel(problem, damping=damping)
    target = np.asarray(certificate.q, dtype=float)
    kernel.q = target.copy()
    kernel.r = np.asarray(certificate.r, dtype=float)
    kernel.unary_r = np.repeat(problem.unary[:, None, :] * 0.5, 2, axis=1)
    kernel.unary_q = kernel.beliefs()[:, None, :] - kernel.unary_r
    # perturb unequal clones, then verify their exact local contraction rate.
    error = np.arange(kernel.q.size).reshape(kernel.q.shape) % 3 / 8
    error -= error[..., :1].copy()
    kernel.q += error
    for step in range(1, 9):
        kernel.step()
        np.testing.assert_allclose(
            gauge(kernel.q), target + damping**step * error, rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(kernel.assignment, assignment)
        assert kernel.cost == cost


def test_asymmetry_selectively_invalidates_the_worse_commitment_certificate():
    problem = anti_equality_cycle(4, [-1, 0, 0, 0])
    good, bad = [1, 0, 1, 0], [0, 1, 0, 1]
    assert certify_assignment(problem, bad).commitment_margin == 3
    assert certify_assignment(problem, bad, F(7, 8)).commitment_margin == 0
    assert certify_assignment(problem, bad, F(19, 20)).commitment_margin == -F(3, 5)
    assert certify_assignment(problem, good, F(19, 20)).commitment_margin == F(2, 5)


def test_general_cross_difference_formula_matches_direct_factor_minimizers():
    problem = PairwiseProblem(
        np.array([[0, 1], [1, 2], [2, 0]]),
        np.array(
            [
                [[0, 2, 7], [8, 4, 1], [3, 6, 9]],
                [[2, 8, 0], [5, 1, 9], [3, 7, 4]],
                [[1, 2, 3], [7, 6, 0], [4, 5, 8]],
            ]
        ),
        np.array([[0, 2, 4], [5, 1, 0], [0, 3, 8]]),
        "certificate_orientation_test",
        0,
        "triangle",
    )
    weights = [F(1, 3), F(3, 5), F(4, 7)]
    for certificate in enumerate_certificates(problem):
        assignment = certificate.assignment
        uneven = certify_assignment(problem, assignment, weights)
        margins = []
        for edge, (u, v) in enumerate(problem.edges):
            for clone, w in enumerate((weights[edge], 1 - weights[edge])):
                for axis, (sender, table) in enumerate(
                    ((u, problem.costs[edge]), (v, problem.costs[edge].T))
                ):
                    for receiving_label in range(problem.d):
                        values = [
                            sum(
                                (
                                    uneven.q[2 * edge + clone, axis, a],
                                    w * F(str(table[a, receiving_label])),
                                )
                            )
                            for a in range(problem.d)
                        ]
                        margins.extend(
                            values[a] - values[assignment[sender]]
                            for a in range(problem.d)
                            if a != assignment[sender]
                        )
        assert uneven.commitment_margin == min(margins)
        # .5/.5 maximizes the minimum consistency margin for this same x.
        assert certificate.commitment_margin >= uneven.commitment_margin


def test_certificate_rejects_invalid_weights_and_assignments():
    problem = anti_equality_cycle(3)
    for weights in (0, 1, [F(1, 2)]):
        with pytest.raises(ValueError):
            certify_assignment(problem, [0, 1, 0], weights)
    with pytest.raises(ValueError):
        certify_assignment(problem, [0, 1, 2])


def test_loss_of_full_commitment_does_not_remove_the_bad_fixed_assignment():
    problem = anti_equality_cycle(4, [-1, 0, 0, 0])
    q = np.array(
        [
            [16, -21],
            [34, -36],
            [-18, 20],
            [-36, 37],
            [20, -18],
            [37, -36],
            [-21, 16],
            [-36, 34],
        ],
        dtype=object,
    ) * F(1, 5)
    r = np.empty_like(q)
    for clone in range(len(q)):
        threshold = F(19, 5) if clone % 2 == 0 else F(1, 5)
        for endpoint in range(2):
            incoming = q[clone, 1 - endpoint]
            assert abs(incoming) != threshold
            r[clone, endpoint] = -max(-threshold, min(threshold, incoming))
    beliefs = np.array([-1, 0, 0, 0], dtype=object)
    ends = np.repeat(problem.edges, 2, axis=0)
    for clone, (u, v) in enumerate(ends):
        beliefs[u] += r[clone, 0]
        beliefs[v] += r[clone, 1]
    np.testing.assert_array_equal(beliefs, np.array([35, -37, 38, -37]) * F(1, 5))
    for clone, (u, v) in enumerate(ends):
        assert q[clone, 0] == beliefs[u] - r[clone, 0]
        assert q[clone, 1] == beliefs[v] - r[clone, 1]
    assert problem.cost(np.array(beliefs < 0, dtype=int)) == 1
    assert certify_assignment(problem, [0, 1, 0, 1], F(19, 20)).strict is False


@pytest.mark.parametrize(
    "weight", [F(1, 2), F(7, 8), F(9, 10), F(15, 16), F(19, 20), F(23, 24), F(31, 32)]
)
def test_bad_fixed_point_continues_through_all_branches_to_exact_threshold(weight):
    problem = anti_equality_cycle(4, [-1, 0, 0, 0])
    q, r = biased_cycle_bad_fixed_point(weight)
    ends = np.repeat(problem.edges, 2, axis=0)
    beliefs = np.array([-1, 0, 0, 0], dtype=object)
    for clone, (u, v) in enumerate(ends):
        threshold = 4 * (weight if clone % 2 == 0 else 1 - weight)
        for endpoint in range(2):
            incoming = q[clone, 1 - endpoint]
            assert r[clone, endpoint] == -max(-threshold, min(threshold, incoming))
        beliefs[u] += r[clone, 0]
        beliefs[v] += r[clone, 1]
    np.testing.assert_array_equal(q, beliefs[ends] - r)
    np.testing.assert_array_equal(np.array(beliefs < 0, dtype=int), [0, 1, 0, 1])
    assert problem.cost(np.array(beliefs < 0, dtype=int)) == 1


def test_strong_cycle_uniqueness_bounds_are_strict_above_threshold():
    for weight in (F(31, 32) + F(1, 10**6), F(49, 50), F(99, 100)):
        strong, weak = 4 * weight, 4 * (1 - weight)
        assert -1 + 8 * weak < 0
        assert -2 * strong + 14 * weak < 0
        assert -2 * strong + 15 * weak < -weak
    assert -1 + 8 * 4 * (1 - F(31, 32)) == 0


@pytest.mark.parametrize("damping", [0.0, 0.5, 0.9])
def test_high_weight_escape_and_warm_baseline_restore_from_bad_attractor(damping):
    problem = anti_equality_cycle(4, [-1, 0, 0, 0])
    certificate = certify_assignment(problem, [0, 1, 0, 1])
    kernel = PairwiseKernel(problem, weights=0.99, damping=damping)
    kernel.q = np.asarray(certificate.q, dtype=float)
    kernel.r = np.asarray(certificate.r, dtype=float)
    kernel.unary_r = np.repeat(problem.unary[:, None, :] * 0.5, 2, axis=1)
    kernel.unary_q = kernel.beliefs()[:, None, :] - kernel.unary_r
    signs = np.array([1, -1, 1, -1])[kernel.ends]
    for _ in range(500):
        kernel.step()
        if np.all(gauge(kernel.q)[:, :, 1] * signs < -2):
            break
    else:
        pytest.fail("the fixed-horizon diagnostic did not enter the proved restore set")
    kernel.weights[:] = 0.5
    kernel.advance(20)
    np.testing.assert_array_equal(kernel.assignment, [1, 0, 1, 0])
    assert kernel.cost == 0
    assert np.all(gauge(kernel.q)[:, :, 1] * signs < -2)


@pytest.mark.parametrize("cost,bias", [(F(1), F(1, 4)), (F(3), F(1)), (F(10), F(4))])
def test_parameter_family_continuation_and_uniqueness_bounds(cost, bias):
    ends = np.repeat(np.array([(i, (i + 1) % 4) for i in range(4)]), 2, axis=0)
    threshold = 1 - bias / (8 * cost)
    for weight in (F(1, 2), threshold):
        q, r = biased_cycle_bad_fixed_point(weight, cost, bias)
        beliefs = np.array([-bias, F(0), F(0), F(0)])
        for clone, (u, v) in enumerate(ends):
            clip_limit = cost * (weight if clone % 2 == 0 else 1 - weight)
            for endpoint in range(2):
                incoming = q[clone, 1 - endpoint]
                assert r[clone, endpoint] == -max(
                    -clip_limit, min(clip_limit, incoming)
                )
            beliefs[u] += r[clone, 0]
            beliefs[v] += r[clone, 1]
        np.testing.assert_array_equal(q, beliefs[ends] - r)
        np.testing.assert_array_equal(np.array(beliefs < 0, dtype=int), [0, 1, 0, 1])
    weight = (threshold + 1) / 2
    strong, weak = cost * weight, cost * (1 - weight)
    assert -bias + 8 * weak < 0
    assert -2 * strong + 15 * weak < -weak


@pytest.mark.parametrize("cost,bias", [(F(1), F(1, 4)), (F(4), F(1)), (F(10), F(4))])
def test_common_subsolution_traps_baseline_bad_state_below_escape_threshold(cost, bias):
    q, r = biased_cycle_positive_subsolution(cost, bias)
    ends = np.repeat(np.array([(i, (i + 1) % 4) for i in range(4)]), 2, axis=0)
    signs = np.array([1, -1, 1, -1], dtype=object)
    beliefs = np.array([-bias, F(0), F(0), F(0)])
    for clone, (u, v) in enumerate(ends):
        beliefs[u] += r[clone, 0]
        beliefs[v] += r[clone, 1]
    np.testing.assert_array_equal(beliefs * signs, np.full(4, 5 * bias / 4))
    raw_q = beliefs[ends] - r
    assert np.all((raw_q - q) * signs[ends] >= 0)
    np.testing.assert_array_equal(raw_q[::2], q[::2])
    np.testing.assert_array_equal(
        raw_q[1::2] * signs[ends][1::2], np.full((4, 2), 9 * bias / 8)
    )
    threshold = 1 - bias / (8 * cost)
    for weight in (F(1, 2), (F(1, 2) + threshold) / 2, threshold):
        for clone in range(8):
            limit = cost * (weight if clone % 2 == 0 else 1 - weight)
            for endpoint in range(2):
                incoming = q[clone, 1 - endpoint]
                assert r[clone, endpoint] == -max(-limit, min(limit, incoming))
    baseline_q, baseline_r = biased_cycle_bad_fixed_point(F(1, 2), cost, bias)
    assert np.all((baseline_q - q) * signs[ends] > 0)
    assert np.all((baseline_r - r) * signs[ends] > 0)
