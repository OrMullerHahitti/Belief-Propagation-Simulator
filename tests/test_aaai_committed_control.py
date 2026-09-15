"""Fixed-pattern existence checks independent of the accelerated BP kernel."""

from itertools import product

import numpy as np
import pytest

from experiments.other.aaai_derived_control.code.committed import (
    committed_weight_interval,
)
from experiments.other.aaai_derived_control.code.control import effective_boundaries


def _gauge(values):
    return values - values[..., :1]


def _factor_response(costs, q, weights):
    domain = costs.shape[-1]
    result = np.zeros_like(q)
    for edge, table in enumerate(costs):
        for clone, alpha in enumerate((weights[edge], 1 - weights[edge])):
            for receiver in range(domain):
                result[2 * edge + clone, 0, receiver] = min(
                    alpha * table[receiver, sender] + q[2 * edge + clone, 1, sender]
                    for sender in range(domain)
                )
                result[2 * edge + clone, 1, receiver] = min(
                    alpha * table[sender, receiver] + q[2 * edge + clone, 0, sender]
                    for sender in range(domain)
                )
    return _gauge(result)


def _q_map(costs, unary, edges, weights, q, damping):
    response = _factor_response(costs, q, weights)
    belief = unary.copy()
    for edge, (left, right) in enumerate(edges):
        for clone in range(2):
            belief[left] += response[2 * edge + clone, 0]
            belief[right] += response[2 * edge + clone, 1]
    raw = np.zeros_like(q)
    for edge, (left, right) in enumerate(edges):
        for clone in range(2):
            raw[2 * edge + clone, 0] = belief[left] - response[2 * edge + clone, 0]
            raw[2 * edge + clone, 1] = belief[right] - response[2 * edge + clone, 1]
    return _gauge(damping * q + (1 - damping) * raw)


def _single_edge():
    return (
        np.array([[[0.0, 4], [3, 1]]]),
        np.array([[0.0, 0.2], [0, 0.2]]),
        np.array([[0, 1]]),
        np.array([0.5]),
        np.zeros((1, 2, 2), dtype=int),
    )


def test_exact_fixed_pattern_interval_is_narrower_than_current_q_interval():
    costs, unary, edges, weights, pattern = _single_edge()
    result = committed_weight_interval(costs, unary, edges, weights, pattern, 0)
    assert result.feasible
    assert result.lower == pytest.approx(7 / 15)
    assert result.upper == pytest.approx(8 / 15)
    current_q = result.q_intercept + 0.5 * result.q_slope
    np.testing.assert_allclose(
        effective_boundaries(costs[0], current_q), [13 / 30, 17 / 30]
    )
    # weight .55 changes no immediate active row, but cannot retain this fixed pattern.
    assert 8 / 15 < 0.55 < 17 / 30
    assert not result.contains(0.55)
    hypothetical_q = result.q_intercept + 0.55 * result.q_slope
    actual_r = _factor_response(costs, hypothetical_q, np.array([0.55]))
    hypothetical_r = result.r_intercept + 0.55 * result.r_slope
    assert np.max(abs(actual_r - hypothetical_r)) > 0.01


@pytest.mark.parametrize("damping", [0.0, 0.5, 0.9])
def test_strict_pattern_is_fixed_and_has_local_q_contraction(damping):
    costs, unary, edges, weights, pattern = _single_edge()
    result = committed_weight_interval(costs, unary, edges, weights, pattern, 0)
    fixed_q = result.q_intercept + 0.5 * result.q_slope
    fixed_r = result.r_intercept + 0.5 * result.r_slope
    np.testing.assert_allclose(_factor_response(costs, fixed_q, weights), fixed_r)
    np.testing.assert_allclose(
        _q_map(costs, unary, edges, weights, fixed_q, damping), fixed_q
    )
    perturbation = np.array([[[0, 0.003], [0, -0.004]], [[0, 0.006], [0, 0.002]]])
    perturbed = fixed_q + perturbation
    np.testing.assert_allclose(_factor_response(costs, perturbed, weights), fixed_r)
    np.testing.assert_allclose(
        _q_map(costs, unary, edges, weights, perturbed, damping) - fixed_q,
        damping * perturbation,
        atol=1e-14,
    )


def test_strict_fixed_pattern_cannot_give_one_variable_different_clone_labels():
    costs, unary, edges, weights, pattern = _single_edge()
    pattern[0, 1, 0] = 1
    result = committed_weight_interval(costs, unary, edges, weights, pattern, 0)
    assert not result.feasible
    assert not any(result.contains(weight) for weight in np.linspace(0.01, 0.99, 99))


def test_unbroken_ties_are_not_a_strict_committed_fixed_pattern():
    costs = np.zeros((1, 2, 2))
    result = committed_weight_interval(
        costs,
        np.zeros((2, 2)),
        np.array([[0, 1]]),
        np.array([0.5]),
        np.zeros((1, 2, 2), dtype=int),
    )
    assert not result.feasible
    assert not result.contains(0.5)


@pytest.mark.parametrize("controlled_edge", [None, 0, 1])
def test_all_linear_inequalities_match_direct_original_factor_scores(controlled_edge):
    rng = np.random.default_rng(1709)
    edges = np.array([[2, 0], [1, 2]])
    for _ in range(6):
        costs = rng.integers(-5, 8, size=(2, 3, 3)).astype(float)
        unary = rng.integers(-8, 9, size=(3, 3)) / 2
        weights = np.array([0.25, 0.75])
        pattern = rng.integers(0, 3, size=(2, 2, 2))
        result = committed_weight_interval(
            costs, unary, edges, weights, pattern, controlled_edge
        )
        samples = [0.13, 0.37, 0.61, 0.89]
        if result.feasible:
            samples.append((result.lower + result.upper) / 2)
        for weight in samples:
            fixed_q = result.q_intercept + weight * result.q_slope
            actual_weights = result.weight_intercept + weight * result.weight_slope
            direct = []
            for edge, clone, side, other in product(
                range(2), range(2), range(2), range(3)
            ):
                sender = pattern[edge, clone, side]
                if sender == other:
                    continue
                alpha = actual_weights[edge] if clone == 0 else 1 - actual_weights[edge]
                table = costs[edge] if side == 0 else costs[edge].T
                local_q = fixed_q[2 * edge + clone, side]
                for receiver in range(3):
                    alternative = alpha * table[other, receiver] + local_q[other]
                    selected = alpha * table[sender, receiver] + local_q[sender]
                    direct.append(alternative - selected)
            np.testing.assert_allclose(
                direct,
                result.constraint_intercept + weight * result.constraint_slope,
                atol=1e-13,
            )
            assert result.contains(weight) == bool(min(direct) > 0)


def test_global_and_single_edge_control_retain_different_other_edge_weights():
    costs = np.array([[[0, 4], [3, 1]], [[2, 6], [5, 0]]], dtype=float)
    unary = np.array([[0, 20], [0, 20], [0, 20]], dtype=float)
    edges = np.array([[0, 1], [1, 2]])
    weights = np.array([0.25, 0.75])
    pattern = np.zeros((2, 2, 2), dtype=int)
    single = committed_weight_interval(costs, unary, edges, weights, pattern, 0)
    common = committed_weight_interval(costs, unary, edges, weights, pattern, None)
    np.testing.assert_allclose(
        single.weight_intercept + 0.4 * single.weight_slope, [0.4, 0.75]
    )
    np.testing.assert_allclose(
        common.weight_intercept + 0.4 * common.weight_slope, [0.4, 0.4]
    )
    for result in (single, common):
        assert result.feasible
        assert result.contains(0.4)
        actual_weights = result.weight_intercept + 0.4 * result.weight_slope
        fixed_q = result.q_intercept + 0.4 * result.q_slope
        np.testing.assert_allclose(
            _q_map(costs, unary, edges, actual_weights, fixed_q, 0.9), fixed_q
        )
