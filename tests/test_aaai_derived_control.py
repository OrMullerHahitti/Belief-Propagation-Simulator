"""Exact small-factor gates for the theory-derived split intervention search."""

from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.aamas.aaai_derived_control.code.control import (
    effective_boundaries,
    selectors,
)
from experiments.aamas.aaai_derived_control.code.intervals import (
    decoded_region_candidates,
)


def _exact_selectors(cost, q, weight):
    """Evaluate both directions directly with rational scalar arithmetic."""
    domain = len(cost)
    result = []
    for clone in range(2):
        alpha = weight if clone == 0 else 1 - weight
        directions = []
        for direction in range(2):
            winners = []
            for receiver in range(domain):
                values = []
                for sender in range(domain):
                    entry = (
                        cost[sender, receiver]
                        if direction == 0
                        else cost[receiver, sender]
                    )
                    values.append(alpha * int(entry) + int(q[clone, direction, sender]))
                winners.append(min(range(domain), key=values.__getitem__))
            directions.append(winners)
        result.append(directions)
    return np.asarray(result)


def _exact_partition(cost, q):
    """Find changes by directly testing all rational line-intersection cells."""
    domain = len(cost)
    points = {Fraction(0), Fraction(1)}
    for clone, direction, receiver in product(range(2), range(2), range(domain)):
        for first, second in product(range(domain), repeat=2):
            if first >= second:
                continue
            if direction == 0:
                slope = int(cost[first, receiver] - cost[second, receiver])
            else:
                slope = int(cost[receiver, first] - cost[receiver, second])
            if slope == 0:
                continue
            alpha = Fraction(
                int(q[clone, direction, second] - q[clone, direction, first]),
                slope,
            )
            weight = alpha if clone == 0 else 1 - alpha
            if 0 < weight < 1:
                points.add(weight)
    ordered = sorted(points)
    signatures = [
        _exact_selectors(cost, q, (left + right) / 2)
        for left, right in zip(ordered, ordered[1:])
    ]
    boundaries = [
        ordered[index + 1]
        for index in range(len(signatures) - 1)
        if not np.array_equal(signatures[index], signatures[index + 1])
    ]
    return boundaries


def test_boundary_is_a_real_lower_envelope_change_in_one_direction():
    cost = np.array([[0, 4], [3, 1]])
    q = np.array([[[0, 1], [0, 100]], [[100, 0], [100, 0]]])
    np.testing.assert_allclose(effective_boundaries(cost, q), [1 / 3])
    left = selectors(cost, q, 1 / 3 - 1e-5)
    right = selectors(cost, q, 1 / 3 + 1e-5)
    assert left[0, 0, 1] == 0
    assert right[0, 0, 1] == 1
    assert np.count_nonzero(left != right) == 1


def test_crossing_between_inactive_lines_is_not_an_effective_boundary():
    cost = np.array([[0, 0, 0], [1, 2, 3], [3, 2, 1]])
    q = np.full((2, 2, 3), 100)
    q[:, :, 0] = 0
    q[0, 0] = [0, 1, 0]
    # the second and third rows tie at weight .5, above the winning first row.
    assert 0.5 * cost[1, 0] + q[0, 0, 1] == 0.5 * cost[2, 0]
    assert effective_boundaries(cost, q).size == 0


@pytest.mark.parametrize("domain", [2, 3, 5])
def test_boundaries_match_exact_rational_active_changes(domain):
    rng = np.random.default_rng(20260915 + domain)
    for _ in range(12):
        cost = rng.integers(-7, 13, size=(domain, domain))
        q = rng.integers(-5, 6, size=(2, 2, domain))
        exact = _exact_partition(cost, q)
        actual = effective_boundaries(cost, q)
        np.testing.assert_allclose(actual, [float(value) for value in exact])
        bounds = [Fraction(0), *exact, Fraction(1)]
        for left, right in zip(bounds, bounds[1:]):
            midpoint = (left + right) / 2
            np.testing.assert_array_equal(
                selectors(cost, q, float(midpoint)),
                _exact_selectors(cost, q, midpoint),
            )


def test_boundary_set_respects_endpoint_and_clone_orientation():
    cost = np.array([[0, 4], [3, 1]])
    q = np.array([[[0, 1], [0, 100]], [[100, 0], [100, 0]]])
    original = effective_boundaries(cost, q)
    np.testing.assert_allclose(effective_boundaries(cost.T, q[:, ::-1]), original)
    np.testing.assert_allclose(
        effective_boundaries(cost, q[::-1]), np.sort(1 - original)
    )


def test_coincident_lines_and_endpoint_ties_do_not_add_interior_boundaries():
    cost = np.array([[0, 0], [0, 0]])
    q = np.zeros((2, 2, 2))
    assert effective_boundaries(cost, q).size == 0
    np.testing.assert_array_equal(selectors(cost, q, 0.4), np.zeros((2, 2, 2)))


def test_complementary_boundary_actions_preserve_every_original_assignment_cost():
    costs = np.array([[[0, 4], [3, 1]], [[2, -1], [5, 0]]])
    edges = [(0, 1), (2, 0)]
    unary = np.array([[0.25, 0], [-0.5, 0.2], [0.1, 0]])
    q = np.array([[[0, 1], [0, 100]], [[100, 0], [100, 0]]])
    weights = [0.03, 0.5, 0.97]
    for table in costs:
        boundaries = np.r_[0, effective_boundaries(table, q), 1]
        weights.extend(((boundaries[:-1] + boundaries[1:]) / 2).tolist())
    for first, second in product(weights, repeat=2):
        split = [
            (weight * table, (1 - weight) * table)
            for weight, table in zip((first, second), costs)
        ]
        for assignment in product(range(2), repeat=3):
            original = sum(unary[i, value] for i, value in enumerate(assignment))
            decomposed = original
            for edge, (left, right) in enumerate(edges):
                a, b = assignment[left], assignment[right]
                original += costs[edge, a, b]
                decomposed += split[edge][0][a, b] + split[edge][1][a, b]
            assert decomposed == pytest.approx(original, abs=1e-12)


def _small_state(costs, unary, edges, weights):
    costs = np.asarray(costs)
    weights = np.asarray(weights)
    return SimpleNamespace(
        problem=SimpleNamespace(costs=costs, d=costs.shape[1], unary=np.asarray(unary)),
        ends=np.repeat(np.asarray(edges), 2, axis=0),
        weights=weights,
        tables=np.asarray(
            [alpha * table for table, w in zip(costs, weights) for alpha in (w, 1 - w)]
        ),
    )


def _exact_beliefs(state, q, changed_edge, weight):
    """Compute next beliefs with direct rational minimizations on every factor."""
    domain = state.problem.d
    beliefs = [[Fraction(float(value)) for value in row] for row in state.problem.unary]
    for edge, table in enumerate(state.problem.costs):
        w = weight if edge == changed_edge else Fraction(float(state.weights[edge]))
        left, right = state.ends[2 * edge]
        for clone, alpha in enumerate((w, 1 - w)):
            for receiver in range(domain):
                beliefs[left][receiver] += min(
                    alpha * int(table[receiver, sender])
                    + int(q[2 * edge + clone, 1, sender])
                    for sender in range(domain)
                )
                beliefs[right][receiver] += min(
                    alpha * int(table[sender, receiver])
                    + int(q[2 * edge + clone, 0, sender])
                    for sender in range(domain)
                )
    return beliefs


def _exact_decode(state, q, changed_edge, weight):
    return tuple(
        min(range(state.problem.d), key=row.__getitem__)
        for row in _exact_beliefs(state, q, changed_edge, weight)
    )


def test_decoding_boundary_inside_fixed_active_region_can_improve_original_cost():
    cost = np.array([[-3, 7], [-1, 0]])
    q = np.array([[[0, 0], [-2, -1]], [[-3, -3], [3, 0]]])
    state = _small_state([cost], [[0.5, -1], [0, -0.5]], [[0, 1]], [0.5])
    boundaries = effective_boundaries(cost, q)
    np.testing.assert_allclose(boundaries, [0.7])
    # both old active-interval midpoints miss the better decoded region w > 17/18.
    assert _exact_decode(state, q, 0, Fraction(35, 100)) == (1, 0)
    assert _exact_decode(state, q, 0, Fraction(85, 100)) == (1, 0)
    candidates = decoded_region_candidates(state, q, 0, boundaries)
    outcomes = {_exact_decode(state, q, 0, Fraction(w)) for w in candidates}
    assert (0, 0) in outcomes
    assert cost[0, 0] + 0.5 == -2.5
    assert cost[1, 0] - 1 == -2
    assert 0.5 in candidates
    assert any(17 / 18 < weight < 1 for weight in candidates)


@pytest.mark.parametrize("domain", [2, 3])
def test_decoded_candidates_cover_exact_regions_with_other_factors_and_unaries(domain):
    rng = np.random.default_rng(512 + domain)
    for _ in range(8):
        costs = rng.integers(-5, 8, size=(2, domain, domain))
        unary = rng.integers(-3, 4, size=(3, domain)) / 2
        q = rng.integers(-3, 4, size=(4, 2, domain))
        state = _small_state(costs, unary, [[2, 0], [1, 2]], [0.25, 0.75])
        for edge in range(2):
            exact_active = _exact_partition(costs[edge], q[2 * edge : 2 * edge + 2])
            partition = [Fraction(0), *exact_active, Fraction(1)]
            required = set()
            for low, high in zip(partition, partition[1:]):
                first = low + (high - low) / 3
                second = low + 2 * (high - low) / 3
                first_beliefs = _exact_beliefs(state, q, edge, first)
                second_beliefs = _exact_beliefs(state, q, edge, second)
                cuts = {low, high}
                for variable in range(3):
                    slopes = [
                        (b - a) / (second - first)
                        for a, b in zip(
                            first_beliefs[variable], second_beliefs[variable]
                        )
                    ]
                    intercepts = [
                        a - slope * first
                        for a, slope in zip(first_beliefs[variable], slopes)
                    ]
                    for a, b in product(range(domain), repeat=2):
                        if a >= b or slopes[a] == slopes[b]:
                            continue
                        crossing = (intercepts[b] - intercepts[a]) / (
                            slopes[a] - slopes[b]
                        )
                        if low < crossing < high:
                            cuts.add(crossing)
                ordered = sorted(cuts)
                for left, right in zip(ordered, ordered[1:]):
                    required.add(_exact_decode(state, q, edge, (left + right) / 2))
            candidates = decoded_region_candidates(
                state, q, edge, np.asarray([float(value) for value in exact_active])
            )
            actual = {_exact_decode(state, q, edge, Fraction(w)) for w in candidates}
            assert required <= actual
            assert all(0 < weight < 1 for weight in candidates)
            assert float(state.weights[edge]) in candidates


def test_decoded_candidates_are_gauge_invariant_and_leave_state_untouched():
    cost = np.array([[-3, 7], [-1, 0]])
    q = np.array([[[0, 0], [-2, -1]], [[-3, -3], [3, 0]]])
    state = _small_state([cost], [[0.5, -1], [0, -0.5]], [[0, 1]], [0.5])
    before_q, before_tables = q.copy(), state.tables.copy()
    bounds = effective_boundaries(cost, q)
    expected = decoded_region_candidates(state, q, 0, bounds)
    shifted = q + np.array([[4, -8], [12, -16]])[:, :, None]
    np.testing.assert_allclose(
        decoded_region_candidates(state, shifted, 0, bounds), expected
    )
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(state.tables, before_tables)
