"""Exact structural witnesses beyond a single damped trajectory certificate."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from experiments.aamas.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    PairwiseProblem,
)
from experiments.aamas.damping_causality.theory_helpers import (
    path_belief_gaps,
    path_raw_q,
    path_step,
)


@pytest.mark.parametrize(
    "point,beliefs",
    [
        ((-20, 11, 11, -12), (-28, 19, -20)),
        ((-4, -17, -9, 4), (4, -13, 12)),
        (
            (Fraction(-9, 2), Fraction(-15, 2), Fraction(-7, 2), Fraction(-1, 2)),
            (3, -3, 3),
        ),
    ],
)
def test_three_strict_path_fixed_points(point: tuple, beliefs: tuple) -> None:
    assert path_raw_q(point) == point
    assert path_belief_gaps(point) == beliefs
    assert all(abs(value) != 8 for value in point)
    for damping in (Fraction(0), Fraction(1, 2), Fraction(9, 10)):
        assert path_step(point, damping) == point


def test_cycle_is_exact_bipartite_hybrid_of_stable_solutions() -> None:
    good = (-20, 11, 11, -12)
    worse = (-4, -17, -9, 4)
    first = (worse[0], good[1], good[2], worse[3])
    second = (good[0], worse[1], worse[2], good[3])
    assert path_raw_q(first) == second
    assert path_raw_q(second) == first
    assert first != second
    expected = tuple(Fraction(a + b, 2) for a, b in zip(first, second))
    assert path_step(first, Fraction(1, 2)) == expected
    assert expected != second


def _exact_path_jacobian(point: tuple) -> np.ndarray:
    epsilon = Fraction(1, 1024)
    baseline = path_raw_q(point)
    columns = []
    for index in range(4):
        perturbed = list(map(Fraction, point))
        perturbed[index] += epsilon
        column = [
            (new - old) / epsilon for new, old in zip(path_raw_q(perturbed), baseline)
        ]
        assert all(value.denominator == 1 for value in column)
        columns.append(column)
    return np.asarray(columns, dtype=int).T


def test_nilpotent_stable_cells_and_positive_unstable_mode() -> None:
    good = _exact_path_jacobian((-20, 11, 11, -12))
    worse = _exact_path_jacobian((-4, -17, -9, 4))
    interior = _exact_path_jacobian(
        (Fraction(-9, 2), Fraction(-15, 2), Fraction(-7, 2), Fraction(-1, 2))
    )
    np.testing.assert_array_equal(good, np.zeros((4, 4), dtype=int))
    np.testing.assert_array_equal(worse @ worse, good)
    squared = interior @ interior
    np.testing.assert_array_equal(squared @ squared - 2 * squared, 3 * np.eye(4))
    signs = np.array([1, -1, -1, 1])
    gauged = signs[:, None] * interior * signs[None, :]
    assert np.all(gauged >= 0)
    assert np.max(np.linalg.eigvals(gauged).real) == pytest.approx(np.sqrt(3))
    assert 0.9 + 0.1 * np.sqrt(3) > 1


def test_suboptimal_attractor_has_invariant_open_box() -> None:
    center = (-4, -17, -9, 4)
    radii = (Fraction(1, 4), Fraction(1), Fraction(1), Fraction(1, 4))
    for directions in product((-1, 1), repeat=4):
        point = tuple(
            value + Fraction(999, 1000) * direction * radius
            for value, direction, radius in zip(center, directions, radii)
        )
        for damping in (Fraction(0), Fraction(1, 2), Fraction(9, 10)):
            result = path_step(point, damping)
            assert all(
                abs(value - target) < radius
                for value, target, radius in zip(result, center, radii)
            )
        assert path_raw_q(path_raw_q(point)) == center


def test_global_order_box_reaches_the_two_extremal_fixed_points() -> None:
    signs = (1, -1, -1, 1)

    def gauged(point: tuple) -> tuple:
        raw = path_raw_q(tuple(sign * value for sign, value in zip(signs, point)))
        return tuple(sign * value for sign, value in zip(signs, raw))

    lower = (-20, -11, -11, -12)
    upper = (-4, 37, 37, 4)
    upper_fixed = (-4, 17, 9, 4)
    assert gauged(lower) == lower
    assert gauged(upper) == upper_fixed
    assert gauged(upper_fixed) == upper_fixed
    for choices in product((0, 1), repeat=4):
        point = tuple((lo, hi)[choice] for lo, hi, choice in zip(lower, upper, choices))
        assert all(
            lo <= value <= hi
            for lo, value, hi in zip(lower, gauged(point), upper_fixed)
        )


def _single_edge_kernel(point: np.ndarray, damping: float) -> PairwiseKernel:
    problem = PairwiseProblem(
        np.array([[0, 1]]),
        np.array([[[16, 0], [0, 16]]]),
        np.zeros((2, 2)),
        "neutral_manifold",
        0,
        "single_edge",
    )
    kernel = PairwiseKernel(problem, damping=damping)
    kernel.q[:, :, 1] = point.reshape(2, 2)
    tables = kernel.tables
    kernel.r[:, 0] = (tables + kernel.q[:, 1, None, :]).min(axis=2)
    kernel.r[:, 1] = (tables + kernel.q[:, 0, :, None]).min(axis=1)
    kernel.t = 1
    return kernel


def test_damping_projects_onto_neutral_manifold_with_strict_decoding() -> None:
    initial = np.array([1.25, 0.75, 1.0, 0.5])
    reverse = -initial[[3, 2, 1, 0]]
    projected = (initial + reverse) / 2
    assert np.max(np.abs(initial)) < 8
    assert not np.array_equal(initial, reverse)
    np.testing.assert_array_equal(-projected[[3, 2, 1, 0]], projected)

    undamped = _single_edge_kernel(initial, 0.0)
    damped = _single_edge_kernel(initial, 0.5)
    for step in range(20):
        undamped.step()
        damped.step()
        actual_undamped = (undamped.q[:, :, 1] - undamped.q[:, :, 0]).ravel()
        actual_damped = (damped.q[:, :, 1] - damped.q[:, :, 0]).ravel()
        expected_undamped = reverse if step % 2 == 0 else initial
        np.testing.assert_array_equal(actual_undamped, expected_undamped)
        np.testing.assert_array_equal(actual_damped, projected)
        beliefs = damped.beliefs()[:, 1] - damped.beliefs()[:, 0]
        np.testing.assert_array_equal(beliefs, [0.5, -0.5])
        assert damped.cost == 0
        assert undamped.cost == 16


@pytest.mark.parametrize("damping", [0.2, 0.5, 0.9])
def test_manifold_mode_formula_predicts_strict_decoding_time(damping: float) -> None:
    initial = np.array([1.25, 0.75, 1.0, 0.5])
    projected = (initial - initial[[3, 2, 1, 0]]) / 2
    transverse = initial - projected
    kernel = _single_edge_kernel(initial, damping)
    total = initial.sum()
    difference = initial[0] + initial[2] - initial[1] - initial[3]
    for step in range(1, 21):
        kernel.step()
        mode = (2 * damping - 1) ** step
        expected = projected + mode * transverse
        actual = (kernel.q[:, :, 1] - kernel.q[:, :, 0]).ravel()
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
        assert np.max(np.abs(actual)) < 8
        if abs(mode) * total < abs(difference):
            assert kernel.cost == 0
        else:
            assert kernel.cost == 16


def test_belief_magnitude_lemma_for_all_small_strict_tables() -> None:
    checked = 0
    for entries in product(range(6), repeat=4):
        table = np.array(entries).reshape(2, 2)
        if np.any(table[:, 0] == table[:, 1]):
            continue
        if np.any(table[0] == table[1]):
            continue
        row_choices = table.argmin(axis=1)
        column_choices = table.argmin(axis=0)
        if row_choices[0] == row_choices[1]:
            continue
        receiver_gap = np.diff(table.min(axis=1)).item()
        sender_gap = np.diff(table.min(axis=0)).item()
        reverse_active = column_choices[0] != column_choices[1]
        assert abs(sender_gap) <= abs(receiver_gap)
        assert (abs(sender_gap) == abs(receiver_gap)) == reverse_active
        if reverse_active and receiver_gap:
            interaction = table[0, 1] + table[1, 0] - table[0, 0] - table[1, 1]
            assert np.sign(interaction) == np.sign(receiver_gap) * np.sign(sender_gap)
        checked += 1
    assert checked > 100


def test_ternary_decoded_label_gauge_removes_signed_dependencies() -> None:
    reparameterized = np.array([[8, 4, 6], [3, 2, 0], [7, 5, 1]], dtype=float)
    outgoing = np.stack((reparameterized.min(axis=1), reparameterized.min(axis=0))) / 2
    table = reparameterized - outgoing[0, :, None] - outgoing[1, None, :]
    point = np.tile(outgoing - outgoing[:, :1], (2, 1, 1))
    preferred = outgoing.argmin(axis=1)
    np.testing.assert_array_equal(preferred, [1, 2])

    def raw(current: np.ndarray) -> np.ndarray:
        response = np.empty_like(current)
        response[:, 0] = (table + current[:, 1, None, :]).min(axis=2)
        response[:, 1] = (table + current[:, 0, :, None]).min(axis=1)
        next_q = response[::-1]
        return next_q - next_q[:, :, :1]

    np.testing.assert_array_equal(raw(point), point)
    epsilon = 2**-12
    columns = []
    for direction in np.eye(8).reshape(8, 2, 2, 2):
        shifted = point.copy()
        shifted[:, :, 1:] += epsilon * direction
        columns.append(((raw(shifted) - point)[:, :, 1:] / epsilon).ravel())
    jacobian = np.stack(columns, axis=1)
    assert np.any(jacobian < 0)
    np.testing.assert_array_equal(jacobian @ jacobian, np.zeros((8, 8)))

    transform = np.zeros((8, 8))
    for clone in range(2):
        for endpoint in range(2):
            best = preferred[endpoint]
            offset = 4 * clone + 2 * endpoint
            labels = [label for label in range(3) if label != best]
            for row, label in enumerate(labels):
                if label != 0:
                    transform[offset + row, offset + label - 1] += 1
                transform[offset + row, offset + best - 1] -= 1
    inverse = np.linalg.inv(transform)
    np.testing.assert_array_equal(inverse, np.round(inverse))
    adapted = transform @ jacobian @ inverse
    assert np.all((adapted == 0) | (adapted == 1))
    assert np.any(adapted > 0)
    np.testing.assert_array_equal(adapted @ adapted, np.zeros((8, 8)))
