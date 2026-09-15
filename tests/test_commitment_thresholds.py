"""Commitment is a common factor minimizer, not just a stable decoded value."""

import numpy as np
import pytest

from propflow.snapshots.commitment import evaluate_commitment, row_thresholds


def test_binary_upper_boundary_and_strict_threshold():
    costs = np.array([[0.0, 5.0], [5.0, 0.0]])
    omega = row_thresholds(costs)
    state = evaluate_commitment(np.array([[0, 4], [0, 5], [0, 6]]), omega)
    assert state.strict.tolist() == [False, False, True]
    assert state.weak.tolist() == [False, True, True]
    np.testing.assert_allclose(state.slack, [-1, 0, 1])


def test_committed_row_need_not_be_incoming_q_argmin():
    table = np.array([[10.0, 10], [0, 0]])
    state = evaluate_commitment(np.array([0.0, 1]), row_thresholds(table))
    assert state.strict and state.row == 1 and state.slack == 9


def test_same_incoming_preference_is_not_outgoing_difference_independence():
    table = np.array([[0.0, 5.0], [5.0, 0.0]])
    messages = np.array([[0.0, 3], [0, 4], [0, 6], [0, 8]])
    assert np.all(messages.argmin(-1) == 0)
    outgoing = (table[None] + messages[..., None]).min(axis=-2)
    np.testing.assert_allclose(outgoing[:, 1] - outgoing[:, 0], [3, 4, 5, 5])
    state = evaluate_commitment(messages, row_thresholds(table))
    assert state.strict.tolist() == [False, False, True, True]


def test_all_domain_values_matter_not_only_the_two_decoded_values():
    table = np.array([[0.0, 0, 10], [5, 5, 0], [20, 20, 20]])
    state = evaluate_commitment(np.zeros(3), row_thresholds(table))
    assert not state.weak


def test_cost_axes_transpose_and_split_scale_are_explicit():
    table = np.array([[0.0, 4, 9], [7, 2, 3]])
    q = np.array([0.0, 4])
    assert not evaluate_commitment(q, row_thresholds(table)).strict
    assert evaluate_commitment(q, row_thresholds(table * 0.5)).strict
    with pytest.raises(ValueError):
        evaluate_commitment(q, row_thresholds(table.T))


def test_reference_offsets_and_batching_preserve_classification():
    rng = np.random.default_rng(410)
    costs = rng.normal(size=(4, 3, 3))
    q = rng.normal(size=(8, 4, 3))
    omega = row_thresholds(costs)
    a = evaluate_commitment(q, omega)
    b = evaluate_commitment(q + 100, omega)
    np.testing.assert_allclose(a.slack, b.slack, atol=1e-12)
    scores = costs[None] + q[..., None]
    winners = scores.argmin(axis=-2)
    brute = (winners == winners[..., :1]).all(axis=-1)
    np.testing.assert_array_equal(a.strict, brute)


def test_reject_nonfinite_and_invalid_tolerances():
    with pytest.raises(ValueError):
        row_thresholds(np.full((2, 2), np.nan))
    with pytest.raises(ValueError):
        evaluate_commitment(np.zeros(2), np.zeros((2, 2)), -1)
