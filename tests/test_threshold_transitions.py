"""Distinguish threshold round trips from changes between saturated outputs."""

import numpy as np
import pytest

from experiments.other.undamped_split_nodes.threshold_transitions import classify_window


def test_two_cycle_saturation_is_not_the_same_as_crossing_and_returning():
    passed = np.array(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 1, 1, 0, 1]]
    )
    plateau = np.array(
        [[5, 5, 5, 5, 5], [0, 5, 5, 5, 5], [5, 5, 5, 5, 5], [0, 5, 5, 5, 5]]
    )
    assert classify_window(passed, plateau).tolist() == [
        "above_alternating_output",
        "above_constant_output",
        "crosses_and_returns",
        "always_below",
        "other",
    ]


def test_saturation_with_nonperiodic_changes_is_not_reported_as_two_cycle():
    assert classify_window(np.ones((4, 1)), np.arange(4)[:, None]).item() == "other"


def test_transition_check_requires_multiple_observations():
    with pytest.raises(ValueError):
        classify_window(np.ones((2, 1)), np.ones((2, 1)))


def test_fixed_verdict_comparison_checks_every_outgoing_difference():
    plateau = np.zeros((4, 2, 10))
    plateau[1::2, 0, 9] = 5
    assert classify_window(np.ones((4, 2)), plateau).tolist() == [
        "above_alternating_output",
        "above_constant_output",
    ]
