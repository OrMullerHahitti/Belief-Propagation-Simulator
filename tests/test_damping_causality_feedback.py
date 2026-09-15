"""Independent arithmetic checks for the controlled sibling-feedback map."""

import numpy as np
import pytest

from experiments.other.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    PairwiseProblem,
)
from experiments.other.damping_causality.code.feedback import (
    FeedbackState,
    active_jacobian,
    advance,
    factor_response,
)


def _problem() -> PairwiseProblem:
    return PairwiseProblem(
        np.array([[0, 1], [1, 2]]),
        np.array([[[16, 0], [0, 16]], [[16, 0], [0, 16]]]),
        np.array([[12, 0], [13, 0], [4, 0]]),
        "exact_path",
        0,
        "path",
    )


@pytest.mark.parametrize("damping", [0.0, 0.01, 0.5, 0.9])
def test_full_scale_equal_clone_reduction(damping: float) -> None:
    problem = _problem()
    full = FeedbackState.zeros(problem)
    split = PairwiseKernel(problem, damping=damping)
    for _ in range(200):
        full = advance(problem, full, damping=damping)
        split.step()
        for reduced, actual in ((full.q, split.q), (full.r, split.r)):
            difference = actual[:, :, 1] - actual[:, :, 0]
            np.testing.assert_allclose(difference[::2], difference[1::2], atol=1e-11)
            np.testing.assert_allclose(reduced, 2 * difference[::2], atol=1e-10)


@pytest.mark.parametrize("gains", [(1, 0), (2, 0), (1, 1), (2, 1), (2, -1)])
def test_active_derivative_matches_perturbed_map(gains: tuple[int, int]) -> None:
    problem = _problem()
    q = np.array([[1.25, -2.75], [3.5, 4.25]])
    damping = 0.3
    external, sibling = gains
    jacobian, margin = active_jacobian(problem, q, damping, external, sibling)
    assert margin > 1

    def transition(point: np.ndarray) -> np.ndarray:
        state = FeedbackState(point, factor_response(problem, point), 5)
        return advance(problem, state, damping, external, sibling).q.ravel()

    epsilon = 1e-5
    numerical = np.column_stack(
        [
            (transition(q + direction) - transition(q - direction)) / (2 * epsilon)
            for direction in epsilon * np.eye(q.size).reshape((-1, *q.shape))
        ]
    )
    np.testing.assert_allclose(jacobian, numerical, atol=1e-8, rtol=0)


def test_selector_tie_is_reported_before_smooth_claim() -> None:
    _, margin = active_jacobian(_problem(), np.array([[16.0, 0], [0, 0]]))
    assert margin == 0


def test_label_swapped_return_preserves_binary_gap_magnitude() -> None:
    problem = _problem()
    state = FeedbackState.zeros(problem)
    for _ in range(6):
        state = advance(problem, state)
    ordinary = advance(problem, state, external_gain=0, sibling_gain=1)
    swapped = advance(problem, state, external_gain=0, sibling_gain=-1)
    np.testing.assert_array_equal(ordinary.q, -swapped.q)
    np.testing.assert_array_equal(abs(ordinary.q), abs(swapped.q))
