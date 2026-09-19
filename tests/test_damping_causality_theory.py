"""Exact witness checks and an independent rational/native-kernel comparison."""

from fractions import Fraction as F

import numpy as np
import pytest

from experiments.aamas.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    PairwiseProblem,
    gauge,
)
from experiments.aamas.damping_causality.theory_helpers import (
    committed_cycle_threshold,
    path_belief_gaps,
    path_cycle_candidate,
    path_raw_q,
    path_second_cycle_candidate,
    path_step,
    regular_scalar_step,
)


def test_undamped_path_enters_exact_strict_two_cycle():
    q = (F(0),) * 4
    expected = ((-4, 11, 11, 4), (-20, -17, -9, -12))
    for step in range(1, 9):
        q = path_step(q, F(0))
        if step >= 4:
            assert q == expected[(step - 4) % 2]
    assert path_belief_gaps(expected[0]) == (-28, -13, -20)
    assert path_belief_gaps(expected[1]) == (4, 19, 12)
    assert all(abs(value) != 8 for state in expected for value in state)


def test_half_damping_enters_proved_invariant_fixed_region():
    q = (F(0),) * 4
    for _ in range(12):
        q = path_step(q, F(1, 2))
    assert q == (F(-8709, 512), F(4777, 512), F(643, 64), F(-691, 64))
    target = (-20, 11, 11, -12)
    assert q[0] < -8 and q[1] > 8 and q[2] > 8 and q[3] < -8
    assert path_raw_q(q) == target
    assert path_raw_q(target) == target
    assert path_belief_gaps(target) == (-28, 19, -20)
    for damping in (F(0), F(1, 2), F(9, 10), F(99, 100)):
        after = path_step(q, damping)
        assert path_raw_q(after) == target
        assert tuple(a - b for a, b in zip(after, target)) == tuple(
            damping * (a - b) for a, b in zip(q, target)
        )


@pytest.mark.parametrize(
    ("damping", "entry_update"), [(F(1, 50), 50), (F(9, 10), 53), (F(99, 100), 516)]
)
def test_other_dampings_have_exact_finite_entry_certificates(damping, entry_update):
    q = (F(0),) * 4
    for step in range(1, entry_update + 1):
        q = path_step(q, damping)
        in_region = q[0] < -8 and q[1] > 8 and q[2] > 8 and q[3] < -8
        assert in_region == (step == entry_update)
    assert path_raw_q(q) == (-20, 11, 11, -12)
    assert path_belief_gaps(q) == (-28, 19, -20)


def test_small_damping_preserves_the_same_strict_path_cycle():
    damping = F(1, 100)
    first, second = path_cycle_candidate(damping)
    assert -8 < first[0] < 8 and -8 < first[3] < 8
    assert first[1] > 8 and first[2] > 8
    assert all(value < -8 for value in second)
    assert path_step(first, damping) == second
    assert path_step(second, damping) == first
    assert 19 * damping**2 + 66 * damping - 1 < 0
    beyond = F(1, 50)
    first, second = path_cycle_candidate(beyond)
    assert second[2] > -8
    assert path_step(second, beyond) != first


def test_path_has_a_different_cycle_above_the_first_pattern_threshold():
    damping = F(2, 125)
    first, second = path_second_cycle_candidate(damping)
    assert first == (F(-540, 127), F(7200583, 677418), F(901, 84), F(40343, 21336))
    assert second == (
        F(-2508, 127),
        F(-16473679, 1354836),
        F(-1027, 168),
        F(-125645, 10668),
    )
    assert -8 < first[0] < 8 and -8 < first[3] < 8
    assert first[1] > 8 and first[2] > 8
    assert all(second[index] < -8 for index in (0, 1, 3))
    assert -8 < second[2] < 8
    assert path_step(first, damping) == second
    assert path_step(second, damping) == first
    assert all(value < 0 for value in path_belief_gaps(first))
    assert all(value > 0 for value in path_belief_gaps(second))
    assert 19 * damping**2 + 66 * damping - 1 > 0
    # the nontrivial two-step eigenvalues satisfy z^2 - trace*z + determinant.
    trace = 3 * damping**2 - 2 * damping + 1
    determinant = damping**4
    assert 1 - trace + determinant == damping * (damping - 1) ** 2 * (damping + 2)
    assert 1 - trace + determinant > 0
    assert 1 + trace + determinant > 0
    assert determinant < 1


@pytest.mark.parametrize("damping", [F(0), F(1, 2), F(9, 10)])
def test_rational_path_recurrence_matches_native_arithmetic_kernel(damping):
    problem = PairwiseProblem(
        np.array([[0, 1], [1, 2]]),
        np.array([[[16, 0], [0, 16]], [[16, 0], [0, 16]]]),
        np.array([[12, 0], [13, 0], [4, 0]]),
        "exact_path",
        0,
        "path3",
    )
    kernel = PairwiseKernel(problem, damping=float(damping))
    kernel.step()
    q = (F(0),) * 4
    for _ in range(24):
        np.testing.assert_allclose(
            gauge(kernel.q)[::2, :, 1].ravel(),
            list(map(float, q)),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            gauge(kernel.beliefs())[:, 1],
            list(map(float, path_belief_gaps(q))),
            atol=1e-12,
            rtol=0,
        )
        q = path_step(q, damping)
        kernel.step()


def test_pattern_threshold_does_not_imply_convergence():
    assert committed_cycle_threshold(2, F(2), F(1)) == F(1, 3)
    # above the fully committed threshold, this partly clipped 2-cycle persists.
    low, high = F(-23, 13), F(40, 13)
    assert regular_scalar_step(low, F(2, 5), 2, F(2), F(1)) == high
    assert regular_scalar_step(high, F(2, 5), 2, F(2), F(1)) == low
    assert -2 < low < 2 < high


def test_converging_messages_can_keep_flipping_decoded_assignments():
    q, fixed = F(0), F(1, 4)
    for step in range(1, 21):
        q = regular_scalar_step(q, F(3, 5), 2, F(2), F(1))
        assert q - fixed == (-F(3, 5)) ** step * (-fixed)
        belief_gap = 1 - 4 * q
        assert belief_gap != 0
        assert (belief_gap < 0) == (step % 2 == 1)


def test_point_nine_fails_in_the_degree_eleven_counterexample():
    threshold = committed_cycle_threshold(11, F(500), F(1))
    assert threshold == F(9999, 11001)
    assert F(9, 10) < threshold < F(19, 20)
    high, low = F(1) + F(10500, 19), F(1) - F(10500, 19)
    assert low < -500 < 500 < high
    assert regular_scalar_step(low, F(9, 10), 11, F(500), F(1)) == high
    assert regular_scalar_step(high, F(9, 10), 11, F(500), F(1)) == low
