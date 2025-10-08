"""Tests for the reporting math utility helpers."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from analyzer.reporting.math_utils import (
    binary_thresholds,
    check_binary_neutral,
    jacobian_blocks_APB,
    multilabel_gaps,
    scfg_scaling,
    sibling_reinforcement_lower_bound,
)
from propflow.snapshots.types import Jacobians


def test_binary_thresholds_and_checks() -> None:
    cost = np.array([[2.0, 0.0], [3.0, 1.0]])
    theta0, theta1 = binary_thresholds(cost)
    assert theta0 == 2.0
    neutral, label = check_binary_neutral(2.5, theta0, theta1)
    assert neutral and label == 0
    neutral, label = check_binary_neutral(-3.5, theta0, theta1)
    assert neutral and label == 1
    neutral, label = check_binary_neutral(0.0, theta0, theta1)
    assert neutral and label == 1


def test_multilabel_gaps_and_selector() -> None:
    cost = np.array(
        [
            [1.0, 2.0, 4.0],
            [0.5, 0.25, 1.25],
            [3.0, 1.0, 1.5],
        ]
    )
    gaps, selector = multilabel_gaps(cost)
    assert np.allclose(gaps[0], np.array([0.0, 1.0, 3.0]))
    block = selector(1)
    assert block.shape == (3, 3)
    assert np.allclose(block.sum(axis=1), np.zeros(3))


def test_scfg_scaling_and_reinforcement() -> None:
    theta = (2.0, 4.0)
    scaled_theta = scfg_scaling(0.5, theta)
    assert scaled_theta == (1.0, 2.0)
    gaps = {0: np.array([0.0, 1.0])}
    scaled_gaps = scfg_scaling(2.0, gaps)
    assert np.allclose(scaled_gaps[0], np.array([0.0, 2.0]))
    bound = sibling_reinforcement_lower_bound(0.3, [0.5, 0.25], 0.6)
    assert abs(bound - max(0.0, 0.6 * 0.5 - 0.3)) < 1e-9


def test_jacobian_blocks_conversion() -> None:
    A = csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    P = csr_matrix(np.eye(2))
    B = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    jac = Jacobians(idxQ={}, idxR={}, A=A, P=P, B=B)
    dense_A, dense_P, dense_B = jacobian_blocks_APB(jac, as_sparse=False)
    assert isinstance(dense_A, np.ndarray)
    assert dense_B[1, 1] == -1.0
