"""Independent coordinate checks for the decoded-reference matrix transform."""

import numpy as np
import pytest
from scipy import sparse

from experiments.other.damping_generalization.code.decoded_gauge_check import (
    coordinate_transforms,
    dag_nilpotency_index,
)


@pytest.mark.parametrize("domain", [2, 3, 10])
def test_reference_change_matches_direct_message_subtraction(domain):
    rng = np.random.default_rng(372)
    references = np.arange(domain)
    messages = rng.normal(size=(domain, domain))
    old = (messages - messages[:, :1])[:, 1:].ravel()
    expected = np.concatenate(
        [
            np.delete(message - message[reference], reference)
            for message, reference in zip(messages, references)
        ]
    )
    transform, inverse = coordinate_transforms(references, domain)
    np.testing.assert_allclose(transform @ old, expected, atol=1e-14)
    np.testing.assert_allclose(inverse @ expected, old, atol=1e-14)
    difference = inverse @ transform - sparse.eye(len(old), dtype=int)
    difference.eliminate_zeros()
    assert difference.nnz == 0


def test_dag_nilpotency_index_distinguishes_signed_cycle_cancellation():
    dag = sparse.csr_matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert dag_nilpotency_index(dag) == 3
    assert dag_nilpotency_index(sparse.csr_matrix([[1, -1], [1, -1]])) is None


def test_invalid_reference_gauges_rejected():
    with pytest.raises(ValueError):
        coordinate_transforms(np.array([0.5]), 3)
    with pytest.raises(ValueError):
        coordinate_transforms(np.array([3]), 3)
