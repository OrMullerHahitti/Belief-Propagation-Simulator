import numpy as np
import pytest

torch = pytest.importorskip("torch")

from propflow.core.agents import VariableAgent, FactorAgent
from propflow.core.components import Message
from propflow.nn.torch_computators import SoftMinTorchComputator
from propflow.bp.computators import MinSumComputator


def _make_factor(name: str, domain: int = 3) -> FactorAgent:
    return FactorAgent(
        name,
        domain,
        ct_creation_func=lambda *_args, **_kwargs: np.zeros((domain, domain)),
        param={},
    )


def test_softmin_r_approximates_min_sum_pairwise():
    D = 3
    var_a = VariableAgent("x1", domain=D)
    var_b = VariableAgent("x2", domain=D)
    factor = _make_factor("f", domain=D)
    factor.cost_table = np.array(
        [[0.0, 1.0, 2.0], [1.5, 0.1, 0.9], [1.1, 0.7, 0.3]],
        dtype=float,
    )
    factor.connection_number = {var_a.name: 0, var_b.name: 1}

    incoming = [
        Message(np.array([0.0, 0.3, 0.2]), sender=var_a, recipient=factor),
        Message(np.array([0.1, 0.0, 0.4]), sender=var_b, recipient=factor),
    ]

    # Min-Sum reference
    ref = MinSumComputator().compute_R(factor.cost_table, incoming)

    # Soft-min with small tau should be close to hard min
    soft = SoftMinTorchComputator(tau=1e-3).compute_R(factor.cost_table, incoming)

    for r_ref, r_soft in zip(ref, soft):
        np.testing.assert_allclose(r_soft.data, r_ref.data, rtol=1e-3, atol=1e-3)


def test_compute_q_matches_additive_exclusion():
    D = 4
    var = VariableAgent("x1", domain=D)
    f1 = _make_factor("f1", domain=D)
    f2 = _make_factor("f2", domain=D)
    incoming = [
        Message(np.array([1.0, 2.0, 3.0, 4.0]), sender=f1, recipient=var),
        Message(np.array([0.5, 1.5, 2.5, 3.5]), sender=f2, recipient=var),
    ]
    out = SoftMinTorchComputator(tau=0.2).compute_Q(incoming)
    total = incoming[0].data + incoming[1].data
    np.testing.assert_allclose(out[0].data, total - incoming[0].data)
    np.testing.assert_allclose(out[1].data, total - incoming[1].data)
