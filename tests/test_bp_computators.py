import numpy as np

from propflow.bp.computators import (
    MinSumComputator,
    MaxSumComputator,
    MaxProductComputator,
    SumProductComputator,
)
from propflow.core.agents import VariableAgent, FactorAgent
from propflow.core.components import Message


def _make_factor(name: str, domain: int = 2) -> FactorAgent:
    return FactorAgent(
        name,
        domain,
        ct_creation_func=lambda *_args, **_kwargs: np.zeros((domain, domain)),
        param={},
    )


def test_compute_q_with_multiple_messages():
    variable = VariableAgent("x1", domain=3)
    f1 = _make_factor("f1", domain=3)
    f2 = _make_factor("f2", domain=3)

    incoming = [
        Message(np.array([1.0, 2.0, 3.0]), sender=f1, recipient=variable),
        Message(np.array([0.5, 1.5, 2.5]), sender=f2, recipient=variable),
    ]

    computator = MinSumComputator()
    outgoing = computator.compute_Q(incoming)

    assert len(outgoing) == 2
    assert outgoing[0].sender is variable
    assert outgoing[0].recipient is f1
    np.testing.assert_allclose(outgoing[0].data, incoming[1].data)
    np.testing.assert_allclose(outgoing[1].data, incoming[0].data)


def test_compute_q_single_message_returns_zero_vector():
    variable = VariableAgent("x1", domain=2)
    factor = _make_factor("f1", domain=2)
    message = Message(np.array([3.0, 1.0]), sender=factor, recipient=variable)

    computator = MinSumComputator()
    outgoing = computator.compute_Q([message])

    assert len(outgoing) == 1
    np.testing.assert_allclose(outgoing[0].data, np.zeros_like(message.data))
    assert outgoing[0].recipient is factor


def _broadcast(msg: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    shape = [msg.shape[0] if i == axis else 1 for i in range(ndim)]
    return msg.reshape(shape)


def test_compute_r_matches_manual_min_sum_reduction():
    var_a = VariableAgent("x1", domain=2)
    var_b = VariableAgent("x2", domain=2)
    factor = _make_factor("f", domain=2)
    factor.cost_table = np.array([[0.0, 1.0], [1.5, 0.2]])
    factor.connection_number = {var_a.name: 0, var_b.name: 1}

    incoming = [
        Message(np.array([0.0, 0.3]), sender=var_a, recipient=factor),
        Message(np.array([0.1, 0.0]), sender=var_b, recipient=factor),
    ]

    computator = MinSumComputator()
    outgoing = computator.compute_R(factor.cost_table, incoming)

    assert len(outgoing) == 2
    shape = factor.cost_table.shape

    combined = factor.cost_table.copy()
    broadcasted = []
    for axis, msg in enumerate(incoming):
        br = _broadcast(msg.data, axis, factor.cost_table.ndim)
        combined += br
        broadcasted.append(br)

    manual_outputs = []
    for axis, br in enumerate(broadcasted):
        temp = combined - br
        axes = tuple(i for i in range(factor.cost_table.ndim) if i != axis)
        manual_outputs.append(temp.min(axis=axes))

    for idx, manual in enumerate(manual_outputs):
        np.testing.assert_allclose(outgoing[idx].data, manual)
        assert outgoing[idx].recipient is incoming[idx].sender


def test_compute_belief_and_assignment_variants():
    messages = [
        Message(np.array([1.0, 2.0]), sender=None, recipient=None),
        Message(np.array([0.5, 0.5]), sender=None, recipient=None),
    ]

    min_sum = MinSumComputator()
    belief = min_sum.compute_belief(messages, domain=2)
    np.testing.assert_allclose(belief, np.array([1.5, 2.5]))
    assert min_sum.get_assignment(belief) == 0

    max_sum = MaxSumComputator()
    max_belief = max_sum.compute_belief(messages, domain=2)
    np.testing.assert_allclose(max_belief, np.array([1.5, 2.5]))
    assert max_sum.get_assignment(max_belief) == 1

    max_product = MaxProductComputator()
    product_belief = max_product.compute_belief(messages, domain=2)
    np.testing.assert_allclose(product_belief, np.array([0.5, 1.0]))

    sum_product = SumProductComputator()
    prob_belief = sum_product.compute_belief(messages, domain=2)
    np.testing.assert_allclose(prob_belief, np.array([0.5, 1.0]))


def test_broadcast_shape_cache_returns_expected_tuple():
    computator = MinSumComputator()
    shape1 = computator._get_broadcast_shape(3, 1, 4)
    shape2 = computator._get_broadcast_shape(3, 1, 4)
    assert shape1 == (1, 4, 1)
    assert shape1 is shape2  # cached tuple reused
