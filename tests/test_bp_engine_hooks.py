import types
from collections import deque

import numpy as np
import pytest

from propflow.bp.engine_base import BPEngine
from propflow.bp.engines import DampingEngine, SplitEngine
from propflow.bp.factor_graph import FactorGraph
from propflow.core.agents import FactorAgent, VariableAgent
from propflow.core.components import MailHandler, Message


def _make_cost_table(num_vars: int, domain_size: int, **_: int) -> np.ndarray:
    """Utility cost-table generator with deterministic values."""
    shape = (domain_size,) * max(1, num_vars)
    data = np.arange(np.prod(shape), dtype=float)
    return data.reshape(shape)


def build_single_factor_graph(domain: int = 3) -> tuple[FactorGraph, VariableAgent, FactorAgent]:
    var = VariableAgent("x1", domain=domain)
    factor = FactorAgent("f1", domain=domain, ct_creation_func=_make_cost_table)
    edges = {factor: [var]}
    graph = FactorGraph(variable_li=[var], factor_li=[factor], edges=edges)
    return graph, var, factor


def _bind_sequence_stub(agent, neighbour, sequence: deque[np.ndarray], record=None):
    def _compute(self):
        if not sequence:
            raise AssertionError("Message sequence exhausted for test stub")
        data = np.array(sequence.popleft(), dtype=float)
        if record is not None:
            record.append(data.copy())
        self.mailer.stage_sending([Message(data, sender=self, recipient=neighbour)])

    return types.MethodType(_compute, agent)


def test_bp_engine_initializes_zero_messages():
    graph, var, factor = build_single_factor_graph(domain=3)

    BPEngine(graph)

    inbox = var.mailer.inbox
    assert len(inbox) == 1
    message = inbox[0]
    assert message.sender is factor
    assert message.recipient is var
    np.testing.assert_allclose(message.data, np.zeros(var.domain))
    assert factor.mailer.inbox == []


def test_bp_engine_step_records_messages():
    graph, var, factor = build_single_factor_graph(domain=3)
    engine = BPEngine(graph)

    var_sequence = deque([
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    ])
    factor_sequence = deque([
        np.array([0.1, 0.2, 0.3]),
        np.array([0.3, 0.2, 0.1]),
    ])

    var.compute_messages = _bind_sequence_stub(var, factor, var_sequence)
    factor.compute_messages = _bind_sequence_stub(factor, var, factor_sequence)

    first_step = engine.step(0)
    np.testing.assert_allclose(first_step.q_messages[var.name][0].data, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(first_step.r_messages[factor.name][0].data, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(first_step.messages[var.name][0].data, [0.1, 0.2, 0.3])

    second_step = engine.step(1)
    np.testing.assert_allclose(second_step.q_messages[var.name][0].data, [4.0, 5.0, 6.0])
    assert second_step.num == 1
    assert not var_sequence
    assert not factor_sequence


def test_damping_engine_reuses_previous_messages():
    graph, var, factor = build_single_factor_graph(domain=2)
    engine = DampingEngine(graph, damping_factor=0.5)

    raw_sent: list[np.ndarray] = []
    var_sequence = deque([
        np.array([2.0, 4.0]),
        np.array([6.0, 8.0]),
    ])
    factor_sequence = deque([
        np.zeros(2),
        np.zeros(2),
    ])

    var.compute_messages = _bind_sequence_stub(var, factor, var_sequence, record=raw_sent)
    factor.compute_messages = _bind_sequence_stub(factor, var, factor_sequence)

    first_step = engine.step(0)
    np.testing.assert_allclose(first_step.q_messages[var.name][0].data, raw_sent[0])
    assert len(var._history) == 1

    second_step = engine.step(1)
    expected = 0.5 * raw_sent[0] + 0.5 * raw_sent[1]
    np.testing.assert_allclose(second_step.q_messages[var.name][0].data, expected)
    assert len(var._history) == 2
    np.testing.assert_allclose(var._history[-1][0].data, expected)


def test_split_engine_calls_policy(monkeypatch):
    graph, *_ = build_single_factor_graph()
    captured = {}

    def fake_split(target_graph, fraction):
        captured["graph"] = target_graph
        captured["fraction"] = fraction

    monkeypatch.setattr("propflow.bp.engines.split_all_factors", fake_split)

    engine = SplitEngine(graph, split_factor=0.42)

    assert captured["graph"] is graph
    assert captured["fraction"] == pytest.approx(0.42)
    assert engine.split_factor == pytest.approx(0.42)


def test_mailhandler_keys_include_agent_type():
    owner = VariableAgent("x_owner", domain=2)
    handler: MailHandler = owner.mailer

    same_name_factor = FactorAgent(
        "shared",
        domain=2,
        ct_creation_func=_make_cost_table,
    )
    same_name_variable = VariableAgent("shared", domain=2)

    handler.receive_messages(Message(np.zeros(2), sender=same_name_factor, recipient=owner))
    handler.receive_messages(Message(np.ones(2), sender=same_name_variable, recipient=owner))

    inbox = handler.inbox
    assert len(inbox) == 2
    assert {msg.sender.type for msg in inbox} == {"factor", "variable"}
