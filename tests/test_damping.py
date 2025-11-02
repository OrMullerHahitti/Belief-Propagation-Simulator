from collections import deque

import numpy as np
import pytest

from propflow.bp.engines import DampingEngine
from propflow.core import FactorAgent, Message, VariableAgent
from propflow.policies import ConvergenceConfig, damp
from propflow.utils import FGBuilder


def _install_message_sequence(agent, neighbor, payloads, record=None):
    """Attach a deterministic compute_messages implementation to an agent."""

    sequence = deque(payloads)

    def compute():
        if not sequence:
            raise AssertionError("Message sequence exhausted")
        data = np.array(sequence.popleft(), dtype=float)
        if record is not None:
            record.append(data.copy())
        agent.mailer.stage_sending([Message(data, sender=agent, recipient=neighbor)])

    agent.mailer.clear_outgoing()
    agent.compute_messages = compute


@pytest.fixture
def deterministic_cycle_graph():
    def fixed_cost_table(num_vars, domain_size, **_):
        size = max(1, num_vars)
        values = np.arange(domain_size**size, dtype=float)
        return values.reshape((domain_size,) * size)

    return FGBuilder.build_cycle_graph(
        num_vars=3,
        domain_size=2,
        ct_factory=fixed_cost_table,
        ct_params={},
    )


def test_damp_function():
    variable = VariableAgent("test_var", 2)
    factor = FactorAgent("test_factor", 2, lambda *_args, **_kwargs: np.zeros((2, 2)))

    current = Message(np.array([1.0, 2.0]), variable, factor)
    variable.mailer.stage_sending([current])

    previous = Message(np.array([4.0, 6.0]), variable, factor)
    variable._history = [[previous.copy()]]
    expected = 0.5 * previous.data + 0.5 * current.data

    damp(variable, 0.5)

    np.testing.assert_allclose(variable.mailer.outbox[0].data, expected)


def test_damping_engine_with_direct_initialization(deterministic_cycle_graph):
    engine = DampingEngine(
        deterministic_cycle_graph,
        damping_factor=0.5,
        convergence_config=ConvergenceConfig(),
        normalize_messages=True,
    )

    variable = deterministic_cycle_graph.variables[0]
    factor = list(deterministic_cycle_graph.G.neighbors(variable))[0]

    previous = Message(np.array([2.0, 6.0]), variable, factor)
    current = Message(np.array([4.0, 8.0]), variable, factor)
    variable._history = [[previous.copy()]]
    variable.mailer.stage_sending([current])
    expected = 0.5 * previous.data + 0.5 * current.data

    damp(variable, engine.damping_factor)

    np.testing.assert_allclose(variable.mailer.outbox[0].data, expected)


def test_damping_engine_with_multiple_iterations(deterministic_cycle_graph):
    engine = DampingEngine(
        deterministic_cycle_graph,
        damping_factor=0.5,
        convergence_config=ConvergenceConfig(),
        normalize_messages=True,
    )

    variable = deterministic_cycle_graph.variables[0]
    factor = list(deterministic_cycle_graph.G.neighbors(variable))[0]

    raw_messages: list[np.ndarray] = []
    _install_message_sequence(
        variable,
        factor,
        [
            np.array([2.0, 4.0], dtype=float),
            np.array([6.0, 8.0], dtype=float),
        ],
        record=raw_messages,
    )
    _install_message_sequence(
        factor,
        variable,
        [np.zeros(2), np.zeros(2)],
    )

    first_step = engine.step(0)
    np.testing.assert_allclose(
        first_step.q_messages[variable.name][0].data, raw_messages[0]
    )
    np.testing.assert_allclose(variable._history[-1][0].data, raw_messages[0])

    second_step = engine.step(1)
    expected = 0.5 * raw_messages[0] + 0.5 * raw_messages[1]
    np.testing.assert_allclose(second_step.q_messages[variable.name][0].data, expected)
    np.testing.assert_allclose(variable._history[-1][0].data, expected)
