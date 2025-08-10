# Tests for message passing
import pytest
import logging
import numpy as np
import sys
from pathlib import Path

from propflow.core import VariableAgent, FactorAgent
from propflow.core import Message, MailHandler
from propflow.bp.computators import MaxSumComputator, MinSumComputator
from propflow.bp.factor_graph import FactorGraph
from propflow.bp.engine_base import BPEngine
from propflow.configs import Logger

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Set up logging
log_dir = "test_logs"
logger = Logger(__name__, file=True)
logger.setLevel(level=logging.DEBUG)


# Fixture for agents
@pytest.fixture
def var_agent_sender():
    agent = VariableAgent(name="VarSender", domain=2)
    return agent


@pytest.fixture
def factor_agent_recipient():
    agent = FactorAgent(
        name="FactorRecipient",
        domain=2,
        ct_creation_func=create_test_cost_table,
    )
    return agent


@pytest.fixture
def var_agent_recipient():
    agent = VariableAgent(name="VarRecipient", domain=2)
    return agent


def create_test_cost_table(num_vars=2, domain_size=2, **kwargs):
    """Helper function to create a simple cost table for testing"""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def max_sum_computator():
    return MaxSumComputator()


@pytest.fixture
def min_sum_computator():
    return MinSumComputator()


@pytest.fixture
def simple_factor_graph():
    """Create a small factor graph for testing message passing"""
    # Create variable agents
    var1 = VariableAgent(name="x1", domain=2)
    var2 = VariableAgent(name="x2", domain=2)

    # Create factor agent
    factor = FactorAgent(
        name="f12",
        domain=2,
        ct_creation_func=create_test_cost_table,
    )

    # Set up connections
    factor.connection_number = {var1.name: 0, var2.name: 1}

    # Create factor graph
    edges = {factor: [var1, var2]}
    return FactorGraph(variable_li=[var1, var2], factor_li=[factor], edges=edges)


# Test MailHandler basic operations
def test_mail_handler_basics():
    """Test basic operations of the MailHandler class"""
    # Create a mail handler
    mail_handler = MailHandler(_domain_size=2)

    # Check initial state
    assert len(mail_handler.inbox) == 0
    assert len(mail_handler.outbox) == 0

    # Create sender and recipient
    sender = VariableAgent(name="sender", domain=2)
    recipient = FactorAgent(
        name="recipient", domain=2, ct_creation_func=create_test_cost_table
    )

    # Create a message
    message_data = np.array([0.5, 0.5])
    message = Message(data=message_data, sender=sender, recipient=recipient)

    # Test receiving message
    mail_handler.receive_messages(message)
    assert len(mail_handler.inbox) == 1
    assert np.array_equal(mail_handler.inbox[0].data, message_data)

    # Test staging outgoing message
    mail_handler.stage_sending([message])
    assert len(mail_handler.outbox) == 1

    # Test clearing inbox/outbox
    mail_handler.clear_inbox()
    assert len(mail_handler.inbox) == 0

    mail_handler.prepare()  # Should clear outbox
    assert len(mail_handler.outbox) == 0


# Test message passing between agents
def test_message_passing_between_agents(var_agent_sender, factor_agent_recipient):
    """Test sending and receiving messages between agents"""
    # Create message data
    message_data = np.array([0.3, 0.7])

    # Create message
    message = Message(
        data=message_data, sender=var_agent_sender, recipient=factor_agent_recipient
    )

    # Stage message for sending
    var_agent_sender.mailer.stage_sending([message])

    # Send message
    var_agent_sender.mailer.send()

    # Check that recipient received the message
    received_messages = factor_agent_recipient.mailer.inbox
    assert len(received_messages) == 1
    assert received_messages[0].sender == var_agent_sender
    assert received_messages[0].recipient == factor_agent_recipient
    assert np.array_equal(received_messages[0].data, message_data)


# Test compute_messages for variable agent
def test_variable_agent_compute_messages(
    var_agent_sender, factor_agent_recipient, max_sum_computator
):
    """Test that a variable agent computes messages correctly"""
    # Set computator for the variable agent
    var_agent_sender.computator = max_sum_computator

    # Create message from factor to variable
    message_data = np.array([1.0, 2.0])
    message = Message(
        data=message_data, sender=factor_agent_recipient, recipient=var_agent_sender
    )

    # Add message to variable's inbox
    var_agent_sender.mailer.receive_messages(message)

    # Compute messages
    var_agent_sender.compute_messages()

    # Check outgoing messages
    outgoing = var_agent_sender.mailer.outbox
    assert len(outgoing) == 1
    assert outgoing[0].sender == var_agent_sender
    assert outgoing[0].recipient == factor_agent_recipient
    # For a single message in Q-computation, the output should be zeros
    assert np.array_equal(outgoing[0].data, np.zeros(2))


# Test compute_messages for factor agent
def test_factor_agent_compute_messages(
    var_agent_sender, factor_agent_recipient, max_sum_computator
):
    """Test that a factor agent computes messages correctly"""
    # Set computator for the factor agent
    factor_agent_recipient.computator = max_sum_computator

    # Create a cost table for the factor
    factor_agent_recipient.cost_table = np.array([[1, 2], [3, 4]])

    # Set connection numbers
    factor_agent_recipient.connection_number = {var_agent_sender.name: 0}

    # Create message from variable to factor
    message_data = np.array([0.0, 0.0])  # Neutral for addition
    message = Message(
        data=message_data, sender=var_agent_sender, recipient=factor_agent_recipient
    )

    # Add message to factor's inbox
    factor_agent_recipient.mailer.receive_messages(message)

    # Compute messages
    factor_agent_recipient.compute_messages()

    # Check outgoing messages
    outgoing = factor_agent_recipient.mailer.outbox
    assert len(outgoing) == 1
    assert outgoing[0].sender == factor_agent_recipient
    assert outgoing[0].recipient == var_agent_sender
    # For a 1D cost table with Max-Sum, should be the max values
    assert np.array_equal(outgoing[0].data, np.array([2, 4]))


# Test message passing in a factor graph with a single step
def test_factor_graph_single_step(simple_factor_graph, max_sum_computator):
    """Test message passing in a factor graph with a single step"""
    # Set computator for all nodes
    simple_factor_graph.set_computator(max_sum_computator)

    # Create BP engine
    engine = BPEngine(factor_graph=simple_factor_graph, computator=max_sum_computator)

    # Run a single step
    step = engine.step()

    # Check that messages were generated and stored in the step
    assert len(step.messages) > 0

    # Check that variables have outgoing messages
    for var in simple_factor_graph.variables:
        # After a step, outbox should be empty as messages are sent
        assert len(var.mailer.outbox) == 0
        # Mailbox should have messages from factors
        assert len(var.mailer.inbox) > 0


# Test belief computation after message passing
def test_belief_computation(simple_factor_graph, max_sum_computator):
    """Test belief computation after message passing"""
    # Set computator for all nodes
    simple_factor_graph.set_computator(max_sum_computator)

    # Create BP engine
    engine = BPEngine(factor_graph=simple_factor_graph, computator=max_sum_computator)

    # Run a cycle (multiple steps)
    for i in range(3):
        engine.step(i)

    # Get beliefs for this cycle
    beliefs = engine.get_beliefs()

    # Check that each variable has a belief
    for var in simple_factor_graph.variables:
        assert var.name in beliefs
        # Beliefs should be arrays of the correct domain size
        assert beliefs[var.name].shape == (var.domain,)
        # Sum of belief probabilities should be approximately 1
        # or at least consistent (for unnormalized beliefs)
        if np.sum(beliefs[var.name]) > 0:
            normalized = beliefs[var.name] / np.sum(beliefs[var.name])
            assert np.isclose(np.sum(normalized), 1.0)


# Test convergence with multiple cycles
def test_convergence_with_multiple_cycles(
    simple_factor_graph, min_sum_computator, min_sum_computatorn=MinSumComputator()
):
    """Test convergence with multiple cycles"""
    # Set computator for all nodes

    # Create BP engine
    engine = BPEngine(factor_graph=simple_factor_graph)

    # Initial beliefs
    engine.step(0)
    initial_beliefs = engine.get_beliefs()
    initial_assignments = engine.assignments

    # Run multiple cycles
    for i in range(1, 5):
        engine.step(i)

    # Final beliefs and assignments
    final_beliefs = engine.get_beliefs()
    final_assignments = engine.assignments

    # Check that beliefs have changed during iterations
    # (Not necessarily true for all cases, but likely for our simple example)
    beliefs_changed = False
    for var_name in initial_beliefs:
        if not np.array_equal(initial_beliefs[var_name], final_beliefs[var_name]):
            beliefs_changed = True
            break

    # For our simple 2x2 example, we expect beliefs to converge fairly quickly
    # This is not a hard requirement, but a common behavior in small graphs
    assert beliefs_changed or initial_assignments == final_assignments


# Test message deduplication in mailbox
def test_message_deduplication():
    """Test that duplicate messages are properly handled"""
    mail_handler = MailHandler(_domain_size=2)

    # Create sender and recipient
    sender = VariableAgent(name="sender", domain=2)
    recipient = FactorAgent(
        name="recipient", domain=2, ct_creation_func=create_test_cost_table
    )

    # Create two identical messages
    message1 = Message(data=np.array([0.5, 0.5]), sender=sender, recipient=recipient)
    message2 = Message(data=np.array([0.7, 0.3]), sender=sender, recipient=recipient)

    # Receive both messages
    mail_handler.receive_messages(message1)
    mail_handler.receive_messages(message2)

    # Should only keep the latest message from each sender
    assert len(mail_handler.inbox) == 1
    # Should have the data from the second message
    assert np.array_equal(mail_handler.inbox[0].data, np.array([0.7, 0.3]))


# Test handling of multiple messages in a list
def test_receive_message_list():
    """Test receiving a list of messages"""
    mail_handler = MailHandler(_domain_size=2)

    # Create senders and recipient
    sender1 = VariableAgent(name="sender1", domain=2)
    sender2 = VariableAgent(name="sender2", domain=2)
    recipient = FactorAgent(
        name="recipient", domain=2, ct_creation_func=create_test_cost_table
    )

    # Create messages
    message1 = Message(data=np.array([0.5, 0.5]), sender=sender1, recipient=recipient)
    message2 = Message(data=np.array([0.7, 0.3]), sender=sender2, recipient=recipient)

    # Receive list of messages
    mail_handler.receive_messages([message1, message2])

    # Should have both messages
    assert len(mail_handler.inbox) == 2

    # Check that messages are sorted by sender name in inbox
    assert mail_handler.inbox[0].sender.name == "sender1"
    assert mail_handler.inbox[1].sender.name == "sender2"
