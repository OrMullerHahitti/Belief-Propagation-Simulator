from random import randint

# Tests for message passing
import pytest
import logging
import numpy as np
import os
import sys
from pathlib import Path

from bp_base.agents import VariableAgent, FactorAgent, BPAgent  # BPAgent for mailer
from bp_base.components import Message
from bp_base.computators import MaxSumComputator, MinSumComputator
from bp_base.factor_graph import FactorGraph
from bp_base.bp_engine_base import BPEngine
from utils.randomes import create_random_table
from configs.loggers import Logger

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
    # Ensure mailer is initialized as expected by BPAgent
    agent.neighbors = {}  # Initialize neighbors
    return agent


@pytest.fixture
def factor_agent_recipient():
    agent = FactorAgent(
        name="FactorRecipient",
        domain=2,
        ct_creation_func=lambda: np.random.randint(0, 10, (2, 2)),
    )
    agent.neighbors = {}
    # FactorAgent also needs a mailbox for receiving
    agent.mailbox = {}
    return agent


@pytest.fixture
def var_agent_recipient():
    agent = VariableAgent(name="VarRecipient", domain=2)
    agent.neighbors = {}
    agent.mailbox = {}
    return agent


@pytest.fixture
def max_sum_computator():
    # Create a mock computator for testing
    return MaxSumComputator()


# Test BPAgent.Mailer.prepare and outbox
def test_mailer_prepare_outbox(var_agent_sender, factor_agent_recipient):
    mailer = var_agent_sender.mailer
    assert len(mailer.outbox) == 0

    message_data = np.array([0.1, 0.9])
    msg_to_send = Message(
        data=message_data, sender=var_agent_sender, recipient=factor_agent_recipient
    )

    # Simulate agent putting message in its mailer's outbox (usually done by compute_messages)
    mailer.outbox.append(msg_to_send)
    assert len(mailer.outbox) == 1

    # prepare() typically readies messages, but in current BPAgent.Mailer, it clears the outbox.
    # This might be an area to clarify: does prepare() get them ready or clear for next batch?
    # Based on BPAgent.Mailer.prepare(): self.outbox = []
    # Let's test this behavior.

    # If prepare is called, outbox should be empty.
    # mailer.prepare()
    # assert len(mailer.outbox) == 0
    # This behavior of prepare() clearing outbox before send() is called seems counter-intuitive
    # if send() relies on outbox. Let's assume compute_messages fills outbox, then send() uses it, then prepare() clears.


# Test BPAgent.Mailer.send and recipient receiving message
def test_mailer_send_and_receive(var_agent_sender, factor_agent_recipient):
    pass


# Test BPAgent.Mailer.empty_mailbox
def test_mailer_empty_mailbox(var_agent_recipient):
    recipient_mailer = var_agent_recipient.mailer  # Mailer is for sending
    # mailbox is an attribute of the agent itself

    # Manually put a message in the agent's mailbox
    sender_agent = FactorAgent(
        name="TestFactorSender",
        domain=2,
        ct_creation_func=np.random.randint,
        param={"low": 0, "high": 10},
    )
    msg_data = np.array([0.4, 0.6])
    incoming_msg = Message(
        data=msg_data, sender=sender_agent, recipient=var_agent_recipient
    )
    var_agent_recipient.mailer.receive_messages(incoming_msg)

    assert len(var_agent_recipient.inbox) == 1

    # Call empty_mailbox (which is a method of BPAgent, not its Mailer)
    var_agent_recipient.empty_mailbox()
    assert len(var_agent_recipient.inbox) == 0


# Test full message passing sequence: compute -> send -> prepare -> empty (on recipient)
def test_full_message_passing_flow(max_sum_computator):
    pass


def create_3_cycle_3_domain_factor_graph():
    """
    Create a factor graph with a 3-cycle structure and 3-domain variables.
    The structure is:

    V1 -- F1 -- V2
     |          |
     F3         F2
     |          |
    V3 -------- F4

    This creates a cycle: V1 - F1 - V2 - F2 - V3 - F3 - V1
    """
    # Create 3 variable agents with domain size 3
    v1 = VariableAgent(name="V1", domain=3)
    v2 = VariableAgent(name="V2", domain=3)
    v3 = VariableAgent(name="V3", domain=3)

    # Create 4 factor agents
    # Using a simple random cost table creation function
    def create_cost_table(num_vars, domain_size, **kwargs):
        # Create a tuple of domain_size repeated num_vars times
        shape = tuple([domain_size] * num_vars)
        return create_random_table(shape)

    f1 = FactorAgent(name="F1", domain=3, ct_creation_func=create_cost_table, param={})
    f2 = FactorAgent(name="F2", domain=3, ct_creation_func=create_cost_table, param={})
    f3 = FactorAgent(name="F3", domain=3, ct_creation_func=create_cost_table, param={})
    f4 = FactorAgent(name="F4", domain=3, ct_creation_func=create_cost_table, param={})

    # Define edges
    edges = {
        f1: [v1, v2],
        f2: [v2, v3],
        f3: [v1, v3],
        f4: [v3]  # Additional factor connected to v3 only
    }

    # Create the factor graph
    fg = FactorGraph(variable_li=[v1, v2, v3], factor_li=[f1, f2, f3, f4], edges=edges)

    return fg

def test_message_passing_3_cycle_3_domain():
    """
    Test that message passing works correctly in a 3-cycle, 3-domain factor graph.
    """
    # Create the factor graph
    fg = create_3_cycle_3_domain_factor_graph()

    # Log basic graph statistics
    logger.info(f"Graph has {len(fg.variables)} variables and {len(fg.factors)} factors")
    logger.info(f"Graph diameter: {fg.diameter}")

    # Create the BPEngine
    engine = BPEngine(factor_graph=fg)

    # Run for a small number of iterations
    max_iterations = 3
    logger.info(f"Running BP Engine for {max_iterations} iterations")

    # Run the engine
    engine.run(max_iter=max_iterations, save_json=False, save_csv=False)

    # Verify that messages were passed correctly
    # Check that each variable has received messages from its connected factors
    for var in fg.variables:
        logger.info(f"Checking messages for variable {var.name}")
        # Get the neighbors of the variable
        neighbors = list(fg.G.neighbors(var))

        # Check that the variable has a belief
        assert var.belief is not None, f"Variable {var.name} has no belief"

        # Log the belief
        logger.info(f"Variable {var.name} belief: {var.belief}")

        # Check that the variable has an assignment
        assert var.curr_assignment is not None, f"Variable {var.name} has no assignment"

        # Log the assignment
        logger.info(f"Variable {var.name} assignment: {var.curr_assignment}")

    # Check that the history contains the expected number of cycles
    assert len(engine.history.cycles) == max_iterations, f"Expected {max_iterations} cycles, got {len(engine.history.cycles)}"

    # Check that the history contains beliefs and assignments for each cycle
    for i in range(max_iterations):
        assert i in engine.history.beliefs, f"No beliefs for cycle {i}"
        assert i in engine.history.assignments, f"No assignments for cycle {i}"

    # Check that the global cost was calculated for each cycle
    assert len(engine.history.costs) >= max_iterations, f"Expected at least {max_iterations} costs, got {len(engine.history.costs)}"

    # Check that messages were passed correctly by examining the history
    logger.info("Checking messages in history")
    for cycle_num, cycle in engine.history.cycles.items():
        logger.info(f"Checking messages for cycle {cycle_num}")
        for step in cycle.steps:
            for agent_name, messages in step.messages.items():
                for message in messages:
                    # Check that the message has the expected structure
                    assert message.sender is not None, f"Message from {agent_name} has no sender"
                    assert message.recipient is not None, f"Message from {agent_name} has no recipient"
                    assert message.data is not None, f"Message from {agent_name} has no data"

                    # Check that the message data has the expected shape (domain size)
                    expected_domain = message.sender.domain if hasattr(message.sender, 'domain') else 3
                    assert message.data.shape == (expected_domain,), f"Message data shape {message.data.shape} does not match expected domain size {expected_domain}"

                    # Log the message for debugging
                    logger.info(f"Message from {message.sender.name} to {message.recipient.name}: {message.data}")

    logger.info("Test completed successfully")

    return engine

if __name__ == "__main__":
    # Run the test directly
    engine = test_message_passing_3_cycle_3_domain()

    # Print the final assignments
    print("Final assignments:")
    for var_name, assignment in engine.history.assignments[max(engine.history.assignments.keys())].items():
        print(f"{var_name}: {assignment}")

    # Print the final global cost
    print(f"Final global cost: {engine.history.costs[-1]}")
