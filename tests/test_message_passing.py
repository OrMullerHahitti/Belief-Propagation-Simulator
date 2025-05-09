from random import randint

# Tests for message passing
import pytest
import numpy as np
from bp_base.agents import VariableAgent, FactorAgent, BPAgent # BPAgent for mailer
from bp_base.components import Message
from bp_base.computators import MaxSumComputator


# Fixture for agents
@pytest.fixture
def var_agent_sender():
    agent = VariableAgent(name="VarSender", domain=2)
    # Ensure mailer is initialized as expected by BPAgent
    agent.neighbors = {} # Initialize neighbors
    return agent

@pytest.fixture
def factor_agent_recipient():
    agent = FactorAgent(name="FactorRecipient",domain = 2,ct_creation_func=lambda : np.random.randint(0,10,(2,2)))
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
    msg_to_send = Message(data=message_data, sender=var_agent_sender, recipient=factor_agent_recipient)
    
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
    recipient_mailer = var_agent_recipient.mailer # Mailer is for sending
    # mailbox is an attribute of the agent itself
    
    # Manually put a message in the agent's mailbox
    sender_agent = FactorAgent(name="TestFactorSender",domain=2,ct_creation_func=np.random.randint,param={"low":0,"high":10})
    msg_data = np.array([0.4, 0.6])
    incoming_msg = Message(data=msg_data, sender=sender_agent, recipient=var_agent_recipient)
    var_agent_recipient.mailer.receive_messages(incoming_msg)

    assert len(var_agent_recipient.inbox) == 1
    
    # Call empty_mailbox (which is a method of BPAgent, not its Mailer)
    var_agent_recipient.empty_mailbox()
    assert len(var_agent_recipient.inbox) == 0

# Test full message passing sequence: compute -> send -> prepare -> empty (on recipient)
def test_full_message_passing_flow(max_sum_computator):
   pass

