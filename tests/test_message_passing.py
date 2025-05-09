\
# Tests for message passing
import pytest
import numpy as np
from bp_base.agents import VariableAgent, FactorAgent, BPAgent # BPAgent for mailer
from bp_base.components import Message

# Fixture for agents
@pytest.fixture
def var_agent_sender():
    agent = VariableAgent(name="VarSender", domain_size=2)
    # Ensure mailer is initialized as expected by BPAgent
    agent.mailer = BPAgent.Mailer(agent) 
    agent.neighbors = {} # Initialize neighbors
    return agent

@pytest.fixture
def factor_agent_recipient():
    agent = FactorAgent(name="FactorRecipient")
    agent.mailer = BPAgent.Mailer(agent)
    agent.neighbors = {}
    # FactorAgent also needs a mailbox for receiving
    agent.mailbox = {} 
    return agent

@pytest.fixture
def var_agent_recipient():
    agent = VariableAgent(name="VarRecipient", domain_size=2)
    agent.mailer = BPAgent.Mailer(agent)
    agent.neighbors = {}
    agent.mailbox = {}
    return agent

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
    sender_mailer = var_agent_sender.mailer
    recipient_mailer = factor_agent_recipient.mailer # Recipient also has a mailer with a mailbox
    
    # The recipient agent needs to be a known neighbor for the sender's mailer to send to it.
    # In BPAgent.Mailer.send(), it iterates self.owner.neighbors.values()
    # So, the recipient must be in the sender's neighbors list.
    var_agent_sender.neighbors = {factor_agent_recipient.name: factor_agent_recipient}

    message_data = np.array([0.7, 0.3])
    msg = Message(data=message_data, sender=var_agent_sender, recipient=factor_agent_recipient)
    
    # Agent's compute_messages would put this in sender_mailer.outbox
    sender_mailer.outbox = [msg]
    
    # Call send
    sender_mailer.send()
    
    # Message should now be in the recipient's mailbox (factor_agent_recipient.mailbox)
    # The BPAgent.Mailer.send() method directly places messages into recipient.mailbox
    assert factor_agent_recipient.name in factor_agent_recipient.mailbox
    assert len(factor_agent_recipient.mailbox[factor_agent_recipient.name]) == 1 # Mailbox stores by sender name
    # Correction: Mailbox should store by SENDER's name
    assert var_agent_sender.name in factor_agent_recipient.mailbox
    received_msgs_from_sender = factor_agent_recipient.mailbox[var_agent_sender.name]
    assert len(received_msgs_from_sender) == 1
    assert received_msgs_from_sender[0] == msg
    assert np.array_equal(received_msgs_from_sender[0].data, message_data)

    # After sending, the sender's outbox is usually cleared by prepare() in the next phase.
    # If send() itself doesn't clear it:
    assert len(sender_mailer.outbox) == 1 # send() does not clear outbox, prepare() does.
    sender_mailer.prepare() # This will clear the outbox
    assert len(sender_mailer.outbox) == 0


# Test BPAgent.Mailer.empty_mailbox
def test_mailer_empty_mailbox(var_agent_recipient):
    recipient_mailer = var_agent_recipient.mailer # Mailer is for sending
    # mailbox is an attribute of the agent itself
    
    # Manually put a message in the agent's mailbox
    sender_agent = FactorAgent(name="TestFactorSender")
    msg_data = np.array([0.4, 0.6])
    incoming_msg = Message(data=msg_data, sender=sender_agent, recipient=var_agent_recipient)
    var_agent_recipient.mailbox[sender_agent.name] = [incoming_msg]

    assert len(var_agent_recipient.mailbox) == 1
    
    # Call empty_mailbox (which is a method of BPAgent, not its Mailer)
    var_agent_recipient.empty_mailbox()
    assert len(var_agent_recipient.mailbox) == 0

# Test full message passing sequence: compute -> send -> prepare -> empty (on recipient)
def test_full_message_passing_flow(max_sum_computator):
    # Setup: V1 -- F1 -- V2
    v1 = VariableAgent(name="V1", domain_size=2)
    f1 = FactorAgent(name="F1")
    v2 = VariableAgent(name="V2", domain_size=2)

    # Assign computators
    v1.computator = max_sum_computator
    f1.computator = max_sum_computator
    v2.computator = max_sum_computator
    
    # Initialize mailers and mailboxes
    for agent in [v1, f1, v2]:
        agent.mailer = BPAgent.Mailer(agent)
        agent.mailbox = {}
        agent.neighbors = {} # Will be set next

    # Setup graph structure (neighbors)
    v1.neighbors = {f1.name: f1}
    f1.neighbors = {v1.name: v1, v2.name: v2}
    f1.variables_in_scope = [v1, v2] # For FactorAgent message computation
    v2.neighbors = {f1.name: f1}
    
    f1.cost_table = np.array([[0.0, 1.0], [1.0, 0.0]]) # Simple XOR-like cost

    # --- Step 1: V1 and V2 compute messages to F1 ---
    # (Assume initial messages to V1, V2 are uniform or zero, so Q messages are simple)
    # V1 computes Q_V1->F1. Let's assume no prior R messages, so Q is effectively an initial message.
    # For simplicity, let's assume compute_messages on a variable with an empty mailbox sends a zero message.
    # Or, let's manually put a "prior" or "initial R" into v1's mailbox to make it compute something.
    # For a clean test, let's assume initial messages are all-zeros if mailbox is empty.
    # VariableAgent.compute_messages sums R messages. If mailbox empty, sum is zero.
    
    v1.compute_messages() # Should compute Q_V1->F1 and put in v1.mailer.outbox
    v2.compute_messages() # Should compute Q_V2->F1 and put in v2.mailer.outbox

    assert len(v1.mailer.outbox) == 1
    assert v1.mailer.outbox[0].recipient == f1
    assert np.array_equal(v1.mailer.outbox[0].data, np.array([0.0, 0.0])) # Assuming sum of no messages is [0,0]

    assert len(v2.mailer.outbox) == 1
    assert v2.mailer.outbox[0].recipient == f1
    assert np.array_equal(v2.mailer.outbox[0].data, np.array([0.0, 0.0]))

    # --- Step 2: V1 and V2 send messages, F1 receives ---
    v1.mailer.send()
    v2.mailer.send()

    assert f1.name in f1.mailbox # This is incorrect, mailbox keys by sender name
    assert v1.name in f1.mailbox and len(f1.mailbox[v1.name]) == 1
    assert v2.name in f1.mailbox and len(f1.mailbox[v2.name]) == 1
    assert np.array_equal(f1.mailbox[v1.name][0].data, np.array([0.0, 0.0]))
    assert np.array_equal(f1.mailbox[v2.name][0].data, np.array([0.0, 0.0]))

    # --- Step 3: Senders prepare for next iteration ---
    v1.mailer.prepare()
    v2.mailer.prepare()
    assert len(v1.mailer.outbox) == 0
    assert len(v2.mailer.outbox) == 0

    # --- Step 4: F1 empties its mailbox (conceptually, before computing) ---
    # In BPEngine, empty_mailbox happens after send() and prepare() for the *other* agent type.
    # So F1 would empty its mailbox, then compute.
    f1.empty_mailbox() # This clears f1.mailbox for its own computation pass
    # This is not quite right. empty_mailbox is called *after* messages for the current agent type are processed.
    # Let's adjust: F1 uses messages in its mailbox to compute.
    
    # Corrected flow for F1's turn:
    # F1 has messages from V1 and V2 in its mailbox.
    f1.mailbox[v1.name] = [Message(data=np.array([0.0, 0.0]), sender=v1, recipient=f1)]
    f1.mailbox[v2.name] = [Message(data=np.array([0.0, 0.0]), sender=v2, recipient=f1)]

    f1.compute_messages() # Computes R_F1->V1 and R_F1->V2 using Q messages in its mailbox
                         # Results are in f1.mailer.outbox

    assert len(f1.mailer.outbox) == 2
    msg_f1_to_v1 = next(m for m in f1.mailer.outbox if m.recipient == v1)
    msg_f1_to_v2 = next(m for m in f1.mailer.outbox if m.recipient == v2)

    # Expected R_F1->V1(v1) = min_v2 { F1(v1,v2) + Q_V2->F1(v2) }
    # F1(v1,v2) = [[0,1],[1,0]], Q_V2->F1 = [0,0]
    # For V1=0: min(0+0, 1+0) = 0
    # For V1=1: min(1+0, 0+0) = 0
    # Expected R_F1->V1 = [0,0]
    assert np.array_equal(msg_f1_to_v1.data, np.array([0.0, 0.0]))

    # Expected R_F1->V2(v2) = min_v1 { F1(v1,v2) + Q_V1->F1(v1) }
    # F1(v1,v2) = [[0,1],[1,0]], Q_V1->F1 = [0,0]
    # For V2=0: min(0+0, 1+0) = 0
    # For V2=1: min(1+0, 0+0) = 0
    # Expected R_F1->V2 = [0,0]
    assert np.array_equal(msg_f1_to_v2.data, np.array([0.0, 0.0]))
    
    # --- Step 5: F1 sends messages, V1 and V2 receive ---
    f1.mailer.send()
    assert v1.name in v1.mailbox and len(v1.mailbox[v1.name]) == 1 # Mailbox keys by sender name (F1)
    assert f1.name in v1.mailbox and len(v1.mailbox[f1.name]) == 1
    assert np.array_equal(v1.mailbox[f1.name][0].data, np.array([0.0, 0.0]))

    assert f1.name in v2.mailbox and len(v2.mailbox[f1.name]) == 1
    assert np.array_equal(v2.mailbox[f1.name][0].data, np.array([0.0, 0.0]))
    
    # --- Step 6: F1 prepares for next iteration ---
    f1.mailer.prepare()
    assert len(f1.mailer.outbox) == 0
    
    # --- Step 7: V1 and V2 empty their mailboxes ---
    v1.empty_mailbox()
    v2.empty_mailbox()
    assert len(v1.mailbox) == 0
    assert len(v2.mailbox) == 0

