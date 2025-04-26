import pytest
import numpy as np
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent

@pytest.fixture
def simple_graph():
    pass

def test_variable_agent_post_init(simple_graph):
    fg, v1, v2, f = simple_graph
    # VariableAgent should have mailbox and messages_to_send initialized
    assert hasattr(v1, "mailbox")
    assert hasattr(v1, "messages_to_send")
    assert isinstance(v1.inbox, list)
    assert isinstance(v1.messages_to_send, list)
    # Domain and name
    assert v1.domain == 3
    assert v1.name == "x1"
    # computator should be set by FactorGraph
    assert hasattr(v1, "computator")
    # After graph init, mailbox/messages_to_send should have messages to neighbors
    assert all(msg.sender == v1 or msg.recipient == v1 for msg in v1.messages_to_send)
    # Assignment and belief properties should work (even if mailbox is zeros)
    _ = v1.curr_assignment
    _ = v1.curr_belief

def test_factor_agent_post_init(simple_graph):
    fg, v1, v2, f = simple_graph
    # FactorAgent should have mailbox and messages_to_send initialized
    assert hasattr(f, "mailbox")
    assert hasattr(f, "messages_to_send")
    assert isinstance(f.mailbox, list)
    assert isinstance(f.messages_to_send, list)
    # Domain and name
    assert f.domain == 3
    assert f.name == "f12"
    # computator should be set by FactorGraph
    assert hasattr(f, "computator")
    # Cost table should be initialized
    assert hasattr(f, "cost_table")
    assert isinstance(f.cost_table, np.ndarray)
    # connection_number should map variables to dimensions
    assert v1 in f.connection_number and v2 in f.connection_number
    # Can compute messages (should return a list)
    msgs = f.compute_messages()
    assert isinstance(msgs, list)
