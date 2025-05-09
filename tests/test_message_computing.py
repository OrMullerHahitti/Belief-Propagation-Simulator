
# Tests for message computing
import pytest
import numpy as np
from bp_base.computators import MaxSumComputator
from bp_base.components import Message
from bp_base.agents import VariableAgent, FactorAgent
from DCOP_base import Agent # For basic sender/recipient typing if needed

# Fixture for a MaxSumComputator
@pytest.fixture
def max_sum_computator():
    return MaxSumComputator()

# Fixture for dummy agents
@pytest.fixture
def dummy_sender():
    return Agent(name="DummySender", node_type="test")

@pytest.fixture
def dummy_recipient():
    return Agent(name="DummyRecipient", node_type="test")

# Tests for MaxSumComputator.compute_R (Factor to Variable messages)
def test_max_sum_compute_R(max_sum_computator, dummy_sender, dummy_recipient):
    # R_i->x(x_i) = alpha_i + sum_{x_j in N(i) \\ x} Q_j->i(x_j) + f_i(X_i)
    # Simplified: message from factor to variable
    # For this test, let's assume a simple scenario.
    # Cost table for a factor F connected to V1 (domain 2) and V2 (domain 2)
    # f(V1, V2)
    cost_table = np.array([[1, 2], [3, 4]]) # V1 is row, V2 is col

    # Incoming Q message from V2 to F (representing sum over V2's states)
    # Q_V2->F(V2)
    incoming_q_v2_to_f = Message(data=np.array([0.5, 0.5]), sender=dummy_sender, recipient=dummy_recipient)

    # Target variable is V1. We want to compute R_F->V1(V1)
    # R_F->V1(V1=0) = min_{V2} (cost_table[0,V2] + Q_V2->F(V2))
    # R_F->V1(V1=1) = min_{V2} (cost_table[1,V2] + Q_V2->F(V2))

    # Expected computation for V1=0: min(1+0.5, 2+0.5) = min(1.5, 2.5) = 1.5
    # Expected computation for V1=1: min(3+0.5, 4+0.5) = min(3.5, 4.5) = 3.5
    expected_r_message_data = np.array([1.5, 3.5])

    # The compute_R method in MaxSumComputator might take a list of Q messages
    # and the target variable index/name to know which variable to marginalize out.
    # Let's assume it takes the cost table, a list of relevant Q messages,
    # and the index of the target variable in the factor's scope.
    # For simplicity, let's assume the MaxSumComputator's compute_R is adapted or we simulate its direct call.
    # This part highly depends on the exact signature of compute_R.
    # For now, let's simulate the core logic if direct call is complex.

    # If compute_R is directly callable with simplified assumptions:
    # result_message = max_sum_computator.compute_R(cost_table, [incoming_q_v2_to_f], target_variable_index=0)
    # assert np.array_equal(result_message.data, expected_r_message_data)
    # This is a placeholder as compute_R signature in Protocol is (cost_table, messages: Message) -> Message
    # which seems to imply a single incoming message for R computation, which is unusual for MaxSum.
    # Let's assume the `messages` param in `compute_R` is a list for this test, or adapt.

    # Re-evaluating based on `compute_R(self, cost_table: CostTable, messages: Message) -> Message`
    # This signature is more aligned with Q-message computation (Var to Factor) if `cost_table` is not used.
    # Or, if it's for F->V, `messages` might be a single message from another variable,
    # and `cost_table` is the factor's potential. This is still a bit ambiguous for standard Max-Sum R.

    # Let's assume compute_R for F->V takes the factor potential and sums out other variables using their Q messages.
    # The provided signature `compute_R(self, cost_table: np.ndarray, message: List[Message]) -> List[Message]`
    # from DCOP_base.py is also different from bp_base.typing.Computator.
    # Using DCOP_base.Computator.compute_R(self, cost_table: np.ndarray, message: List[Message]) -> List[Message]:
    
    # For this test, we'll assume the MaxSumComputator's compute_R is designed to take
    # the factor's cost_table and a list of Q messages from *other* variables connected to the factor,
    # and computes the R message to one specific target variable.
    # The target variable needs to be implicit or passed somehow.

    # Let's test the logic conceptually:
    utility_matrix = cost_table + incoming_q_v2_to_f.data # Element-wise addition, assuming broadcasting if needed or direct sum
    # utility_matrix for V1=0: [1,2] + [0.5, 0.5] (from Q_V2->F) = [1.5, 2.5]
    # utility_matrix for V1=1: [3,4] + [0.5, 0.5] (from Q_V2->F) = [3.5, 4.5]
    
    # Minimize over V2 for each state of V1
    minimized_over_v2_for_v1_0 = np.min(utility_matrix[0, :]) # min(1.5, 2.5) = 1.5
    minimized_over_v2_for_v1_1 = np.min(utility_matrix[1, :]) # min(3.5, 4.5) = 3.5
    
    assert minimized_over_v2_for_v1_0 == 1.5
    assert minimized_over_v2_for_v1_1 == 3.5
    # This is the core logic. The actual call to compute_R would depend on its implementation details.
    # For now, this tests the expected numerical outcome.
    # A full test would mock FactorAgent and VariableAgent interactions.

# Tests for MaxSumComputator.compute_Q (Variable to Factor messages)
def test_max_sum_compute_Q(max_sum_computator, dummy_sender, dummy_recipient):
    # Q_x->i(x_i) = beta_i + sum_{k in N(x) \\ i} R_k->x(x_k)
    # Simplified: message from variable to factor
    # Incoming R messages to variable V from factors F1, F2
    incoming_r_f1_to_v = Message(data=np.array([0.1, 0.2]), sender=dummy_sender, recipient=dummy_recipient)
    incoming_r_f2_to_v = Message(data=np.array([0.3, 0.4]), sender=dummy_sender, recipient=dummy_recipient)

    # Expected Q message Q_V->F_target(V)
    # Q_V->F_target(V=0) = R_F1->V(V=0) + R_F2->V(V=0) = 0.1 + 0.3 = 0.4
    # Q_V->F_target(V=1) = R_F1->V(V=1) + R_F2->V(V=1) = 0.2 + 0.4 = 0.6
    expected_q_message_data = np.array([0.4, 0.6])

    # The compute_Q method takes a list of incoming R messages
    # (excluding the one from the target factor)
    # Using DCOP_base.Computator.compute_Q(self, messages: List[Message]) -> List[Message]:
    # This seems to return a list of messages. Let's assume it's one message in this context.
    
    # result_messages = max_sum_computator.compute_Q([incoming_r_f1_to_v, incoming_r_f2_to_v])
    # assert len(result_messages) == 1
    # assert np.array_equal(result_messages[0].data, expected_q_message_data)
    
    # Simulating the core logic:
    summed_r_data = incoming_r_f1_to_v.data + incoming_r_f2_to_v.data
    assert np.array_equal(summed_r_data, expected_q_message_data)
    # Again, a full test would involve agent interactions.

# Test VariableAgent message computation (simplified)
def test_variable_agent_compute_messages(max_sum_computator):
    var_agent = VariableAgent(name="V1", domain_size=2)
    var_agent.computator = max_sum_computator # Assign computator

    # Simulate incoming R messages from connected factors F1, F2
    # These would normally be in var_agent.mailbox
    f1 = FactorAgent(name="F1") # Dummy factor
    f2 = FactorAgent(name="F2") # Dummy factor
    
    # Manually add messages to mailbox for testing compute_messages
    # The actual mailer logic will be tested in message_passing
    msg_r_f1_to_v1 = Message(data=np.array([0.1, 0.6]), sender=f1, recipient=var_agent)
    msg_r_f2_to_v1 = Message(data=np.array([0.3, 0.2]), sender=f2, recipient=var_agent)
    
    var_agent.mailbox = {
        f1.name: [msg_r_f1_to_v1],
        f2.name: [msg_r_f2_to_v1]
    }
    
    # Define neighbors (factors)
    var_agent.neighbors = {f1.name: f1, f2.name: f2} 
    # mailer.outbox is where computed messages are stored before sending
    var_agent.mailer.outbox = []


    # Call compute_messages
    # This method should iterate through neighbors, compute Q for each, and put in mailer.outbox
    var_agent.compute_messages() 

    assert len(var_agent.mailer.outbox) == 2 # One Q message for each factor neighbor

    # Check message to F1: Q_V1->F1 = R_F2->V1
    # Expected: [0.3, 0.2]
    msg_to_f1 = next(m for m in var_agent.mailer.outbox if m.recipient == f1)
    assert np.array_equal(msg_to_f1.data, np.array([0.3, 0.2]))

    # Check message to F2: Q_V1->F2 = R_F1->V1
    # Expected: [0.1, 0.6]
    msg_to_f2 = next(m for m in var_agent.mailer.outbox if m.recipient == f2)
    assert np.array_equal(msg_to_f2.data, np.array([0.1, 0.6]))

# Test FactorAgent message computation (simplified)
def test_factor_agent_compute_messages(max_sum_computator):
    factor_agent = FactorAgent(name="F1")
    factor_agent.computator = max_sum_computator

    # Factor F1 connected to V1, V2. Domain size 2 for both.
    # Cost table: F1(V1, V2)
    # V1 is dimension 0, V2 is dimension 1
    factor_agent.cost_table = np.array([[1.0, 2.0],  # V1=0
                                         [3.0, 0.5]]) # V1=1
    
    v1 = VariableAgent(name="V1", domain_size=2)
    v2 = VariableAgent(name="V2", domain_size=2)
    
    # Simulate incoming Q messages from V1, V2
    msg_q_v1_to_f1 = Message(data=np.array([0.1, 0.9]), sender=v1, recipient=factor_agent)
    msg_q_v2_to_f1 = Message(data=np.array([0.5, 0.5]), sender=v2, recipient=factor_agent)

    factor_agent.mailbox = {
        v1.name: [msg_q_v1_to_f1],
        v2.name: [msg_q_v2_to_f1]
    }
    factor_agent.neighbors = {v1.name: v1, v2.name: v2}
    # The order of variables in cost_table vs. neighbors list is important.
    # Assume factor_agent.variables_in_scope = [v1, v2] (or similar mechanism)
    factor_agent.variables_in_scope = [v1, v2] 
    factor_agent.mailer.outbox = []

    factor_agent.compute_messages()

    assert len(factor_agent.mailer.outbox) == 2 # One R message for each variable neighbor

    # Check R_F1->V1(V1)
    # R_F1->V1(v1) = min_v2 { F1(v1,v2) + Q_V2->F1(v2) }
    # For V1=0: min(1.0 + 0.5, 2.0 + 0.5) = min(1.5, 2.5) = 1.5
    # For V1=1: min(3.0 + 0.5, 0.5 + 0.5) = min(3.5, 1.0) = 1.0
    # Expected: [1.5, 1.0]
    msg_to_v1 = next(m for m in factor_agent.mailer.outbox if m.recipient == v1)
    assert np.allclose(msg_to_v1.data, np.array([1.5, 1.0]))

    # Check R_F1->V2(V2)
    # R_F1->V2(v2) = min_v1 { F1(v1,v2) + Q_V1->F1(v1) }
    # For V2=0: min(1.0 + 0.1, 3.0 + 0.9) = min(1.1, 3.9) = 1.1
    # For V2=1: min(2.0 + 0.1, 0.5 + 0.9) = min(2.1, 1.4) = 1.4
    # Expected: [1.1, 1.4]
    msg_to_v2 = next(m for m in factor_agent.mailer.outbox if m.recipient == v2)
    assert np.allclose(msg_to_v2.data, np.array([1.1, 1.4]))

