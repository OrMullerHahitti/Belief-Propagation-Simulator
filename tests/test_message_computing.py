# Tests for message computing
import pytest
import numpy as np
from propflow.bp.computators import MaxSumComputator, MinSumComputator
from propflow.core import Agent  # For basic sender/recipient typing if needed
from propflow.core import Message
from propflow.core import VariableAgent, FactorAgent


# Fixture for a MaxSumComputator
@pytest.fixture
def max_sum_computator():
    return MaxSumComputator()


# Fixture for a MinSumComputator
@pytest.fixture
def min_sum_computator():
    return MinSumComputator()


# Fixture for dummy agents
@pytest.fixture
def dummy_sender():
    return Agent(name="DummySender", node_type="test")


@pytest.fixture
def dummy_recipient():
    return Agent(name="DummyRecipient", node_type="test")


# Fixtures for variable and factor agents
@pytest.fixture
def variable_agent():
    return VariableAgent(name="x1", domain=3)


@pytest.fixture
def factor_agent():
    # Create a factor agent with a simple cost table
    factor = FactorAgent(
        name="f1",
        domain=3,
        ct_creation_func=create_test_cost_table,
    )
    # Set up connection numbers for the factor
    factor.connection_number = {"x1": 0, "x2": 1}
    return factor


def create_test_cost_table(num_vars, domain_size, **kwargs):
    """Helper function to create a simple cost table for testing"""
    # Create a 3x3 cost table with predictable values
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# Test compute_Q method with simple messages
def test_compute_q_simple(max_sum_computator, variable_agent, factor_agent):
    """Test Q message computation with a simple case"""
    # Create test messages going into variable agent
    message1 = Message(
        data=np.array([1.0, 2.0, 3.0]), sender=factor_agent, recipient=variable_agent
    )

    # Test with a single message
    result = max_sum_computator.compute_Q([message1])
    assert len(result) == 1
    assert result[0].sender == variable_agent
    assert result[0].recipient == factor_agent
    np.testing.assert_array_equal(
        result[0].data, np.zeros(3)
    )  # Should be zeros for single message

    # Create another factor for testing with multiple messages
    factor_agent2 = FactorAgent(
        name="f2", domain=3, ct_creation_func=create_test_cost_table
    )
    message2 = Message(
        data=np.array([4.0, 5.0, 6.0]), sender=factor_agent2, recipient=variable_agent
    )

    # Test with multiple messages
    result = max_sum_computator.compute_Q([message1, message2])
    assert len(result) == 2

    # For first message, the outgoing message should be the data from message2
    np.testing.assert_array_equal(result[0].data, np.array([4.0, 5.0, 6.0]))

    # For second message, the outgoing message should be the data from message1
    np.testing.assert_array_equal(result[1].data, np.array([1.0, 2.0, 3.0]))


# Test compute_Q method with the MinSumComputator
def test_compute_q_min_sum(min_sum_computator, variable_agent, factor_agent):
    """Test Q message computation with MinSumComputator"""
    # Create test messages going into variable agent
    factor_agent2 = FactorAgent(
        name="f2", domain=3, ct_creation_func=create_test_cost_table
    )

    message1 = Message(
        data=np.array([1.0, 2.0, 3.0]), sender=factor_agent, recipient=variable_agent
    )

    message2 = Message(
        data=np.array([4.0, 5.0, 6.0]), sender=factor_agent2, recipient=variable_agent
    )

    # Test with multiple messages
    result = min_sum_computator.compute_Q([message1, message2])
    assert len(result) == 2

    # Same behavior for min-sum as max-sum for Q messages
    np.testing.assert_array_equal(result[0].data, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(result[1].data, np.array([1.0, 2.0, 3.0]))


# Test compute_R method with simple cost table
def test_compute_r_simple(max_sum_computator, variable_agent, factor_agent):
    """Test R message computation with a simple cost table"""
    # Set up a simple cost table for the factor
    cost_table = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Create a second variable agent
    variable_agent2 = VariableAgent(name="x2", domain=3)

    # Create test messages going into factor agent
    message1 = Message(
        data=np.array([0.0, 0.0, 0.0]),  # Neutral values for addition
        sender=variable_agent,
        recipient=factor_agent,
    )

    message2 = Message(
        data=np.array([0.0, 0.0, 0.0]),  # Neutral values for addition
        sender=variable_agent2,
        recipient=factor_agent,
    )

    # Set up factor connections for compute_R
    factor_agent.connection_number = {variable_agent.name: 0, variable_agent2.name: 1}

    # Test R message computation
    result = max_sum_computator.compute_R(cost_table, [message1, message2])
    assert len(result) == 2

    # Check message to first variable (marginalize out dim 1)
    # For max-sum, this should be [3, 6, 9] (max of each row)
    np.testing.assert_array_equal(result[0].data, np.array([3, 6, 9]))

    # Check message to second variable (marginalize out dim 0)
    # For max-sum, this should be [7, 8, 9] (max of each column)
    np.testing.assert_array_equal(result[1].data, np.array([7, 8, 9]))


# Test compute_R method with the MinSumComputator
def test_compute_r_min_sum(min_sum_computator, variable_agent, factor_agent):
    """Test R message computation with MinSumComputator"""
    # Set up a simple cost table for the factor
    cost_table = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Create a second variable agent
    variable_agent2 = VariableAgent(name="x2", domain=3)

    # Create test messages going into factor agent
    message1 = Message(
        data=np.array([0.0, 0.0, 0.0]),  # Neutral values for addition
        sender=variable_agent,
        recipient=factor_agent,
    )

    message2 = Message(
        data=np.array([0.0, 0.0, 0.0]),  # Neutral values for addition
        sender=variable_agent2,
        recipient=factor_agent,
    )

    # Set up factor connections for compute_R
    factor_agent.connection_number = {variable_agent.name: 0, variable_agent2.name: 1}

    # Test R message computation
    result = min_sum_computator.compute_R(cost_table, [message1, message2])
    assert len(result) == 2

    # Check message to first variable (marginalize out dim 1)
    # For min-sum, this should be [1, 4, 7] (min of each row)
    np.testing.assert_array_equal(result[0].data, np.array([1, 4, 7]))

    # Check message to second variable (marginalize out dim 0)
    # For min-sum, this should be [1, 2, 3] (min of each column)
    np.testing.assert_array_equal(result[1].data, np.array([1, 2, 3]))


# Test compute_R with non-zero incoming messages
def test_compute_r_with_incoming_values(
    max_sum_computator, variable_agent, factor_agent
):
    """Test R message computation with non-zero incoming messages"""
    # Set up a simple cost table for the factor
    cost_table = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Create a second variable agent
    variable_agent2 = VariableAgent(name="x2", domain=3)

    # Create test messages with non-zero values
    message1 = Message(
        data=np.array([1.0, 0.0, 2.0]),  # Will be added to cost table
        sender=variable_agent,
        recipient=factor_agent,
    )

    message2 = Message(
        data=np.array([0.0, 1.0, 3.0]),  # Will be added to cost table
        sender=variable_agent2,
        recipient=factor_agent,
    )

    # Set up factor connections
    factor_agent.connection_number = {variable_agent.name: 0, variable_agent2.name: 1}

    # Test with max-sum (addition for combine, max for reduce)
    result = max_sum_computator.compute_R(cost_table, [message1, message2])

    # Expected computation for message to var1 (max over dim 1):
    # For row 0: max([1+0, 2+1, 3+3]) = max([1, 3, 6]) = 6
    # For row 1: max([4+0, 5+1, 6+3]) = max([4, 6, 9]) = 9
    # For row 2: max([7+0, 8+1, 9+3]) = max([7, 9, 12]) = 12
    # Result: [6, 9, 12]
    np.testing.assert_array_equal(result[0].data, np.array([6, 9, 12]))

    # Expected computation for message to var2 (max over dim 0):
    # Due to the message computation logic in the current implementation,
    # the actual calculation for the message to var2 is:
    # For col 0: max([1+1, 4+0, 7+2]) = max([2, 4, 9]) = 9
    # For col 1: max([2+1, 5+0, 8+2]) = max([3, 5, 10]) = 10
    # For col 2: max([3+1, 6+0, 9+2]) = max([4, 6, 11]) = 11
    # Result: [9, 10, 11]
    np.testing.assert_array_equal(result[1].data, np.array([9, 10, 11]))


#
# # Test JIT acceleration if available
# def test_jit_acceleration_flag():
#     """Test that the JIT acceleration flag is properly set"""
#     from bp.bp_computators import HAS_NUMBA
#
#     computator = BPComputator(reduce_func=np.max, combine_func=np.add, use_jit=True)
#     assert computator._use_jit == HAS_NUMBA
#
#     computator_disabled = BPComputator(
#         reduce_func=np.max, combine_func=np.add, use_jit=False
#     )
#     assert not computator_disabled._use_jit
#
#     # For common numpy functions, operation type should be recognized
#     assert computator._operation_type == 0  # 0 for addition
#
#     # Test with multiplication
#     computator2 = BPComputator(
#         reduce_func=np.max, combine_func=np.multiply, use_jit=False
#     )
#     assert computator2._operation_type == 1  # 1 for multiplication
#
#
# def test_jit_vs_numpy_paths(variable_agent, factor_agent):
#     """Ensure JIT and non-JIT implementations produce the same results."""
#     # Prepare messages
#     factor_agent2 = FactorAgent(
#         name="f2", domain=3, ct_creation_func=create_test_cost_table
#     )
#     msg1 = Message(
#         data=np.array([1.0, 2.0, 3.0]), sender=factor_agent, recipient=variable_agent
#     )
#     msg2 = Message(
#         data=np.array([4.0, 5.0, 6.0]), sender=factor_agent2, recipient=variable_agent
#     )
#
#     comp_jit = MaxSumComputator(use_jit=True)
#     comp_np = MaxSumComputator(use_jit=False)
#
#     res_jit = comp_jit.compute_Q([msg1, msg2])
#     res_np = comp_np.compute_Q([msg1, msg2])
#     for a, b in zip(res_jit, res_np):
#         np.testing.assert_array_equal(a.data, b.data)
#
#     # R-message comparison
#     cost_table = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     variable_agent2 = VariableAgent(name="x2", domain=3)
#     factor_agent.connection_number = {variable_agent.name: 0, variable_agent2.name: 1}
#     msg_r1 = Message(data=np.zeros(3), sender=variable_agent, recipient=factor_agent)
#     msg_r2 = Message(data=np.zeros(3), sender=variable_agent2, recipient=factor_agent)
#
#     res_jit_r = comp_jit.compute_R(cost_table, [msg_r1, msg_r2])
#     res_np_r = comp_np.compute_R(cost_table, [msg_r1, msg_r2])
#     for a, b in zip(res_jit_r, res_np_r):
#         np.testing.assert_array_equal(a.data, b.data)
