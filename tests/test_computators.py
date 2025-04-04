import pytest
import numpy as np
import logging
from bp_base.computators import MinSumComputator, MaxSumComputator
from bp_base.components import Message

# Configure logger
logger = logging.getLogger(__name__)

# Configure test logging
@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force= True
    )
    # Fix the typo in format string
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force = True
    )
    # Add this line to ensure logs are displayed
    logging.getLogger().setLevel(logging.INFO)
    yield

class MockNode:
    """Mock node class for testing purposes"""
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"Node({self.id})"

def test_compute_q_min_sum():
    """Test compute_Q method with min-sum algorithm"""
    logger.info("Starting test_compute_q_min_sum")

    # Create mock variable and factor nodes
    var_node = MockNode("X")
    factor1 = MockNode("F1")
    factor2 = MockNode("F2")
    factor3 = MockNode("F3")

    logger.debug(f"Created mock nodes: var={var_node}, factors={factor1}, {factor2}, {factor3}")

    # Create incoming messages from factors to variable
    # Each message has values for 3 possible states of the variable
    messages = [
        Message(np.array([0.0, 2.0, 5.0]), sender=factor1, recipient=var_node),
        Message(np.array([1.0, 0.0, 3.0]), sender=factor2, recipient=var_node),
        Message(np.array([2.0, 1.0, 0.0]), sender=factor3, recipient=var_node)
    ]

    logger.debug(f"Created test messages: {[(m.sender, m.data) for m in messages]}")

    # Initialize min-sum computator
    computator = MinSumComputator()

    # Compute variable-to-factor messages
    logger.info("Computing Q messages")
    results = computator.compute_Q(messages)

    # Should be 3 outgoing messages (one for each factor)
    expected_count = 3
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # Verify content of messages
    # To F1: should have combined F2 and F3 messages
    f1_message = next(m for m in results if m.recipient == factor1)
    assert f1_message.sender == var_node
    # The values should be F2 + F3 messages, normalized to min=0
    expected_f1 = np.array([3.0, 1.0, 3.0]) - 1.0  # [2.0, 0.0, 2.0]
    logger.info(f"EXPECTED F1 message: {expected_f1}")
    logger.info(f"ACTUAL F1 message: {f1_message.data}")
    np.testing.assert_array_almost_equal(f1_message.data, expected_f1,
                                         err_msg=f"F1 message mismatch: expected {expected_f1}, got {f1_message.data}")

    # To F2: should have combined F1 and F3 messages
    f2_message = next(m for m in results if m.recipient == factor2)
    # The values should be F1 + F3 messages, normalized to min=0
    expected_f2 = np.array([2.0, 3.0, 5.0]) - 2.0  # [0.0, 1.0, 3.0]
    logger.info(f"EXPECTED F2 message: {expected_f2}")
    logger.info(f"ACTUAL F2 message: {f2_message.data}")
    np.testing.assert_array_almost_equal(f2_message.data, expected_f2,
                                         err_msg=f"F2 message mismatch: expected {expected_f2}, got {f2_message.data}")

    # To F3: should have combined F1 and F2 messages
    f3_message = next(m for m in results if m.recipient == factor3)
    # The values should be F1 + F2 messages, normalized to min=0
    expected_f3 = np.array([1.0, 2.0, 8.0]) - 1.0  # [0.0, 1.0, 7.0]
    logger.info(f"EXPECTED F3 message: {expected_f3}")
    logger.info(f"ACTUAL F3 message: {f3_message.data}")
    np.testing.assert_array_almost_equal(f3_message.data, expected_f3,
                                         err_msg=f"F3 message mismatch: expected {expected_f3}, got {f3_message.data}")

    logger.info("test_compute_q_min_sum completed successfully")

def test_compute_q_max_sum():
    """Test compute_Q method with max-sum algorithm"""
    logger.info("Starting test_compute_q_max_sum")

    # Create mock variable and factor nodes
    var_node = MockNode("X")
    factor1 = MockNode("F1")
    factor2 = MockNode("F2")

    # Create incoming messages from factors to variable
    messages = [
        Message(np.array([0.0, 2.0, -1.0]), sender=factor1, recipient=var_node),
        Message(np.array([1.0, 0.0, 3.0]), sender=factor2, recipient=var_node)
    ]

    logger.debug(f"Created test messages: {[(m.sender, m.data) for m in messages]}")

    # Initialize max-sum computator
    computator = MaxSumComputator()

    # Compute variable-to-factor messages
    logger.info("Computing Q messages")
    results = computator.compute_Q(messages)

    # Check results
    expected_count = 2
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # To F1: should contain message from F2
    f1_message = next(m for m in results if m.recipient == factor1)
    expected_f1 = np.array([1.0, 0.0, 3.0]) - 0.0  # already normalized
    logger.info(f"EXPECTED F1 message: {expected_f1}")
    logger.info(f"ACTUAL F1 message: {f1_message.data}")
    np.testing.assert_array_almost_equal(f1_message.data, expected_f1,
                                         err_msg=f"F1 message mismatch: expected {expected_f1}, got {f1_message.data}")

    # To F2: should contain message from F1
    f2_message = next(m for m in results if m.recipient == factor2)
    expected_f2 = np.array([0.0, 2.0, -1.0]) - (-1.0)  # [1.0, 3.0, 0.0]
    logger.info(f"EXPECTED F2 message: {expected_f2}")
    logger.info(f"ACTUAL F2 message: {f2_message.data}")
    np.testing.assert_array_almost_equal(f2_message.data, expected_f2,
                                         err_msg=f"F2 message mismatch: expected {expected_f2}, got {f2_message.data}")

    logger.info("test_compute_q_max_sum completed successfully")

def test_compute_r_min_sum():
    """Test compute_R method with min-sum algorithm"""
    logger.info("Starting test_compute_r_min_sum")

    # Create mock variable and factor nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")

    # Create a simple 3x3 cost table for a factor connected to 2 variables
    # Each variable has 3 possible values
    cost_table = np.array([
        [5.0, 2.0, 8.0],  # costs when X1=0, X2={0,1,2}
        [1.0, 3.0, 4.0],  # costs when X1=1, X2={0,1,2}
        [7.0, 0.0, 6.0]   # costs when X1=2, X2={0,1,2}
    ])

    logger.debug(f"Created cost table:\n{cost_table}")

    # Create incoming messages from variables to factor
    incoming_messages = [
        Message(np.array([0.0, 1.0, 3.0]), sender=var1, recipient=factor_node),
        Message(np.array([2.0, 0.0, 4.0]), sender=var2, recipient=factor_node)
    ]

    logger.debug(f"Created incoming messages: {[(m.sender, m.data) for m in incoming_messages]}")

    # Initialize min-sum computator
    computator = MinSumComputator()

    # Compute factor-to-variable messages
    logger.info("Computing R messages")
    results = computator.compute_R(cost_table, incoming_messages)

    # Should be 2 outgoing messages (one for each variable)
    expected_count = 2
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # Verify content of messages
    # To X1: minimize over X2
    x1_message = next(m for m in results if m.recipient == var1)
    assert x1_message.sender == factor_node

    # For each value of X1, we add X2's message to the cost table and take min over X2
    x1_costs = cost_table + incoming_messages[1].data.reshape(1, 3)
    expected_x1 = np.min(x1_costs, axis=1)  # min over X2
    logger.info(f"EXPECTED X1 message: {expected_x1}")
    logger.info(f"ACTUAL X1 message: {x1_message.data}")
    np.testing.assert_array_almost_equal(x1_message.data, expected_x1,
                                         err_msg=f"X1 message mismatch: expected {expected_x1}, got {x1_message.data}")

    # To X2: minimize over X1
    x2_message = next(m for m in results if m.recipient == var2)
    assert x2_message.sender == factor_node

    # For each value of X2, we add X1's message to the cost table and take min over X1
    x2_costs = cost_table + incoming_messages[0].data.reshape(3, 1)
    expected_x2 = np.min(x2_costs, axis=0)  # min over X1
    logger.info(f"EXPECTED X2 message: {expected_x2}")
    logger.info(f"ACTUAL X2 message: {x2_message.data}")
    np.testing.assert_array_almost_equal(x2_message.data, expected_x2,
                                         err_msg=f"X2 message mismatch: expected {expected_x2}, got {x2_message.data}")

    logger.info("test_compute_r_min_sum completed successfully")

def test_compute_r_max_sum():
    """Test compute_R method with max-sum algorithm"""
    logger.info("Starting test_compute_r_max_sum")

    # Create mock variable and factor nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")
    var3 = MockNode("X3")

    # Create a 2x2x2 cost table for a factor connected to 3 variables
    # Each variable has 2 possible values
    cost_table = np.zeros((2, 2, 2))
    cost_table[0, 0, 0] = 1.0  # all variables = 0
    cost_table[0, 0, 1] = 2.0  # X1=0, X2=0, X3=1
    cost_table[0, 1, 0] = 3.0  # X1=0, X2=1, X3=0
    cost_table[0, 1, 1] = 4.0  # X1=0, X2=1, X3=1
    cost_table[1, 0, 0] = 5.0  # X1=1, X2=0, X3=0
    cost_table[1, 0, 1] = 6.0  # X1=1, X2=0, X3=1
    cost_table[1, 1, 0] = 7.0  # X1=1, X2=1, X3=0
    cost_table[1, 1, 1] = 8.0  # X1=1, X2=1, X3=1

    logger.debug(f"Created 3D cost table with shape {cost_table.shape}")

    # Create incoming messages from variables to factor
    incoming_messages = [
        Message(np.array([0.0, 1.0]), sender=var1, recipient=factor_node),
        Message(np.array([0.5, 0.0]), sender=var2, recipient=factor_node),
        Message(np.array([0.0, 0.2]), sender=var3, recipient=factor_node)
    ]

    logger.debug(f"Created incoming messages: {[(m.sender, m.data) for m in incoming_messages]}")

    # Initialize max-sum computator
    computator = MaxSumComputator()

    # Compute factor-to-variable messages
    logger.info("Computing R messages")
    results = computator.compute_R(cost_table, incoming_messages)

    # Should be 3 outgoing messages (one for each variable)
    expected_count = 3
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # Calculate expected values for comparison
    # For X1 message (maximize over X2 and X3)
    x1_costs = cost_table.copy()
    # Add X2's message reshaped for broadcasting
    x1_costs = x1_costs + incoming_messages[1].data.reshape(1, 2, 1)
    # Add X3's message reshaped for broadcasting
    x1_costs = x1_costs + incoming_messages[2].data.reshape(1, 1, 2)
    expected_x1 = np.max(x1_costs, axis=(1, 2))  # max over X2 and X3

    # Verify one of the messages as an example
    x1_message = next(m for m in results if m.recipient == var1)
    assert x1_message.sender == factor_node
    logger.info(f"EXPECTED X1 message shape: {expected_x1.shape}")
    logger.info(f"ACTUAL X1 message shape: {x1_message.data.shape}")
    logger.info(f"EXPECTED X1 message: {expected_x1}")
    logger.info(f"ACTUAL X1 message: {x1_message.data}")
    np.testing.assert_array_almost_equal(x1_message.data, expected_x1,
                                         err_msg=f"X1 message mismatch: expected {expected_x1}, got {x1_message.data}")

    logger.info("test_compute_r_max_sum completed successfully")

def test_empty_messages():
    """Test handling of empty message lists"""
    logger.info("Starting test_empty_messages")

    computator = MinSumComputator()

    # compute_Q with empty messages should return empty list
    logger.debug("Testing compute_Q with empty messages")
    expected_result = []
    actual_result = computator.compute_Q([])
    logger.info(f"EXPECTED: empty list, ACTUAL: {actual_result}")
    assert actual_result == expected_result, f"Expected empty list but got {actual_result}"

    # compute_R with empty messages should return empty list
    logger.debug("Testing compute_R with empty messages")
    expected_result = []
    actual_result = computator.compute_R(np.array([]), [])
    logger.info(f"EXPECTED: empty list, ACTUAL: {actual_result}")
    assert actual_result == expected_result, f"Expected empty list but got {actual_result}"

    logger.info("test_empty_messages completed successfully")

def test_compute_r_min_sum_4vars():
    """Test compute_R method with min-sum algorithm for 4 variables"""
    logger.info("Starting test_compute_r_min_sum_4vars")

    # Create mock variable and factor nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")
    var3 = MockNode("X3")
    var4 = MockNode("X4")

    # Create a 2x2x2x2 cost table for a factor connected to 4 variables
    # Each variable has 2 possible values
    cost_table = np.zeros((2, 2, 2, 2))
    # Fill with some test values (different value for each combination)
    index = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    cost_table[i, j, k, l] = index
                    index += 1

    logger.debug(f"Created 4D cost table with shape {cost_table.shape}")

    # Create incoming messages from variables to factor
    incoming_messages = [
        Message(np.array([0.0, 1.0]), sender=var1, recipient=factor_node),
        Message(np.array([0.5, 0.0]), sender=var2, recipient=factor_node),
        Message(np.array([0.0, 0.2]), sender=var3, recipient=factor_node),
        Message(np.array([0.3, 0.0]), sender=var4, recipient=factor_node)
    ]

    logger.debug(f"Created incoming messages: {[(m.sender, m.data) for m in incoming_messages]}")

    # Initialize min-sum computator
    computator = MinSumComputator()

    # Compute factor-to-variable messages
    logger.info("Computing R messages")
    results = computator.compute_R(cost_table, incoming_messages)

    # Should be 4 outgoing messages (one for each variable)
    expected_count = 4
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # Calculate expected values for comparison
    # For X1 message (minimize over X2, X3, and X4)
    x1_costs = cost_table.copy()
    # Add X2's message reshaped for broadcasting
    x1_costs = x1_costs + incoming_messages[1].data.reshape(1, 2, 1, 1)
    # Add X3's message reshaped for broadcasting
    x1_costs = x1_costs + incoming_messages[2].data.reshape(1, 1, 2, 1)
    # Add X4's message reshaped for broadcasting
    x1_costs = x1_costs + incoming_messages[3].data.reshape(1, 1, 1, 2)
    expected_x1 = np.min(x1_costs, axis=(1, 2, 3))  # min over X2, X3, and X4

    # Verify message to X1
    x1_message = next(m for m in results if m.recipient == var1)
    assert x1_message.sender == factor_node
    logger.info(f"EXPECTED X1 message: {expected_x1}")
    logger.info(f"ACTUAL X1 message: {x1_message.data}")
    np.testing.assert_array_almost_equal(x1_message.data, expected_x1,
                                      err_msg=f"X1 message mismatch: expected {expected_x1}, got {x1_message.data}")

    # Verify message to X2
    x2_costs = cost_table.copy()
    x2_costs = x2_costs + incoming_messages[0].data.reshape(2, 1, 1, 1)
    x2_costs = x2_costs + incoming_messages[2].data.reshape(1, 1, 2, 1)
    x2_costs = x2_costs + incoming_messages[3].data.reshape(1, 1, 1, 2)
    expected_x2 = np.min(x2_costs, axis=(0, 2, 3))  # min over X1, X3, and X4

    x2_message = next(m for m in results if m.recipient == var2)
    assert x2_message.sender == factor_node
    logger.info(f"EXPECTED X2 message: {expected_x2}")
    logger.info(f"ACTUAL X2 message: {x2_message.data}")
    np.testing.assert_array_almost_equal(x2_message.data, expected_x2,
                                      err_msg=f"X2 message mismatch: expected {expected_x2}, got {x2_message.data}")

    logger.info("test_compute_r_min_sum_4vars completed successfully")

def test_compute_r_max_sum_mixed_dims():
    """Test compute_R method with max-sum algorithm for 4 variables with mixed dimensions"""
    logger.info("Starting test_compute_r_max_sum_mixed_dims")

    # Create mock variable and factor nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")
    var3 = MockNode("X3")
    var4 = MockNode("X4")

    # Create a 2x3x2x3 cost table for a factor connected to 4 variables
    # X1 has 2 values, X2 has 3 values, X3 has 2 values, X4 has 3 values
    cost_table = np.zeros((2, 3, 2, 3))
    
    # Fill with some test values (just use the index for simplicity)
    index = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                for l in range(3):
                    cost_table[i, j, k, l] = index
                    index += 1

    logger.debug(f"Created mixed dimensions cost table with shape {cost_table.shape}")

    # Create incoming messages from variables to factor with appropriate dimensions
    incoming_messages = [
        Message(np.array([0.0, 1.0]), sender=var1, recipient=factor_node),                # X1: 2 values
        Message(np.array([0.5, 0.0, 0.8]), sender=var2, recipient=factor_node),          # X2: 3 values
        Message(np.array([0.0, 0.2]), sender=var3, recipient=factor_node),                # X3: 2 values
        Message(np.array([0.3, 0.0, 0.4]), sender=var4, recipient=factor_node)           # X4: 3 values
    ]

    logger.debug(f"Created incoming messages: {[(m.sender, m.data.shape) for m in incoming_messages]}")

    # Initialize max-sum computator
    computator = MaxSumComputator()

    # Compute factor-to-variable messages
    logger.info("Computing R messages")
    results = computator.compute_R(cost_table, incoming_messages)

    # Should be 4 outgoing messages (one for each variable)
    expected_count = 4
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # Verify dimensions of resulting messages
    assert results[0].data.shape == (2,), f"Expected X1 message shape (2,), got {results[0].data.shape}"
    assert results[1].data.shape == (3,), f"Expected X2 message shape (3,), got {results[1].data.shape}"
    assert results[2].data.shape == (2,), f"Expected X3 message shape (2,), got {results[2].data.shape}"
    assert results[3].data.shape == (3,), f"Expected X4 message shape (3,), got {results[3].data.shape}"

    # Calculate expected values for X1 message
    x1_costs = cost_table.copy()
    x1_costs = x1_costs + incoming_messages[1].data.reshape(1, 3, 1, 1)
    x1_costs = x1_costs + incoming_messages[2].data.reshape(1, 1, 2, 1)
    x1_costs = x1_costs + incoming_messages[3].data.reshape(1, 1, 1, 3)
    expected_x1 = np.max(x1_costs, axis=(1, 2, 3))  # max over X2, X3, and X4

    # Verify message to X1
    x1_message = next(m for m in results if m.recipient == var1)
    logger.info(f"EXPECTED X1 message: {expected_x1}")
    logger.info(f"ACTUAL X1 message: {x1_message.data}")
    np.testing.assert_array_almost_equal(x1_message.data, expected_x1,
                                      err_msg=f"X1 message mismatch: expected {expected_x1}, got {x1_message.data}")

    logger.info("test_compute_r_max_sum_mixed_dims completed successfully")

def test_compute_r_min_sum_5vars():
    """Test compute_R method with min-sum algorithm for 5 variables"""
    logger.info("Starting test_compute_r_min_sum_5vars")

    # Create mock variable and factor nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")
    var3 = MockNode("X3")
    var4 = MockNode("X4")
    var5 = MockNode("X5")

    # Create a cost table for 5 binary variables (2x2x2x2x2)
    cost_table = np.zeros((2, 2, 2, 2, 2))
    
    # Fill with alternating values for simplicity
    for idx in np.ndindex(cost_table.shape):
        cost_table[idx] = sum(idx) % 2

    logger.debug(f"Created 5D cost table with shape {cost_table.shape}")

    # Create incoming messages from variables to factor
    incoming_messages = [
        Message(np.array([0.0, 0.1]), sender=var1, recipient=factor_node),
        Message(np.array([0.2, 0.0]), sender=var2, recipient=factor_node),
        Message(np.array([0.0, 0.3]), sender=var3, recipient=factor_node),
        Message(np.array([0.4, 0.0]), sender=var4, recipient=factor_node),
        Message(np.array([0.0, 0.5]), sender=var5, recipient=factor_node)
    ]

    logger.debug(f"Created incoming messages: {[(m.sender, m.data) for m in incoming_messages]}")

    # Initialize min-sum computator
    computator = MinSumComputator()

    # Compute factor-to-variable messages
    logger.info("Computing R messages")
    results = computator.compute_R(cost_table, incoming_messages)

    # Should be 5 outgoing messages (one for each variable)
    expected_count = 5
    actual_count = len(results)
    logger.info(f"EXPECTED count: {expected_count}, ACTUAL count: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} messages but got {actual_count}"

    # Verify all messages have the correct shapes
    for i, var in enumerate([var1, var2, var3, var4, var5]):
        message = next(m for m in results if m.recipient == var)
        assert message.data.shape == (2,), f"Expected message shape (2,), got {message.data.shape}"
        assert message.sender == factor_node, "Factor node should be the sender"

    # Calculate expected values for X1 message (minimizing over all other variables)
    x1_costs = cost_table.copy()
    # Add messages from all other variables
    x1_costs = x1_costs + incoming_messages[1].data.reshape(1, 2, 1, 1, 1)
    x1_costs = x1_costs + incoming_messages[2].data.reshape(1, 1, 2, 1, 1)
    x1_costs = x1_costs + incoming_messages[3].data.reshape(1, 1, 1, 2, 1)
    x1_costs = x1_costs + incoming_messages[4].data.reshape(1, 1, 1, 1, 2)
    expected_x1 = np.min(x1_costs, axis=(1, 2, 3, 4))  # min over all variables except X1

    # Verify message to X1
    x1_message = next(m for m in results if m.recipient == var1)
    logger.info(f"EXPECTED X1 message: {expected_x1}")
    logger.info(f"ACTUAL X1 message: {x1_message.data}")
    np.testing.assert_array_almost_equal(x1_message.data, expected_x1,
                                      err_msg=f"X1 message mismatch: expected {expected_x1}, got {x1_message.data}")

    logger.info("test_compute_r_min_sum_5vars completed successfully")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Running tests directly")
    pytest.main(["-xvs", "--log-cli-level=INFO", __file__])

