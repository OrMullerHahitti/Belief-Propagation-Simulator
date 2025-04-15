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
    expected_f1 = np.array([3.0, 1.0, 3.0])
    logger.info(f"EXPECTED F1 message: {expected_f1}")
    logger.info(f"ACTUAL F1 message: {f1_message.data}")
    np.testing.assert_array_almost_equal(f1_message.data, expected_f1,
                                         err_msg=f"F1 message mismatch: expected {expected_f1}, got {f1_message.data}")

    # To F2: should have combined F1 and F3 messages
    f2_message = next(m for m in results if m.recipient == factor2)
    # The values should be F1 + F3 messages, normalized to min=0
    expected_f2 = np.array([2.0, 3.0, 5.0])
    logger.info(f"EXPECTED F2 message: {expected_f2}")
    logger.info(f"ACTUAL F2 message: {f2_message.data}")
    np.testing.assert_array_almost_equal(f2_message.data, expected_f2,
                                         err_msg=f"F2 message mismatch: expected {expected_f2}, got {f2_message.data}")

    # To F3: should have combined F1 and F2 messages
    f3_message = next(m for m in results if m.recipient == factor3)
    # The values should be F1 + F2 messages, normalized to min=0
    expected_f3 = np.array([1.0, 2.0, 8.0])  # [0.0, 1.0, 7.0]
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
    expected_f1 = np.array([1.0, 0.0, 3.0])   # already normalized
    logger.info(f"EXPECTED F1 message: {expected_f1}")
    logger.info(f"ACTUAL F1 message: {f1_message.data}")
    np.testing.assert_array_almost_equal(f1_message.data, expected_f1,
                                         err_msg=f"F1 message mismatch: expected {expected_f1}, got {f1_message.data}")

    # To F2: should contain message from F1
    f2_message = next(m for m in results if m.recipient == factor2)
    expected_f2 = np.array([0.0, 2.0, -1.0])   # [1.0, 3.0, 0.0]
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

def test_bidirectional_message_passing():
    """Test bidirectional message passing between factor and variable nodes"""
    logger.info("Starting test_bidirectional_message_passing")

    # Create mock nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")
    var3 = MockNode("X3")

    # Create a 3D cost table for the factor (3x3x3)
    cost_table = np.zeros((3, 3, 3))
    # Fill with some pattern
    for i in range(3):
        for j in range(3):
            for k in range(3):
                cost_table[i, j, k] = (i + 2*j + 3*k) % 10

    logger.debug(f"Created 3D cost table with shape {cost_table.shape}")

    # STEP 1: Variable to factor messages (initialization)
    var_to_factor_messages = [
        Message(np.array([0.0, 1.0, 2.0]), sender=var1, recipient=factor_node),
        Message(np.array([2.0, 0.0, 1.0]), sender=var2, recipient=factor_node),
        Message(np.array([1.0, 2.0, 0.0]), sender=var3, recipient=factor_node)
    ]

    logger.info("STEP 1: Computing initial factor-to-variable messages")

    # Initialize min-sum computator
    min_sum = MinSumComputator()

    # Compute factor-to-variable messages
    factor_to_var_messages = min_sum.compute_R(cost_table, var_to_factor_messages)

    # Should be 3 outgoing messages from factor
    assert len(factor_to_var_messages) == 3, f"Expected 3 messages but got {len(factor_to_var_messages)}"

    # Verify each message has the correct sender, recipient, and shape
    for msg in factor_to_var_messages:
        assert msg.sender == factor_node, "Message sender should be factor node"
        assert msg.recipient in [var1, var2, var3], "Message recipient should be a variable node"
        assert msg.data.shape == (3,), f"Message should have shape (3,), got {msg.data.shape}"

    # STEP 2: Send these messages back to variables and compute responses
    logger.info("STEP 2: Computing variable-to-factor messages")

    # Variable nodes will process these messages and send responses
    var_to_factor_messages_2 = min_sum.compute_Q(factor_to_var_messages)

    # Should be 3 outgoing messages from variables
    assert len(var_to_factor_messages_2) == 3, f"Expected 3 messages but got {len(var_to_factor_messages_2)}"

    # Verify each message has the correct sender, recipient, and shape
    for msg in var_to_factor_messages_2:
        assert msg.recipient == factor_node, "Message recipient should be factor node"
        assert msg.sender in [var1, var2, var3], "Message sender should be a variable node"
        assert msg.data.shape == (3,), f"Message should have shape (3,), got {msg.data.shape}"

    # STEP 3: Factor processes the updated messages
    logger.info("STEP 3: Computing updated factor-to-variable messages")

    # Compute new factor-to-variable messages based on updated var-to-factor messages
    factor_to_var_messages_2 = min_sum.compute_R(cost_table, var_to_factor_messages_2)

    # Verify messages have changed from the first round
    for i, (msg1, msg2) in enumerate(zip(factor_to_var_messages, factor_to_var_messages_2)):
        assert not np.array_equal(msg1.data, msg2.data), f"Message {i} should have changed after iteration"

    logger.info("test_bidirectional_message_passing completed successfully")

def test_message_content_verification():
    """Test the actual content of messages passed between nodes"""
    logger.info("Starting test_message_content_verification")

    # Create mock nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")

    # Create a simple 2x2 cost table
    cost_table = np.array([
        [1.0, 3.0],  # costs when X1=0, X2={0,1}
        [4.0, 2.0]   # costs when X1=1, X2={0,1}
    ])

    logger.info(f"Cost table:\n{cost_table}")

    # ---- Step 1: Initial variable-to-factor messages ----
    var_to_factor_messages = [
        Message(np.array([0.0, 0.0]), sender=var1, recipient=factor_node),  # Initial uniform belief
        Message(np.array([0.0, 0.0]), sender=var2, recipient=factor_node)   # Initial uniform belief
    ]

    logger.info("Computing factor-to-variable messages (Round 1)")
    min_sum = MinSumComputator()
    factor_to_var_messages = min_sum.compute_R(cost_table, var_to_factor_messages)

    # Check the actual message content for X1
    x1_message = next(m for m in factor_to_var_messages if m.recipient == var1)
    # When X1=0, min cost over X2 is 1.0; when X1=1, min cost is 2.0
    # Without normalizing, expected message is [1.0, 2.0]
    expected_x1_message = np.array([1.0, 2.0])
    logger.info(f"Expected X1 message: {expected_x1_message}")
    logger.info(f"Actual X1 message: {x1_message.data}")
    np.testing.assert_array_almost_equal(x1_message.data, expected_x1_message)

    # Check the actual message content for X2
    x2_message = next(m for m in factor_to_var_messages if m.recipient == var2)
    # Without normalizing, expected message is [1.0, 2.0]
    expected_x2_message = np.array([1.0, 2.0])
    logger.info(f"Expected X2 message: {expected_x2_message}")
    logger.info(f"Actual X2 message: {x2_message.data}")
    np.testing.assert_array_almost_equal(x2_message.data, expected_x2_message)

    # ---- Step 2: Variable processes factor messages and sends back ----
    logger.info("Computing variable-to-factor messages (Round 2)")
    var_to_factor_messages_2 = min_sum.compute_Q(factor_to_var_messages)

    # Each variable only received one message, so should just pass it back
    # but with appropriate sender/recipient
    for msg in var_to_factor_messages_2:
        assert msg.recipient == factor_node
        if msg.sender == var1:
            np.testing.assert_array_almost_equal(msg.data, expected_x2_message)
            logger.info(f"X1 sending message to factor: {msg.data}")
        elif msg.sender == var2:
            np.testing.assert_array_almost_equal(msg.data, expected_x1_message)
            logger.info(f"X2 sending message to factor: {msg.data}")

    # ---- Step 3: Factor processes the updated variable messages ----
    logger.info("Computing factor-to-variable messages (Round 3)")
    factor_to_var_messages_3 = min_sum.compute_R(cost_table, var_to_factor_messages_2)

    # Now verify the updated messages
    # For X1, combine cost_table with X2's message: [1.0, 2.0]
    # Updated costs: [[2.0, 5.0], [5.0, 4.0]]
    # Min over X2: [2.0, 4.0], without normalization
    x1_message_3 = next(m for m in factor_to_var_messages_3 if m.recipient == var1)
    expected_x1_message_3 = np.array([2.0, 4.0])
    logger.info(f"Expected X1 message (Round 3): {expected_x1_message_3}")
    logger.info(f"Actual X1 message (Round 3): {x1_message_3.data}")
    np.testing.assert_array_almost_equal(x1_message_3.data, expected_x1_message_3)

    logger.info("test_message_content_verification completed successfully")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Running tests directly")
    pytest.main(["-xvs", "--log-cli-level=INFO", __file__])

def test_explicit_connection_numbers():
    """Test compute_R with explicitly defined connection numbers"""
    # Create mock variable and factor nodes
    factor_node = MockNode("F")
    var1 = MockNode("X1")
    var2 = MockNode("X2")

    # Explicitly define connection numbers - intentionally NOT in message order
    factor_node.connection_numbers = {
        var2: 0,  # X2 gets dimension 0
        var1: 1   # X1 gets dimension 1
    }

    # Create a cost table - shape matches our dimension ordering (X2, X1)
    cost_table = np.array([
        [5.0, 2.0, 8.0],  # costs when X2=0, X1={0,1,2}
        [1.0, 3.0, 4.0],  # costs when X2=1, X1={0,1,2}
        [7.0, 0.0, 6.0]   # costs when X2=2, X1={0,1,2}
    ])

    # Create messages in a DIFFERENT order than the connection numbers
    incoming_messages = [
        Message(np.array([0.0, 1.0, 3.0]), sender=var1, recipient=factor_node),
        Message(np.array([2.0, 0.0, 4.0]), sender=var2, recipient=factor_node)
    ]

    # Initialize computator
    computator = MinSumComputator()

    # Compute messages
    results = computator.compute_R(cost_table, incoming_messages)

    # Verify the dimension ordering was respected
    # (The results will be different if dimensions were auto-assigned)
    x1_message = next(m for m in results if m.recipient == var1)
    x2_message = next(m for m in results if m.recipient == var2)

    # Calculate expected with explicit dimension handling
    x1_costs = cost_table + incoming_messages[1].data.reshape(3, 1)
    expected_x1 = np.min(x1_costs, axis=0)

    x2_costs = cost_table + incoming_messages[0].data.reshape(1, 3)
    expected_x2 = np.min(x2_costs, axis=1)

    np.testing.assert_array_almost_equal(x1_message.data, expected_x1)
    np.testing.assert_array_almost_equal(x2_message.data, expected_x2)