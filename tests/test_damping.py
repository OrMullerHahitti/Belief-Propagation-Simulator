import numpy as np
from bp_base.agents import VariableAgent
from bp_base.components import Message
from policies.damping import damp, TD


class MockAgent:
    def __init__(self, name):
        self.name = name


def test_damp():
    """Test that damp correctly applies damping to messages."""
    # Create variable agents with messages
    var1 = VariableAgent(name="var1", domain=2)
    var2 = VariableAgent(name="var2", domain=2)

    # Create senders and recipients for messages
    sender1 = MockAgent("sender1")
    sender2 = MockAgent("sender2")
    recipient1 = MockAgent("recipient1")
    recipient2 = MockAgent("recipient2")

    # Create previous iteration messages
    prev_msg1 = Message(sender=sender1, recipient=recipient1, data=np.array([1.0, 2.0]))
    prev_msg2 = Message(sender=sender2, recipient=recipient2, data=np.array([3.0, 4.0]))

    # Create current outbox messages
    curr_msg1 = Message(sender=sender1, recipient=recipient1, data=np.array([5.0, 6.0]))
    curr_msg2 = Message(sender=sender2, recipient=recipient2, data=np.array([7.0, 8.0]))

    # Set up variable agents
    var1._history = [[prev_msg1, prev_msg2]]  # Set _history directly
    var1.mailer.outbox = [curr_msg1, curr_msg2]

    # Apply damping
    damping_factor = 0.5
    damp(var1, damping_factor)

    # Check that messages are correctly damped
    # New message = (1-x) * old_message + x * new_message
    expected_msg1_data = (
        1 - damping_factor
    ) * prev_msg1.data + damping_factor * np.array([5.0, 6.0])
    expected_msg2_data = (
        1 - damping_factor
    ) * prev_msg2.data + damping_factor * np.array([7.0, 8.0])

    np.testing.assert_array_almost_equal(curr_msg1.data, expected_msg1_data)
    np.testing.assert_array_almost_equal(curr_msg2.data, expected_msg2_data)


def test_damp_extreme_values():
    """Test damping with extreme values (0 and 1)."""
    # Create variable agent with messages
    var = VariableAgent(name="var", domain=2)

    # Create sender and recipient for message
    sender = MockAgent("sender")
    recipient = MockAgent("recipient")

    # Create messages
    prev_msg = Message(sender=sender, recipient=recipient, data=np.array([1.0, 2.0]))
    curr_msg = Message(sender=sender, recipient=recipient, data=np.array([3.0, 4.0]))

    # Set up variable agent
    var._history = [[prev_msg]]  # Set _history directly
    var.mailer.outbox = [curr_msg]

    # Test with damping factor = 0 (should keep old message)
    damp(var, 0.0)
    np.testing.assert_array_almost_equal(curr_msg.data, prev_msg.data)

    # Reset message
    curr_msg.data = np.array([3.0, 4.0])

    # Test with damping factor = 1 (should keep new message)
    damp(var, 1.0)
    np.testing.assert_array_almost_equal(curr_msg.data, np.array([3.0, 4.0]))


def test_damp_empty_list():
    """Test damping with an empty list of variables."""
    # This should not raise any errors
    TD([], 0.5)


def test_damp_no_messages():
    """Test damping with a variable that has no messages."""
    var = VariableAgent(name="var", domain=2)
    var._history = []  # Set _history directly
    var.mailer.outbox = []

    # This should not raise any errors
    damp(var, 0.5)
