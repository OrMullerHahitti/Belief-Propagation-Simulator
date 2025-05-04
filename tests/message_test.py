import numpy as np
from unittest.mock import Mock
from DCOP_base import Agent
from DCOP_base import Message


def test_message_initialization():
    sender = Mock(spec=Agent)
    recipient = Mock(spec=Agent)
    message_data = np.array([1, 2, 3])
    message = Message(message=message_data, sender=sender, recipient=recipient)
    assert np.array_equal(message.data, message_data)
    assert message.sender == sender
    assert message.recipient == recipient


def test_message_hash():
    sender = Mock(spec=Agent)
    recipient = Mock(spec=Agent)
    message_data = np.array([1, 2, 3])
    message = Message(message=message_data, sender=sender, recipient=recipient)
    assert hash(message) == hash((sender, recipient))


def test_message_equality():
    sender = Mock(spec=Agent)
    recipient = Mock(spec=Agent)
    message_data1 = np.array([1, 2, 3])
    message_data2 = np.array([4, 5, 6])
    message1 = Message(message=message_data1, sender=sender, recipient=recipient)
    message2 = Message(message=message_data2, sender=sender, recipient=recipient)
    assert message1 == message2


def test_message_inequality_different_sender():
    sender1 = Mock(spec=Agent)
    sender2 = Mock(spec=Agent)
    recipient = Mock(spec=Agent)
    message_data = np.array([1, 2, 3])
    message1 = Message(message=message_data, sender=sender1, recipient=recipient)
    message2 = Message(message=message_data, sender=sender2, recipient=recipient)
    assert message1 != message2


def test_message_inequality_different_recipient():
    sender = Mock(spec=Agent)
    recipient1 = Mock(spec=Agent)
    recipient2 = Mock(spec=Agent)
    message_data = np.array([1, 2, 3])
    message1 = Message(message=message_data, sender=sender, recipient=recipient1)
    message2 = Message(message=message_data, sender=sender, recipient=recipient2)
    assert message1 != message2
