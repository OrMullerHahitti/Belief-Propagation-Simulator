from ..core.components import Message
from typing import List


def multiply_messages(messages: List[Message], factor: int):
    """
    Multiply the data in each message by a given factor.

    :param messages: List of Message objects.
    :param factor: The factor by which to multiply the message data.
    """
    for message in messages:
        message.data *= factor


def multiply_messages_attentive(
    messages: List[Message], factor: int | float, iteration: int = 0
):
    """
    Multiply the data in each message by a given factor, but only for messages
    that are not sent by the specified agent.

    :param messages: List of Message objects.
    :param factor: The factor by which to multiply the message data.
    :param iteration: The iteration number.
    """
    for message in messages:
        message.data *= factor
