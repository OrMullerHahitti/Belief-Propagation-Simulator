from functools import partial

from .base_models.components import Message
from typing import List
from .base_models.agents import FGAgent


def multiply_messages(messages: List[Message], factor: int):
    """
    Multiply the data in each message by a given factor.

    :param messages: List of Message objects.
    :param factor: The factor by which to multiply the message data.
    """
    for message in messages:
        message.data *= factor


double_messages = partial(multiply_messages, factor=2)
