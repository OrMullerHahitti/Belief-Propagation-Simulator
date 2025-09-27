"""Utility functions for manipulating lists of messages.

This module provides helper functions for performing bulk operations on lists
of `Message` objects, such as modifying their data content.
"""
from ..core.components import Message
from typing import List


def multiply_messages(messages: List[Message], factor: float) -> None:
    """Multiplies the data content of each message in a list by a given factor.

    This function modifies the `data` attribute of each `Message` object in place.

    Args:
        messages: A list of `Message` objects.
        factor: The number by which to multiply each message's data.
    """
    for message in messages:
        message.data *= factor


def multiply_messages_attentive(
    messages: List[Message], factor: float, iteration: int = 0
) -> None:
    """Multiplies the data of each message by a given factor.

    Note:
        The current implementation multiplies all messages in the list, similar
        to `multiply_messages`. The `iteration` parameter is not used. The original
        intent may have been to apply a more selective (attentive) multiplication.

    Args:
        messages: A list of `Message` objects.
        factor: The factor by which to multiply the message data.
        iteration: The current iteration number (currently unused).
    """
    for message in messages:
        message.data *= factor
