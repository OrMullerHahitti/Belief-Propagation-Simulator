"""Utility functions for manipulating lists of messages.

This module provides helper functions for performing bulk operations on lists
of `Message` objects, such as modifying their data content.
"""

from typing import List

from ..core.components import Message


def multiply_messages(messages: List[Message], factor: float) -> None:
    """Multiplies the data content of each message in a list by a given factor.

    This function modifies the `data` attribute of each `Message` object in place.

    Args:
        messages: A list of `Message` objects.
        factor: The number by which to multiply each message's data.
    """
    for message in messages:
        message.data *= factor
