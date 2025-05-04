from typing import List, Tuple

from bp_base.components import Message

def order_lists_of_messages(
    list1: List[Message], list2: List[Message]
) -> Tuple[List[Message], List[Message]]:
    """
    Order two lists of messages based on the sender and recipient attributes.
    This function assumes that the messages in both lists are from the same sender and recipient pairs.

    :param list1: First list of messages
    :param list2: Second list of messages
    :return: Ordered lists of messages
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")

    # Create a mapping from (sender, recipient) to message
    mapping1 = {(msg.sender, msg.recipient): msg for msg in list1}
    mapping2 = {(msg.sender, msg.recipient): msg for msg in list2}

    # Sort the keys based on sender and recipient
    keys = sorted(mapping1.keys(), key=lambda x: (x[0], x[1]))

    # Create ordered lists based on sorted keys
    ordered_list1 = [mapping1[key] for key in keys]
    ordered_list2 = [mapping2[key] for key in keys]

    return ordered_list1, ordered_list2