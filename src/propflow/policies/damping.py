"""Message Damping Policies for Belief Propagation.

This module provides functions that implement message damping, a technique used
to stabilize belief propagation by preventing oscillations. Damping works by
blending a newly computed message with the message from a previous iteration.
"""
from ..core.agents import VariableAgent
from ..configs.global_config_mapping import POLICY_DEFAULTS
from typing import List


def TD(variables: List[VariableAgent], x: float = None, diameter: int = None) -> None:
    """Applies temporal damping to the outgoing messages of a list of variables.

    This function applies damping using messages from a previous cycle, determined
    by the `diameter`. The new message is a weighted average of the message
    from `diameter` iterations ago and the current message.

    The update rule is:
    `new_message = x * previous_cycle_message + (1 - x) * current_message`

    Args:
        variables: A list of `VariableAgent` objects to apply damping to.
        x: The damping factor, representing the weight of the previous message.
            If None, the default from `POLICY_DEFAULTS` is used.
        diameter: The number of iterations in a cycle, used to retrieve the
            message from the previous cycle. If None, the default from
            `POLICY_DEFAULTS` is used.
    """
    if x is None:
        x = POLICY_DEFAULTS["damping_factor"]
    if diameter is None:
        diameter = POLICY_DEFAULTS["damping_diameter"]

    for variable in variables:
        last_iter = variable.last_cycle(diameter)
        outbox = variable.mailer.outbox
        if not last_iter or not outbox:
            continue
        last_msg_map = {msg.recipient.name: msg for msg in last_iter}
        for msg in outbox:
            last_msg = last_msg_map.get(msg.recipient.name)
            if last_msg is not None:
                msg.data = x * last_msg.data + (1 - x) * msg.data


def damp(variable: VariableAgent, x: float = None) -> None:
    """Applies damping to the outgoing messages of a single variable agent.

    This function updates each outgoing message in the variable's outbox by
    blending it with the corresponding message from the previous iteration.

    The update rule is:
    `new_message = x * previous_iteration_message + (1 - x) * current_message`

    Args:
        variable: The `VariableAgent` whose outbox messages will be damped.
        x: The damping factor, representing the weight of the previous message.
            If None, the default from `POLICY_DEFAULTS` is used.
    """
    if x is None:
        x = POLICY_DEFAULTS["damping_factor"]

    last_iter = variable.last_iteration
    outbox = variable.mailer.outbox
    if not last_iter or not outbox:
        return
    last_msg_map = {msg.recipient.name: msg for msg in last_iter}
    for msg in outbox:
        last_msg = last_msg_map.get(msg.recipient.name)
        if last_msg is not None:
            msg.data = x * last_msg.data + (1 - x) * msg.data
