"""Message Damping Policies for Belief Propagation.

This module provides functions that implement message damping, a technique used
to stabilize belief propagation by preventing oscillations. Damping works by
blending a newly computed message with the message from a previous iteration.
"""

from typing import List

from ..configs.global_config_mapping import PolicyDefaults
from ..core.agents import VariableAgent


def _apply_damping(outbox, last_iter, x: float) -> None:
    """Blends outbox messages with previous messages using damping factor x."""
    if not last_iter or not outbox:
        return
    last_msg_map = {msg.recipient.name: msg for msg in last_iter}
    for msg in outbox:
        last_msg = last_msg_map.get(msg.recipient.name)
        if last_msg is not None:
            msg.data = x * last_msg.data + (1 - x) * msg.data


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
        x = PolicyDefaults().damping_factor
        if x is None:
            raise ValueError("Damping factor is None")
    if diameter is None:
        diameter = PolicyDefaults().damping_diameter
        if diameter is None:
            raise ValueError("Damping diameter is None")

    for variable in variables:
        _apply_damping(variable.mailer.outbox, variable.last_cycle(diameter), x)


def damp(agent, x: float = None) -> None:
    """Applies damping to outgoing messages of a variable or factor agent.

    Blends each outgoing message with the corresponding message from the
    previous iteration: ``new = x * prev + (1 - x) * current``.

    Args:
        agent: The agent (variable or factor) whose outbox messages will be damped.
        x: The damping factor (weight of previous message). Defaults to PolicyDefaults.
    """
    if x is None:
        x = PolicyDefaults().damping_factor
    _apply_damping(agent.mailer.outbox, agent.last_iteration, x)


# backwards-compatible alias
damp_factor = damp
