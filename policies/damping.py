import numpy as np

from bp_base.agents import FactorAgent, VariableAgent
from bp_base.components import Message
from typing import Tuple, List, Iterable

from bp_base.factor_graph import FactorGraph


def TD(variables: List[VariableAgent], x: float, diameter: int = 1):
    for variable in variables:
        last_iter = variable.last_cycle(diameter)
        outbox = variable.mailer.outbox
        if not last_iter or not outbox:
            return
        # Create a mapping from recipient name to last message
        last_msg_map = {msg.recipient.name: msg for msg in last_iter}
        for msg in outbox:
            last_msg = last_msg_map.get(msg.recipient.name)
            if last_msg is not None:
                msg.data = x * last_msg.data + (1 - x) * msg.data


def damp(variable: VariableAgent, x: float) -> None:
    """
    Apply damping to the outgoing messages of a variable agent.
    For each message in the outbox, update as:
    new_message = (1-x) * last_message + x * current_message
    :param variable: Variable agent whose outbox will be damped.
    :param x: Damping factor.
    """
    last_iter = variable.last_iteration
    outbox = variable.mailer.outbox
    if not last_iter or not outbox:
        return
    # Create a mapping from recipient name to last message
    last_msg_map = {msg.recipient.name: msg for msg in last_iter}
    for msg in outbox:
        last_msg = last_msg_map.get(msg.recipient.name)
        if last_msg is not None:
            msg.data = x * last_msg.data + (1 - x) * msg.data
