import numpy as np

from bp_base.agents import FactorAgent, VariableAgent
from bp_base.components import Message
from typing import Tuple, List, Iterable

from bp_base.factor_graph import FactorGraph


def TD(var_a: Iterable[VariableAgent], x: float):
    for variable in var_a:
        if isinstance(variable, VariableAgent):
            for messages in sorted(
                zip(variable.last_iteration, variable.mailer.inbox),
                key=lambda y: y[0].sender.name,
            ):
                messages[1].data = (1 - x) * messages[0].data + x * messages[1].data


def damp(variable: VariableAgent, x: float) -> None:
    """
    Apply damping to the messages of the variable agents.
    For each message in the outbox, update as:
    new_message = (sum of messages from factor to variable) * (1-x) + x * (corresponding variable message in the inbox)
    :param var_a: Iterable of variable agents.
    :param x: Damping factor.
    """
    pass
