from bp_base.agents import FactorAgent, VariableAgent
from bp_base.components import Message
from typing import Tuple, List, Iterable

from bp_base.factor_graph import FactorGraph


def damp(var_a: Iterable[VariableAgent], x: float):
    for variable in var_a:
        if isinstance(variable, VariableAgent):
            for messages in sorted(
                zip(variable.last_iteration, variable.mailer.outbox),
                key=lambda y: y[0].recipient,
            ):
                messages[1].data = (1 - x) * messages[0].data + x * messages[1].data
