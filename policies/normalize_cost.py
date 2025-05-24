from typing import List

from bp_base.agents import FactorAgent, VariableAgent
from bp_base.factor_graph import FactorGraph
import networkx as nx


def init_normalization(li: List[FactorAgent]):
    x = len(li)
    for factor in li:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table / x


def normalize_after_cycle(variables: List[VariableAgent]):
    """
    Normalize the message data of all variables in the factor graph after each cycle.
    """
    for var in variables:
        for message in var.mailer.inbox:
            if message.data is not None:
                # Normalize the message data
                message.data = message.data - message.data.min()
