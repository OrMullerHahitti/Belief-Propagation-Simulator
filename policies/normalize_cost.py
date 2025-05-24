from typing import List

from bp_base.agents import FactorAgent
from bp_base.factor_graph import FactorGraph
import networkx as nx

def init_normalization(li: List[FactorAgent]):
    x = len(li)
    for factor in li:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table / x


def normalize_after_cycle(fg:FactorGraph):
    variables = fg.G.variables
    normalized_weights = {node: 1.0 / fg.G.degree(node) if fg.G.degree(node) > 0 else 0
                          for node in variables}
    for node, weight in normalized_weights.items():
        for message in node.inbox:
            message.data = message.data * weight

