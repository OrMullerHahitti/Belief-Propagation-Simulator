from base_all.agents import FactorAgent
from typing import Iterable, List

from bp_base.factor_graph import FactorGraph


def cost_reduction_all_factors_once(fg: FactorGraph, x: float):

    for factor in fg.factors:
        if factor.cost_table is not None:
            factor.save_original()
            factor.cost_table = factor.cost_table * x


def discount(fac_a: Iterable[FactorAgent], x: float):
    for factor in fac_a:
        if factor.cost_table is not None:
            factor.save_original()
            factor.cost_table = factor.cost_table * x


def discount_attentive(fg: FactorGraph):
    variables = {n for n, d in fg.G.nodes(data=True) if d.get("bipartite") == 0}

    normalized_weights = {
        node: 1.0 / fg.G.degree(node) if fg.G.degree(node) > 0 else 0
        for node in variables
    }
    for node, weight in normalized_weights.items():
        for message in node.inbox:
            message.data = message.data * weight
