"""Policies for Cost and Message Reduction/Discounting.

This module provides a collection of functions that implement different
strategies for modifying costs within a factor graph. These strategies can be
used as policies within a belief propagation engine to influence the
algorithm's behavior, such as improving convergence or exploring different
solution spaces.
"""
from ..core.agents import FactorAgent
from typing import Iterable

from ..bp.factor_graph import FactorGraph


def cost_reduction_all_factors_once(fg: FactorGraph, x: float) -> None:
    """Applies a one-time cost reduction to all factors in the graph.

    This function iterates through all factor agents in the provided factor
    graph and multiplies their cost tables by a given factor. It also saves
    the original cost table before modification.

    Args:
        fg: The `FactorGraph` to modify.
        x: The multiplicative factor to apply to the cost tables.
    """
    for factor in fg.factors:
        if factor.cost_table is not None:
            factor.save_original()
            factor.cost_table = factor.cost_table * x


def discount(fac_a: Iterable[FactorAgent], x: float) -> None:
    """Applies a discount factor to the cost tables of specified factor agents.

    Args:
        fac_a: An iterable of `FactorAgent` objects whose cost tables will be discounted.
        x: The multiplicative discount factor to apply.
    """
    for factor in fac_a:
        if factor.cost_table is not None:
            factor.save_original()
            factor.cost_table = factor.cost_table * x


def discount_attentive(fg: FactorGraph) -> None:
    """Applies an attentive discount to incoming messages for all variable nodes.

    The discount factor for each variable's messages is inversely proportional
    to the variable's degree in the factor graph. This means variables with more
    connections (higher degree) will have their incoming messages discounted more heavily.

    Args:
        fg: The `FactorGraph` containing the variables to be updated.
    """
    variables = {n for n, d in fg.G.nodes(data=True) if d.get("bipartite") == 0}

    normalized_weights = {
        node: 1.0 / fg.G.degree(node) if fg.G.degree(node) > 0 else 0
        for node in variables
    }
    for node, weight in normalized_weights.items():
        for message in node.inbox:
            message.data = message.data * weight


def reduce_R(fac_a: FactorAgent, x: float) -> None:
    """Reduces the outgoing R-messages from a factor agent by a factor `x`.

    This function iterates through all staged outgoing messages (R-messages)
    of a given factor agent and multiplies their data by a reduction factor.

    Args:
        fac_a: The `FactorAgent` whose outgoing messages will be reduced.
        x: The factor by which to reduce the message data.
    """
    for messages in fac_a.outbox.values():
        for message in messages:
            message.data *= x
