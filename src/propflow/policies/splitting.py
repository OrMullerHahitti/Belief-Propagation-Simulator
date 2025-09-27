"""A Policy for Splitting Factors in a Factor Graph.

This module provides a function that implements the factor splitting policy.
This technique modifies the structure of the factor graph by replacing each
factor with two new "cloned" factors. The original cost table is distributed
between these two clones. This can be useful for altering the message-passing
dynamics and can sometimes help with convergence or finding better solutions.
"""
from __future__ import annotations
from typing import List
import networkx as nx
from copy import deepcopy

from ..core.agents import FactorAgent
from ..bp.factor_graph import FactorGraph
from ..configs.global_config_mapping import POLICY_DEFAULTS


def split_all_factors(
    fg: FactorGraph,
    p: float = None,
) -> None:
    """Performs an in-place replacement of every factor with two cloned factors.

    Each factor `f` with cost table `C` is replaced by two new factors, `f'`
    and `f''`, with cost tables `p*C` and `(1-p)*C`, respectively. The new
    factors inherit all the connections of the original factor.

    This function directly modifies the provided `FactorGraph` object, including
    its underlying `networkx.Graph` and its list of factors.

    Args:
        fg: The `FactorGraph` to modify.
        p: The splitting proportion, which must be between 0 and 1. This
           determines how the original cost is distributed between the two
           new factors. If None, the default from `POLICY_DEFAULTS` is used.

    Raises:
        AssertionError: If `p` is not in the range (0, 1).
    """
    if p is None:
        p = POLICY_DEFAULTS["split_factor"]

    assert 0.0 < p < 1.0, "p must be in (0,1)"
    G: nx.Graph = fg.G

    # Iterate over a copy of the factors list to avoid mutation issues during iteration.
    original_factors: List[FactorAgent] = list(fg.factors)

    for f in original_factors:
        # Build the two new factor agents with distributed costs.
        cost1 = p * f.cost_table
        cost2 = (1.0 - p) * f.cost_table

        f1 = f.create_from_cost_table(cost_table=cost1, name=f"{f.name}'")
        f2 = f.create_from_cost_table(cost_table=cost2, name=f"{f.name}''")

        # Copy the dimension mapping to ensure message axes stay aligned.
        f1.connection_number = deepcopy(f.connection_number)
        f2.connection_number = deepcopy(f.connection_number)

        # Add the new nodes and replicate the edges of the original factor.
        for v, edge_data in G[f].items():
            G.add_edge(f1, v, **edge_data)
            G.add_edge(f2, v, **edge_data)

        # Register the new factors in the FactorGraph.
        fg.factors.append(f1)
        fg.factors.append(f2)

        # Remove the original factor node and its reference from the graph.
        G.remove_node(f)
        fg.factors.remove(f)
