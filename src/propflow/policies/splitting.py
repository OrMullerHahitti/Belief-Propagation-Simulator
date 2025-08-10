# utils/factor_split.py
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
    """
    In-place replacement of every FactorAgent by two clones with
    cost-tables p*C  and (1-p)*C, names f'  and f''.
    NetworkX graph, FactorGraph.factors list, and connection maps
    are all updated.
    """
    # Use centralized default if not provided
    if p is None:
        p = POLICY_DEFAULTS["split_factor"]

    assert 0.0 < p < 1.0, "p must be in (0,1)"
    G: nx.Graph = fg.G

    # Work on a *copy* of the factors list to avoid mutation issues
    original_factors: List[FactorAgent] = list(fg.factors)

    for f in original_factors:
        #  build new agents
        cost1 = p * f.cost_table
        cost2 = (1.0 - p) * f.cost_table

        f1 = f.create_from_cost_table(cost_table=cost1, name=f"{f.name}'")
        f2 = f.create_from_cost_table(cost_table=cost2, name=f"{f.name}''")

        # copy dimension-mapping so message axes stay aligned
        f1.connection_number = deepcopy(f.connection_number)
        f2.connection_number = deepcopy(f.connection_number)

        #  add nodes and replicate edges + edge attributes
        for v, edge_data in G[f].items():
            G.add_edge(f1, v, **edge_data)
            G.add_edge(f2, v, **edge_data)

        # register in FactorGraph
        fg.factors.append(f1)
        fg.factors.append(f2)

        #  remove old node and its reference
        G.remove_node(f)
        fg.factors.remove(f)
