# implementation of factor graph given everything in bp_base
from abc import ABC
from typing import List, Dict, Tuple, Union

from networkx import Graph

from bp_base.agents import VariableNode, FactorNode
from DCOP_base import Agent


def _strip_name(agent: Agent) -> List[int]:
    """Extract variable indices from factor name (e.g., 'f123' -> [1, 2, 3])."""
    name = agent.name
    if name.startswith('f'):
        try:
            indices = [int(i) for i in name[1:]]
            return indices
        except ValueError:
            raise ValueError(f"Invalid factor name: {name}. Factor names should start with 'f' followed by digits.")
    else:
        raise ValueError(f"Invalid agent name: {name}. Only factor names should be processed by this function.")


def _create_edges(variable_li: List[VariableNode], factor_li: List[FactorNode]) -> Dict[FactorNode, List[VariableNode]]:
    """Create edges based on variable indices in factor names."""
    adjacency = {}
    for f in factor_li:
        try:
            indices = _strip_name(f)
            neighbors = [v for v in variable_li if int(v.name[1:]) in indices]  # Match variable indices
            adjacency[f] = neighbors
        except ValueError as e:
            print(f"Warning: Skipping factor {f.name} due to naming error: {e}")
            adjacency[f] = []  # Ensure every factor has a key in adjacency
    return adjacency


class FactorGraph():
    def __init__(self, variable_li: List[VariableNode], factor_li: List[FactorNode]):
        self.graph = Graph()  # Use composition instead of inheritance

        if not variable_li and not factor_li:
            raise ValueError("Variable and factor lists cannot both be empty.")

        self.graph.add_nodes_from(variable_li)
        self.graph.add_nodes_from(factor_li)

        adjacency = _create_edges(variable_li, factor_li)
        self.graph.add_edges_from((f, v) for f, neighbors in adjacency.items() for v in neighbors)
        def __str__(self):
            return f"FactorGraph: {self.graph.nodes()}"
        def __repr__(self):
            return self.__str__()

if __name__ == "__main__":
    # The following code should run without error
    # Create variable nodes
    var1 = VariableNode(name="v1")
    var2 = VariableNode(name="v2")
    var3 = VariableNode(name="v3")

    # Create factor nodes
    factor12 = FactorNode(name="f12", cost_table=[[0.1, 0.9], [0.8, 0.2]])
    factor23 = FactorNode(name="f23", cost_table=[[0.2, 0.8], [0.7, 0.3]])

    # Create factor graph
    fg = FactorGraph(variable_li=[var1, var2, var3], factor_li=[factor12, factor23])
    print(fg.graph.nodes())
    print(fg.graph.edges())
    # Output:
    # ['v1', 'v2', 'v3', 'f12', 'f23']
    # [('f12', 'v1'), ('f12', 'v2'), ('f23', 'v2'), ('f23', 'v3')]