# implementation of factor graph given everything in bp_base
from abc import ABC
from typing import List, Dict, Tuple, Union


from networkx import Graph,bipartite

from bp_base.agents import VariableAgent, FactorAgent
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





class FactorGraph:
    def __init__(self, variable_li: List[VariableAgent], factor_li: List[FactorAgent]):
        self.g = Graph()  # Use composition instead of inheritance

        if not variable_li and not factor_li:
            raise ValueError("Variable and factor lists cannot both be empty.")


        self.g.add_nodes_from(variable_li, bipartite=0)  # Add variable nodes to the graph
        self.g.add_nodes_from(factor_li, bipartite=1)  # Add factor nodes to the graph
        self.dom =0



    def add_edge(self, variable: VariableAgent, factor: FactorAgent) -> None:
        """Add edges to the graph.

        :param variable: Variable node
        :param factor: Factor node
        add edges from variable to factor and vice versa and addding the "dom" to both of them
        """
        self._add_domains(variable, factor)  # adding the domains and updating the dom to be +1 for the next factor
        self.g.add_edge(variable, factor)
        self.g.add_edge(factor, variable)

    def __str__(self):
        return f"FactorGraph: {self.g.nodes()}"


    def __repr__(self):
        return self.__str__()

    def _add_domains(self, variable, factor):
        """Add domains to the factor node."""
        variable.add_domain(factor, self.dom)
        factor.add_domain(variable,self.dom)
        self.dom += 1






if __name__ == "__main__":
    # The following code should run without error
    # Create variable nodes
    var1 = VariableAgent(name="v1")
    var2 = VariableAgent(name="v2")
    var3 = VariableAgent(name="v3")

    # Create factor nodes
    factor12 = FactorAgent(name="f12", cost_table=[[0.1, 0.9], [0.8, 0.2]])
    factor23 = FactorAgent(name="f23", cost_table=[[0.2, 0.8], [0.7, 0.3]])

    # Create factor graph
    fg = FactorGraph(variable_li=[var1, var2, var3], factor_li=[factor12, factor23])
    print(fg.graph.nodes())
    print(fg.graph.edges())
    # Output:
    # ['v1', 'v2', 'v3', 'f12', 'f23']
    # [('f12', 'v1'), ('f12', 'v2'), ('f23', 'v2'), ('f23', 'v3')]