from typing import TypeAlias, Tuple, List

from bp_base.agents import VariableAgent, FactorAgent
from abc import ABC, abstractmethod
EdgeList:TypeAlias = List[Tuple[VariableAgent, FactorAgent]]
class Edges:
    def __init__(self, variable, factor):
        self.variable = variable
        self.factor = factor
        self.dom = 0

    def add_edges(self):
        """Add edges to the graph.

        :param variable: Variable node
        :param factor: Factor node
        add edges from variable to factor and vice versa and adding the "dom" to both of them
        """
        self._add_domains(self.variable, self.factor)  # adding the domains and updating the dom to be +1 for the next factor