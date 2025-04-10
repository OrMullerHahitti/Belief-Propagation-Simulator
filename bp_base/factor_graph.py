# implementation of factor graph given everything in bp_base
from __future__ import annotations

from abc import ABC
from typing import List, Dict, Tuple, Union, TypeAlias

import numpy as np
from networkx import Graph,bipartite,diameter

from bp_base.agents import VariableAgent, FactorAgent
from DCOP_base import Agent

Edges : TypeAlias = Dict[FactorAgent, List[VariableAgent]]
Edges.__doc__ = "Edges is a dictionary where keys are FactorAgent instances and values are lists of VariableAgent instances. This represents the edges in the factor graph, connecting factors to their corresponding variables."

class FactorGraph:
    def __init__(self, variable_li: List[VariableAgent], factor_li: List[FactorAgent],edges: Edges) -> None:
        self.G = Graph(type ="Factor")  # Use composition instead of inheritance

        if not variable_li and not factor_li:
            raise ValueError("Variable and factor lists cannot both be empty.")


        self.G.add_nodes_from(variable_li, bipartite=0)  # Add variable nodes to the graph
        self.G.add_nodes_from(factor_li, bipartite=1)  # Add factor nodes to the graph
        self.edges = edges

        #initialize the graph with the edges , and the mailboxes of the nodes
        self._add_edges()
        self._initialize_mailbox()
        self._initialize_cost_table()
        for node in bipartite.sets(self.G)[1]:
            if isinstance(node, FactorAgent):
                node.set_name_for_factor()
        self.diameter = diameter(self.G)


    def _add_edges(self):
        """Add edges to the graph.

        :param variable: Variable node
        :param factor: Factor node
        add edges from variable to factor and vice versa and addding the "dom" to both of them
        """
        for factor in self.edges:
            for i,variable in enumerate(self.edges[factor]):
                self.G.add_edge(factor, variable, dim = i)
                factor.set_dim_for_variable(variable, i)
    def step(self):
        """Run the factor graph algorithm."""

        #compute messages to send and put them in the mailbox
        for agent in self.G.nodes():
            agent.messages_to_send = agent.compute_messages(agent.mailbox)
            agent.empty_mailbox()
        #send the messages to the right nodes
        for agent in self.G.nodes():
            for message in agent.messages_to_send:
                message.sender.send_message(message.recipient, message)
    def cycle(self):
        for i in range(self.diameter):
            self.step()
    def run(self, max_iter: int = 1000) -> None:
        """
        Run the factor graph algorithm for a maximum number of iterations.
        :param max_iter: Maximum number of iterations to run.
        """
        for i in range(max_iter):
            self.cycle()
            if self.is_converged():
                break





    def _initialize_mailbox(self):
        """Initialize the mailbox for each Agent."""
        for edge in self.G.edges():
            factor, variable = edge
            factor.recieve_message(data=np.zeros(factor.domain),sender=variable, recipient=factor)
            variable.recieve_message(data=np.zeros(variable.domain),sender=factor, recipient=variable)
    def _initialize_cost_table(self):
        """Initialize the cost table for each FactorAgent."""
        for factor in bipartite.sets(self.G)[1]:
            if isinstance(factor, FactorAgent):
                factor.initiate_cost_table()
            else:
                raise TypeError("Node must be a FactorAgent.")


    def __str__(self):
        return f"FactorGraph: {self.G.nodes()}"


    def __repr__(self):
        return self.__str__()








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