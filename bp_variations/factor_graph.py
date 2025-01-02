from typing import Dict, List, Union
from nodes import VariableNode, FactorNode  # Assuming Node subclasses are in a `nodes` module
from abstract_graph_system import AbstractGraphSystem  # Base class


class FactorGraph(AbstractGraphSystem):
    """
    Specialized implementation of AbstractGraphSystem for factor graphs.
    Supports variable and factor nodes, along with connecting them.
    """

    def __init__(self):
        self.nodes: Dict[str, Union[VariableNode, FactorNode]] = {}

    def add_agent(self, agent: Union[VariableNode, FactorNode]) -> None:
        """
        Add a VariableNode or FactorNode to the graph.
        """
        if agent.name in self.nodes:
            raise ValueError(f"Agent with name {agent.name} already exists.")
        self.nodes[agent.name] = agent

    def connect(self, agent1_name: str, agent2_name: str) -> None:
        """
        Connect two nodes (VariableNode or FactorNode) in the graph.
        """
        if agent1_name not in self.nodes or agent2_name not in self.nodes:
            raise ValueError("Both agents must exist in the graph.")

        agent1 = self.nodes[agent1_name]
        agent2 = self.nodes[agent2_name]
        agent1.add_neighbor(agent2)
        agent2.add_neighbor(agent1)

    def get_agents(self) -> List[Union[VariableNode, FactorNode]]:
        """
        Retrieve all nodes (agents) in the graph.
        """
        return list(self.nodes.values())

    def all_variables(self) -> List[VariableNode]:
        """
        Retrieve all variable nodes in the graph.
        """
        return [node for node in self.nodes.values() if isinstance(node, VariableNode)]

    def all_factors(self) -> List[FactorNode]:
        """
        Retrieve all factor nodes in the graph.
        """
        return [node for node in self.nodes.values() if isinstance(node, FactorNode)]
