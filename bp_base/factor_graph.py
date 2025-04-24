import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

from DCOP_base import Computator
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.computators import BPComputator

logger = logging.getLogger(__name__)

class FactorGraph:
    """
    Represents a factor graph for belief propagation.
    The graph structure is bipartite, with variable nodes connected to factor nodes.
    """

    def __init__(self, variable_li: List[VariableAgent], factor_li: List[FactorAgent], 
                 edges: Dict[FactorAgent, List[VariableAgent]]):
        """
        Initialize the factor graph with variable nodes, factor nodes, and edges.
        
        :param variable_li: List of variable agents
        :param factor_li: List of factor agents
        :param edges: Dict mapping factor agents to their connected variable agents
        """
        self.variables = variable_li
        self.factors = factor_li

        # Create a bipartite graph
        self.G = nx.Graph()
        
        # Add nodes
        self.G.add_nodes_from(self.variables)
        self.G.add_nodes_from(self.factors)
        
        # Add edges and set up factor nodes
        self.add_edges(edges)
        
        # Initialize cost tables for factor nodes
        self.initialize_cost_tables()
        
        # Initialize mailboxes for all nodes
        self.initialize_messages()
        
    def add_edges(self, edges: Dict[FactorAgent, List[VariableAgent]]) -> None:
        """
        Add edges between factor nodes and variable nodes.
        
        :param edges: Dictionary mapping factor nodes to lists of variable nodes
        """
        for factor, variables in edges.items():
            # Ensure connection_number exists
            if not hasattr(factor, "connection_number"):
                factor.connection_number = {}
            for i, var in enumerate(variables):
                self.G.add_edge(factor, var, dim=i)
                # Set dimension index for the variable in the factor's cost table
                factor.connection_number[var] = i
    
    def initialize_cost_tables(self) -> None:
        """
        Initialize cost tables for factor nodes.
        """
        for node in self.G.nodes():
            if isinstance(node, FactorAgent):
                node.initiate_cost_table()
    
    def initialize_messages(self) -> None:
        """
        Initialize mailboxes for all nodes with zero messages.
        Each node creates outgoing messages to all its neighbors.
        """
        # First ensure all nodes have empty mailboxes
        for node in self.G.nodes():
            if not hasattr(node, 'mailbox'):
                node.mailbox = []
        
        # For each node, create outgoing messages to all its neighbors
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            for neighbor in neighbors:
                # Check if neighbor has a domain attribute
                logger.info("Initializing mailbox for node: %s", node)
                zero_data = np.zeros(neighbor.domain)
                message = Message(
                    data=zero_data,
                    sender=node,
                    recipient=neighbor
                )
                node.messages_to_send.append(message)
                  # Initialize messages to send

    def set_computator(self, computator: Computator,**kwargs) -> None:
        """
        Set the computator for all nodes in the graph.

        :param computator: The computator to be set
        """
        for node in self.G.nodes():
            node.computator = computator

    def __getstate__(self):
        """
        Custom method to control what gets pickled.
        This helps ensure compatibility when unpickling.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Custom method to control unpickling behavior.
        """
        # Update the object's state with what was in the pickle
        self.__dict__.update(state)

        # Make sure G is reconstructed if it's missing
        if not hasattr(self, 'G') or self.G is None:
            import networkx as nx
            self.G = nx.Graph()

            # Rebuild graph from variables and factors
            if hasattr(self, 'variables') and hasattr(self, 'factors'):
                # Add nodes
                self.G.add_nodes_from(self.variables)
                self.G.add_nodes_from(self.factors)

                # Rebuild edges from connection_number info
                for factor in self.factors:
                    if hasattr(factor, 'connection_number'):
                        for var, dim in factor.connection_number.items():
                            self.G.add_edge(factor, var, dim=dim)
