import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message

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
        # Create a bipartite graph
        self.G = nx.Graph()
        
        # Add nodes
        self.G.add_nodes_from(variable_li)
        self.G.add_nodes_from(factor_li)
        
        # Add edges and set up factor nodes
        self.add_edges(edges)
        
        # Initialize cost tables for factor nodes
        self.initialize_cost_tables()
        
        # Initialize mailboxes for all nodes
        self.initialize_mailbox()
        
    def add_edges(self, edges: Dict[FactorAgent, List[VariableAgent]]) -> None:
        """
        Add edges between factor nodes and variable nodes.
        
        :param edges: Dictionary mapping factor nodes to lists of variable nodes
        """
        for factor, variables in edges.items():
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
    
    def initialize_mailbox(self) -> None:
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
            # Get all neighbors of the node
            neighbors = list(self.G.neighbors(node))
            
            # For each neighbor, create a message from this node
            for neighbor in neighbors:
                # Create a zero message with appropriate domain size
                zero_data = np.zeros(neighbor.domain)
                
                # Create message with this node as sender and neighbor as recipient
                message = Message(
                    data=zero_data,
                    sender=node,
                    recipient=neighbor
                )
                
                # Add the message to this node's mailbox
                node.mailbox.append(message)
