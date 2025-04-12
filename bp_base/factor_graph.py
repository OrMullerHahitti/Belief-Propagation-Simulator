import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any

from bp_base.agents import VariableAgent, FactorAgent


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
                self.G.add_edge(factor, var)
                # Set dimension index for the variable in the factor's cost table
                factor.set_dim_for_variable(var, i)
    
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
        """
        for edge in self.G.edges():
            factor, variable = None, None
            
            # Determine which node is the factor and which is the variable
            if isinstance(edge[0], FactorAgent) and isinstance(edge[1], VariableAgent):
                factor, variable = edge[0], edge[1]
            elif isinstance(edge[0], VariableAgent) and isinstance(edge[1], FactorAgent):
                variable, factor = edge[0], edge[1]
            
            # If both nodes are identified, initialize their mailboxes
            if factor is not None and variable is not None:
                # Initialize mailboxes with zeros
                if not hasattr(factor, 'mailbox'):
                    factor.mailbox = []
                if not hasattr(variable, 'mailbox'):
                    variable.mailbox = []
                
                factor.mailbox.append(np.zeros(factor.domain))
                variable.mailbox.append(np.zeros(variable.domain))
