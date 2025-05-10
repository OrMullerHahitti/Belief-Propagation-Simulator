from copy import deepcopy

import networkx as nx
import numpy as np
from typing import List, Dict
import logging

from bp_base.DCOP_base import Computator
from bp_base.agents import VariableAgent, FactorAgent

logger = logging.getLogger(__name__)


class FactorGraph:
    """
    Represents a bipartite factor graph for belief propagation.
    The graph structure is bipartite, with variable nodes (set 0) connected only to factor nodes (set 1).
    """

    def __init__(
        self,
        variable_li: List[VariableAgent],
        factor_li: List[FactorAgent],
        edges: Dict[FactorAgent, List[VariableAgent]],
    ):
        """
        Initialize the factor graph with variable nodes, factor nodes, and edges.
        Enforces bipartite structure: variables <-> factors only.

        :param variable_li: List of variable agents
        :param factor_li: List of factor agents
        :param edges: Dict mapping factor agents to their connected variable agents
        """
        self.variables = variable_li
        self.factors = factor_li
        self._original_factors = deepcopy(factor_li)

        # Create a bipartite graph
        self.G = nx.Graph()

        # Add nodes with bipartite attribute
        self.G.add_nodes_from(self.variables, bipartite=0)
        self.G.add_nodes_from(self.factors, bipartite=1)

        # Add edges and set up factor nodes
        self._add_edges(edges)

        # Initialize cost tables for factor nodes
        self._initialize_cost_tables()

        # Initialize mailboxes for all nodes
        self._initialize_messages()

    @property
    def global_cost(self) -> int | float:
        """
        Calculate the global cost of the factor graph at the current state.
        Based on current variable assignments and factor cost tables.
        """
        # Get current assignments for all variables
        var_assignments = {var: var.curr_assignment for var in self.variables}

        total_cost = 0.0
        # For each factor, calculate the cost based on the assignments of connected variables
        for factor in self.factors:
            if factor.cost_table is not None:  # additional check better coverage
                indices = []
                valid_lookup = True  # converage

                # Build the indices list in the right order according to connection_number in the factor
                for var, dim in factor.connection_number.items():
                    if var in var_assignments:
                        # build empty indices in compliance with the cost table
                        while len(indices) <= dim:
                            indices.append(None)
                        indices[dim] = var_assignments[var]
                    else:
                        valid_lookup = False  # coverage
                        break

                # If we have assignments for all connected variables, add the cost
                if valid_lookup and None not in indices:
                    total_cost += factor.cost_table[tuple(indices)]

        return total_cost

    @property
    def curr_assignment(self) -> Dict[VariableAgent, int]:
        """
        Compute the current assignment based on incoming messages.
        :return: Current assignment as a dictionary mapping variable agents to their assignments.
        """
        return {node: node.curr_assignment for node in self.variables}

    def set_computator(self, computator: Computator, **kwargs) -> None:
        """
        Set the computator for all nodes in the graph.

        :param computator: The computator to be set
        """
        for node in self.G.nodes():
            node.computator = computator

    def normalize_messages(self) -> None:
        """
        Normalize the messages in the graph.
        This is a placeholder for normalization logic.
        """
        for node in nx.bipartite.sets(self.G)[0]:
            if isinstance(node, VariableAgent):
                for message in node.mailer.inbox:
                    # Normalize the message
                    message.data -= np.min(message.data)

    def visualize(self) -> None:
        """
        Visualize the factor graph using matplotlib.
        """
        import matplotlib.pyplot as plt

        pos = nx.bipartite_layout(self.G, nodes=self.variables)
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=self.variables,
            node_shape="o",
            node_color="lightblue",
            node_size=300,
        )
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=self.factors,
            node_shape="s",
            node_color="lightgreen",
            node_size=300,
        )
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)
        plt.show()

    def _add_edges(self, edges: Dict[FactorAgent, List[VariableAgent]]) -> None:
        """
        Add edges between factor nodes and variable nodes.
        Enforces bipartite structure: only factor-variable edges allowed.

        :param edges: Dictionary mapping factor nodes to lists of variable nodes
        """
        for factor, variables in edges.items():
            # Ensure connection_number exists
            if not hasattr(factor, "connection_number"):
                factor.connection_number = {}
            for i, var in enumerate(variables):
                # Enforce bipartite: only connect factor <-> variable
                if not (
                    (factor in self.factors and var in self.variables)
                    or (factor in self.variables and var in self.factors)
                ):
                    raise ValueError(
                        "Edges must connect a factor node to a variable node (bipartite structure)."
                    )
                self.G.add_edge(factor, var, dim=i)
                # Set dimension index for the variable in the factor's cost table
                factor.connection_number[var] = i
        logger.info("FactorGraph is bipartite: variables <-> factors only.")

    def _initialize_cost_tables(self) -> None:
        """
        Initialize cost tables for factor nodes.
        """
        for node in list(nx.bipartite.sets(self.G))[1]:
            if isinstance(node, FactorAgent):
                # Initialize cost table for the factor node
                node.initiate_cost_table()
                logger.info("Cost table initialized for factor node: %s", node.name)

    def _initialize_messages(self) -> None:
        """
        Initialize mailboxes for all nodes with zero messages.
        Each node creates outgoing messages to all its neighbors.
        """
        # For each node, create outgoing messages to all its neighbors
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if isinstance(node, VariableAgent):
                for neighbor in neighbors:
                    # Check if neighbor has a domain attribute
                    logger.info("Initializing mailbox for node: %s", node)

                    node.mailer.set_first_message(node, neighbor)
                    # Initialize messages to send

    def get_variable_agents(self) -> List[VariableAgent]:
        """Return a list of all variable agents in the graph."""
        return self.variables

    def get_factor_agents(self) -> List[FactorAgent]:
        """Return a list of all factor agents in the graph."""
        return self.factors

    @property
    def diameter(self) -> int:
        """Return the diameter of the factor graph."""
        if not self.G:
            return 0
        # Check if graph is connected, diameter is infinite for disconnected graphs
        if not nx.is_connected(self.G):
            # Or handle as an error, or return a specific value e.g. -1 or float('inf')
            # For now, returning the diameter of the largest connected component
            if not list(nx.connected_components(self.G)):  # Handle empty graph case
                return 0
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
            if (
                not subgraph.nodes()
            ):  # Handle case where largest_cc is empty or subgraph is empty
                return 0
            return nx.diameter(subgraph)
        return nx.diameter(self.G)

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
        if not hasattr(self, "G") or self.G is None:
            import networkx as nx

            self.G = nx.Graph()

            # Rebuild graph from variables and factors
            if hasattr(self, "variables") and hasattr(self, "factors"):
                # Add nodes
                self.G.add_nodes_from(self.variables, bipartite=0)
                self.G.add_nodes_from(self.factors, bipartite=1)

                # Rebuild edges from connection_number info
                for factor in self.factors:
                    if hasattr(factor, "connection_number"):
                        for var, dim in factor.connection_number.items():
                            self.G.add_edge(factor, var, dim=dim)
