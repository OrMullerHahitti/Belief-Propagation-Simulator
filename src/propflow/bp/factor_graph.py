from copy import deepcopy

import networkx as nx
import numpy as np
from typing import List, Dict
import logging

from ..core.dcop_base import Computator
from ..core.agents import VariableAgent, FactorAgent

logger = logging.getLogger(__name__)


class FactorGraph:
    """Represents a bipartite factor graph for belief propagation.

    This class encapsulates the structure of a factor graph, which consists of
    variable nodes and factor nodes. It enforces a bipartite structure where
    variables are only connected to factors and vice versa. It uses a
    `networkx.Graph` to manage the underlying connections.

    Attributes:
        variables (List[VariableAgent]): A list of all variable agents in the graph.
        factors (List[FactorAgent]): A list of all factor agents in the graph.
        G (nx.Graph): The underlying networkx graph structure.
    """

    def __init__(
        self,
        variable_li: List[VariableAgent],
        factor_li: List[FactorAgent],
        edges: Dict[FactorAgent, List[VariableAgent]],
    ):
        """Initializes the factor graph.

        Args:
            variable_li: A list of `VariableAgent` objects.
            factor_li: A list of `FactorAgent` objects.
            edges: A dictionary mapping each factor agent to a list of the
                variable agents it connects to.
        """
        self.variables = variable_li
        self.factors = factor_li
        self.G = nx.Graph()

        # Add nodes with bipartite attribute
        self.G.add_nodes_from(self.variables, bipartite=0)
        self.G.add_nodes_from(self.factors, bipartite=1)

        self._add_edges(edges)
        self._initialize_cost_tables()
        self._original_factors = deepcopy(factor_li)
        self._lb = None  # Lower bound
        self._ub = None  # Upper bound

    @property
    def lb(self) -> int | float:
        """The lower bound of the problem, can be set externally."""
        return self._lb

    @lb.setter
    def lb(self, value: int | float) -> None:
        """Sets the lower bound of the factor graph.

        Args:
            value: The lower bound value to set.

        Raises:
            ValueError: If the value is not an integer or float.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("Lower bound must be an integer or float.")
        self._lb = value

    @property
    def global_cost(self) -> int | float:
        """Calculates the global cost based on current variable assignments.

        This property queries each variable for its current assignment and uses
        the original, unmodified factor cost tables to compute the total cost.
        """
        var_name_assignments = {var.name: var.curr_assignment for var in self.variables}
        total_cost = 0.0
        for factor in self.factors:
            if factor.cost_table is not None:
                indices = []
                valid_lookup = True
                for var_name, dim in factor.connection_number.items():
                    if var_name in var_name_assignments:
                        while len(indices) <= dim:
                            indices.append(None)
                        indices[dim] = var_name_assignments[var_name]
                    else:
                        valid_lookup = False
                        break
                if valid_lookup and None not in indices:
                    cost_table = factor.original_cost_table if factor.original_cost_table is not None else factor.cost_table
                    total_cost += cost_table[tuple(indices)]
        return total_cost

    @property
    def curr_assignment(self) -> Dict[VariableAgent, int]:
        """dict: The current assignment for all variables in the graph."""
        return {node: node.curr_assignment for node in self.variables}

    @property
    def edges(self) -> Dict[FactorAgent, List[VariableAgent]]:
        """dict: Reconstructs the edge dictionary mapping factors to variables."""
        edge_dict = {}
        var_by_name = {v.name: v for v in self.variables}
        for factor in self.factors:
            if hasattr(factor, 'connection_number'):
                # Sort variables by their dimension index
                vars_with_dims = []
                for var_name, dim in factor.connection_number.items():
                    if var_name in var_by_name:
                        vars_with_dims.append((var_by_name[var_name], dim))
                vars_with_dims.sort(key=lambda x: x[1])
                edge_dict[factor] = [var for var, _ in vars_with_dims]
        return edge_dict

    def set_computator(self, computator: Computator, **kwargs) -> None:
        """Assigns a computator to all nodes in the graph.

        Args:
            computator: The computator instance to assign.
            **kwargs: Additional arguments (not currently used).
        """
        for node in self.G.nodes():
            node.computator = computator

    def normalize_messages(self) -> None:
        """Normalizes all incoming messages for all variable nodes.

        This is a common technique to prevent numerical instability in belief
        propagation algorithms by shifting message values.
        """
        for node in nx.bipartite.sets(self.G)[0]:
            if isinstance(node, VariableAgent):
                for message in node.mailer.inbox:
                    message.data -= np.min(message.data)

    def visualize(self) -> None:
        """Visualizes the factor graph using matplotlib.

        Variable nodes are drawn as circles, and factor nodes are drawn as squares.
        """
        import matplotlib.pyplot as plt

        pos = nx.bipartite_layout(self.G, nodes=self.variables)
        nx.draw_networkx_nodes(
            self.G, pos, nodelist=self.variables, node_shape="o",
            node_color="lightblue", node_size=300
        )
        nx.draw_networkx_nodes(
            self.G, pos, nodelist=self.factors, node_shape="s",
            node_color="lightgreen", node_size=300
        )
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)
        plt.show()

    def _add_edges(self, edges: Dict[FactorAgent, List[VariableAgent]]) -> None:
        """Adds edges and configures factor-variable connections.

        Args:
            edges: A dictionary mapping factors to the variables they connect.
        """
        for factor, variables in edges.items():
            if not hasattr(factor, "connection_number"):
                factor.connection_number = {}
            for i, var in enumerate(variables):
                if not ((factor in self.factors and var in self.variables) or
                        (factor in self.variables and var in self.factors)):
                    raise ValueError("Edges must connect a factor to a variable.")
                self.G.add_edge(factor, var, dim=i)
                factor.connection_number[var.name] = i
        logger.info("FactorGraph is bipartite: variables <-> factors only.")

    def _initialize_cost_tables(self) -> None:
        """Initializes the cost tables for all factor nodes in the graph."""
        for node in list(nx.bipartite.sets(self.G))[1]:
            if isinstance(node, FactorAgent):
                node.initiate_cost_table()
                logger.debug("Cost table initialized for factor node: %s", node.name)

    def get_variable_agents(self) -> List[VariableAgent]:
        """Returns a list of all variable agents in the graph."""
        return self.variables

    def get_factor_agents(self) -> List[FactorAgent]:
        """Returns a list of all factor agents in the graph."""
        return self.factors

    @property
    def diameter(self) -> int:
        """int: The diameter of the factor graph.

        If the graph is not connected, it returns the diameter of the
        largest connected component.
        """
        if not self.G:
            return 0
        if not nx.is_connected(self.G):
            if not list(nx.connected_components(self.G)):
                return 0
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
            if not subgraph.nodes():
                return 0
            return nx.diameter(subgraph)
        return nx.diameter(self.G)

    def __getstate__(self) -> dict:
        """Custom method to control what gets pickled.

        This is used to ensure the object can be correctly serialized.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom method to control unpickling behavior.

        This ensures that the `networkx.Graph` object is correctly
        reconstructed if it was not pickled.
        """
        self.__dict__.update(state)
        if not hasattr(self, "G") or self.G is None:
            self.G = nx.Graph()
            if hasattr(self, "variables") and hasattr(self, "factors"):
                self.G.add_nodes_from(self.variables, bipartite=0)
                self.G.add_nodes_from(self.factors, bipartite=1)
                var_name_to_obj = {var.name: var for var in self.variables}
                for factor in self.factors:
                    if hasattr(factor, "connection_number"):
                        for var_name, dim in factor.connection_number.items():
                            if var_name in var_name_to_obj:
                                var = var_name_to_obj[var_name]
                                self.G.add_edge(factor, var, dim=dim)

    @property
    def original_factors(self) -> List[FactorAgent]:
        """list[FactorAgent]: A deep copy of the original factor agents.

        This is preserved to allow for calculating the true global cost,
        even if factor costs are modified during a simulation run.
        """
        return self._original_factors
