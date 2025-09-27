"""Utilities for creating, loading, and manipulating factor graphs.

This module provides a collection of helper functions and classes for common
tasks related to factor graphs, such as building graphs with specific
topologies (random, cycle), calculating bounds, and safely handling pickled
graph objects.
"""
import pickle
import sys
from typing import Callable, Dict, Any, List, Tuple
from functools import lru_cache

import networkx as nx
import numpy as np

from .path_utils import find_project_root
from ..bp.factor_graph import FactorGraph
from ..configs.global_config_mapping import get_ct_factory, CTFactory
from ..core.agents import VariableAgent, FactorAgent

project_root = find_project_root()
sys.path.append(str(project_root))


def _make_variable(idx: int, domain: int) -> VariableAgent:
    """Creates a single `VariableAgent` with a standardized name."""
    return VariableAgent(name=f"x{idx}", domain=domain)


def _make_factor(
    name: str, domain: int, ct_factory: Callable | CTFactory | str, ct_params: dict
) -> FactorAgent:
    """Creates a single `FactorAgent`, deferring cost table creation."""
    ct_fn = get_ct_factory(ct_factory)
    return FactorAgent(name=name, domain=domain, ct_creation_func=ct_fn, param=ct_params)


def _build_factor_edge_list(
    edges: List[Tuple[VariableAgent, VariableAgent]], domain_size: int, ct_factory: Any, ct_params: dict
) -> Dict[FactorAgent, List[VariableAgent]]:
    """Creates factor nodes for binary constraints and maps them to variables."""
    edge_dict = {}
    for a, b in edges:
        fname = f"f{a.name[1:]}{b.name[1:]}"
        fnode = _make_factor(fname, domain_size, ct_factory, ct_params)
        edge_dict[fnode] = [a, b]
    return edge_dict


def _make_connections_density(
    variable_list: List[VariableAgent], density: float
) -> List[Tuple[VariableAgent, VariableAgent]]:
    """Creates a random graph of variable connections based on a given density."""
    r_graph = nx.erdos_renyi_graph(len(variable_list), density)
    variable_map = dict(enumerate(variable_list))
    full_graph = nx.relabel_nodes(r_graph, variable_map)
    return list(full_graph.edges())


class FGBuilder:
    """A builder class providing static methods to construct factor graphs."""

    @staticmethod
    def build_random_graph(
        num_vars: int,
        domain_size: int,
        ct_factory: Callable | CTFactory | str,
        ct_params: Dict[str, Any],
        density: float,
    ) -> FactorGraph:
        """Builds a factor graph with random binary constraints.

        Args:
            num_vars: The number of variables in the graph.
            domain_size: The size of the domain for each variable.
            ct_factory: The factory for creating cost tables.
            ct_params: Parameters for the cost table factory.
            density: The density of the graph (probability of an edge).

        Returns:
            A `FactorGraph` instance with a random topology.
        """
        variables = [_make_variable(i + 1, domain_size) for i in range(num_vars)]
        connections = _make_connections_density(variables, density)
        edges = _build_factor_edge_list(connections, domain_size, ct_factory, ct_params)
        factors = list(edges.keys())
        return FactorGraph(variables, factors, edges)

    @staticmethod
    def build_cycle_graph(
        num_vars: int,
        domain_size: int,
        ct_factory: Callable | CTFactory | str,
        ct_params: Dict[str, Any],
        **kwargs,
    ) -> FactorGraph:
        """Builds a factor graph with a simple cycle topology.

        The graph structure is `x1 – f12 – x2 – ... – xn – fn1 – x1`.

        Args:
            num_vars: The number of variables in the cycle.
            domain_size: The size of the domain for each variable.
            ct_factory: The factory for creating cost tables.
            ct_params: Parameters for the cost table factory.
            **kwargs: Catches unused arguments like `density` for API consistency.

        Returns:
            A `FactorGraph` instance with a cycle topology.
        """
        variables = [_make_variable(i + 1, domain_size) for i in range(num_vars)]
        edges = {}
        for j in range(num_vars):
            a, b = variables[j], variables[(j + 1) % num_vars]
            f_name = f"f{a.name[1:]}{b.name[1:]}"
            f_node = _make_factor(f_name, domain_size, ct_factory, ct_params)
            edges[f_node] = [a, b]
        factors = list(edges.keys())
        return FactorGraph(variables, factors, edges)


def get_message_shape(domain_size: int, connections: int = 2) -> tuple[int, ...]:
    """Calculates the shape of a cost table for a factor.

    Args:
        domain_size: The size of the domain for each connected variable.
        connections: The number of variables connected to the factor.

    Returns:
        A tuple representing the shape of the cost table.
    """
    return (domain_size,) * connections


@lru_cache(maxsize=128)
def get_broadcast_shape(ct_dims: int, domain_size: int, ax: int) -> tuple[int, ...]:
    """Calculates the shape for broadcasting a message into a cost table."""
    shape = [1] * ct_dims
    shape[ax] = domain_size
    return tuple(shape)


def generate_random_cost(fg: FactorGraph) -> float:
    """Calculates a total cost based on a random assignment for each factor.

    Args:
        fg: The factor graph to evaluate.

    Returns:
        The sum of costs from a random assignment in each factor's cost table.
    """
    cost = 0.0
    for fact in fg.factors:
        random_index = tuple(np.random.randint(0, fact.domain, size=fact.cost_table.ndim))
        cost += fact.cost_table[random_index]
    return cost


class SafeUnpickler(pickle.Unpickler):
    """A custom unpickler to handle module path changes during deserialization.

    This class overrides `find_class` to intercept and correct module paths
    that may have changed between the time of pickling and unpickling,
    preventing `ImportError` or `AttributeError`.
    """
    def find_class(self, module: str, name: str) -> Any:
        """Finds a class, handling potential module path changes."""
        module_mapping = {
            "bp.factor_graph": "propflow.bp.factor_graph",
            "bp.agents": "propflow.core.agents",
            "bp.components": "propflow.core.components",
        }
        module = module_mapping.get(module, module)
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import {module}.{name}: {e}")
            return type(name, (), {})


def load_pickle_safely(file_path: str) -> Any:
    """Loads a pickle file using the `SafeUnpickler` to prevent import errors.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The deserialized object, or `None` if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            return SafeUnpickler(f).load()
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


def repair_factor_graph(fg: FactorGraph) -> FactorGraph:
    """Attempts to repair a loaded factor graph by ensuring essential attributes exist.

    This is useful when unpickling older versions of `FactorGraph` objects
    that may be missing attributes added in newer versions.

    Args:
        fg: The `FactorGraph` object to repair.

    Returns:
        The repaired `FactorGraph` object.
    """
    if not hasattr(fg, "G") or fg.G is None:
        print("Initializing missing NetworkX graph")
        fg.G = nx.Graph()
        if hasattr(fg, "variables") and hasattr(fg, "factors"):
            fg.G.add_nodes_from(fg.variables)
            fg.G.add_nodes_from(fg.factors)
            for factor in fg.factors:
                if hasattr(factor, "connection_number"):
                    for var, dim in factor.connection_number.items():
                        fg.G.add_edge(factor, var, dim=dim)
    for node in fg.G.nodes():
        if not hasattr(node, "mailbox"):
            node.mailbox = []
        if hasattr(node, "type") and node.type == "factor":
            if not hasattr(node, "cost_table") or node.cost_table is None:
                try:
                    if hasattr(node, "initiate_cost_table"):
                        node.initiate_cost_table()
                except Exception as e:
                    print(f"Could not initialize cost table for {node}: {e}")
    return fg


def get_bound(factor_graph: FactorGraph, reduce_func: Callable = np.min) -> float:
    """Calculates a simple bound on the total cost of the factor graph.

    This is typically used to get a lower bound by summing the minimum values
    from each factor's cost table.

    Args:
        factor_graph: The factor graph to analyze.
        reduce_func: The function to apply to each cost table to get a single
            value (e.g., `np.min` for a lower bound, `np.max` for an upper bound).
            Defaults to `np.min`.

    Returns:
        The calculated bound.
    """
    bound = 0.0
    for factor in factor_graph.factors:
        if hasattr(factor, "cost_table") and factor.cost_table is not None:
            bound += reduce_func(factor.cost_table)
    return bound
