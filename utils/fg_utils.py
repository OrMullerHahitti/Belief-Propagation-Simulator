import pickle
import sys
import os
from typing import Callable, Dict, Any, List, Tuple
from functools import lru_cache

import networkx as nx
import numpy as np

from base_models.protocols import Message
from utils.path_utils import find_project_root  # Added import
from bp_base.factor_graph import FactorGraph
from base_models.agents import VariableAgent, FactorAgent

# Make sure your project root is in the Python path
project_root = find_project_root()
sys.path.append(str(project_root))


def _make_variable(idx: int, domain: int) -> VariableAgent:
    name = f"x{idx}"
    return VariableAgent(name=name, domain=domain)


def _make_factor(
    name: str, domain: int, ct_factory: Callable, ct_params: dict
) -> FactorAgent:
    # we postpone cost‑table creation until FactorGraph initialises
    return FactorAgent(
        name=name,
        domain=domain,
        ct_creation_func=ct_factory,
        param=ct_params,
    )


def _build_factor_edge_list(
    edges: List[Tuple[VariableAgent, VariableAgent]], domain_size, ct_factory, ct_params
) -> Dict[FactorAgent, List[VariableAgent]]:
    """
    Build a dictionary of edges from a list of edges.
    :param edges: List of edges
    :return: Dictionary of edges
    """
    edge_dict = {}
    for edge in edges:
        a, b = edge
        fname = f"f{a.name[1:]}{b.name[1:]}"
        fnode = _make_factor(fname, domain_size, ct_factory, ct_params)
        edge_dict[fnode] = [a, b]
    return edge_dict


def _make_connections_density(
    variable_list: List[VariableAgent], density: float
) -> List[Tuple[VariableAgent, VariableAgent]]:
    """ """
    r_graph = nx.erdos_renyi_graph(len(variable_list), density)
    variable_map = dict(enumerate(variable_list))
    full_graph = nx.relabel_nodes(r_graph, variable_map)
    return list(full_graph.edges())


##-------------------------- fg builder --------------------------##
class FGBuilder:

    @staticmethod
    def build_random_graph(
        num_vars: int,
        domain_size: int,
        ct_factory: Callable,
        ct_params: Dict[str, Any],
        density: float,
    ):
        """
        Build a random binary constraints graph.
        :param num_vars: Number of variables
        :param domain_size: Size of the domain
        :param ct_factory: Cost table factory
        :param ct_params: Cost table parameters
        :param density: Density of the graph
        :return: List of variables, list of factors, dictionary of edges
        """
        variables: List[VariableAgent] = [
            _make_variable(i + 1, domain_size) for i in range(num_vars)
        ]
        connections = _make_connections_density(variables, density)
        edges: Dict[FactorAgent, List[VariableAgent]] = _build_factor_edge_list(
            connections, domain_size, ct_factory, ct_params
        )
        factors = list(edges.keys())

        return FactorGraph(variables, factors, edges)



    ### ------------ IMPORTANT:  DO NOT CHANGE ------------------ ###
    @staticmethod
    def build_cycle_graph(
        *,
        num_vars: int,
        domain_size: int,
        ct_factory: Callable,
        ct_params: Dict[str, Any],
        density: float = 1.0,  # density is not used in cycle graph, but kept for consistency
    ):
        """
        Simple N-variable cycle: x1–f12–x2–f23–…–xn–fn1–x1
        Returns (variables, factors, edges).

        Parameters
        ----------
        num_vars
        domain_size
        ct_factory
        ct_params
        density
        """
        variables: List[VariableAgent] = [
            _make_variable(i + 1, domain_size) for i in range(num_vars)
        ]

        factors: List[FactorAgent] = []
        edges: Dict[FactorAgent, List[VariableAgent]] = {}

        for j in range(num_vars):
            a, b = variables[j], variables[(j + 1) % num_vars]
            f_name = f"f{a.name[1:]}{b.name[1:]}"
            f_node = _make_factor(f_name, domain_size, ct_factory, ct_params)
            factors.append(f_node)
            edges[f_node] = [a, b]

        return FactorGraph(variables, factors, edges)


    ##### ------------ Private Methods ------------------ #####


def get_message_shape(domain_size: int,connections: int = 2, ) -> tuple[int, ...]:
    """
    Calculate the shape of the cost table based on the number of connections and domain size.

    Args:
        connections (int): Number of connections for the factor.
        domain_size (int): Size of the domain for each variable.

    Returns:
        tuple[int, ...]: Shape of the cost table.
    """
    return (domain_size,) * connections

@lru_cache(maxsize=128)
def get_broadcast_shape(ct_dims,domain_size: int, ax:int) -> tuple[int, ...]:
    #create ones np array with the shape of the cost table
    br_message = np.ones(ct_dims)
    br_message[ax] = domain_size
    return tuple(br_message)

print(get_broadcast_shape(5,3,2))

def generate_random_cost(fg: FactorGraph) -> int | float:
    cost = 0
    for fact in fg.factors:
        random_index = tuple(
            np.random.randint(0, fact.domain, size=fact.cost_table.ndim)
        )
        cost += fact.cost_table[random_index]
    return cost


# Custom unpickler to handle potential issues
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle potential module renames or reorganizations
        if module == "bp_base.factor_graph" and name == "FactorGraph":
            from bp_base.factor_graph import FactorGraph

            return FactorGraph
        elif module == "bp_base.agents" and name == "VariableAgent":
            from base_models.agents import VariableAgent

            return VariableAgent
        elif module == "bp_base.agents" and name == "FactorAgent":
            from base_models.agents import FactorAgent

            return FactorAgent
        elif module == "bp_base.components" and name == "Message":
            from base_models.components import Message

            return Message
        # Add more mappings as needed

        try:
            # Default behavior - try to import normally
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import {module}.{name}: {e}")
            # Return a placeholder class if the real one can't be imported
            return type(name, (), {})


# Safely load pickle file
def load_pickle_safely(file_path):
    try:
        with open(file_path, "rb") as f:
            return SafeUnpickler(f).load()
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


# Fix potential issues with the factor graph after loading
def repair_factor_graph(fg):
    """Attempt to repair any issues with the loaded factor graph"""

    # Ensure G exists
    if not hasattr(fg, "G") or fg.G is None:
        print("Initializing missing NetworkX graph")
        fg.G = nx.Graph()

        # Reconstruct graph from variables and factors
        if hasattr(fg, "variables") and hasattr(fg, "factors"):
            # Add nodes
            fg.G.add_nodes_from(fg.variables)
            fg.G.add_nodes_from(fg.factors)

            # Try to reconstruct edges
            for factor in fg.factors:
                if hasattr(factor, "connection_number"):
                    for var, dim in factor.connection_number.items():
                        fg.G.add_edge(factor, var, dim=dim)

    # Ensure all required attributes exist
    for node in fg.G.nodes():
        # Ensure mailbox exists
        if not hasattr(node, "mailbox"):
            node.mailbox = []

        # Ensure other attributes exist based on node type
        if hasattr(node, "type") and node.type == "factor":
            if not hasattr(node, "cost_table") or node.cost_table is None:
                try:
                    # Try to initialize cost table if missing
                    if hasattr(node, "initiate_cost_table"):
                        node.initiate_cost_table()
                except Exception as e:
                    print(f"Could not initialize cost table for {node}: {e}")

    return fg

#
# try:
#     # Import all the required classes
#     from bp_base.factor_graph import FactorGraph
#     from base_models.agents import VariableAgent, FactorAgent
#     from base_models.components import Message
#
#     print(f"NetworkX version: {nx.__version__}")
#
#     # Try to load the pickle
#     pickle_path = os.path.join(
#         project_root,
#         "configs",
#         "factor_graphs",
#         "factor-graph-cycle-3-random_intlow1,high100-number5.pkl",
#     )
#     print(f"Attempting to load: {pickle_path}")
#
#     # Check if file exists
#     if not os.path.exists(pickle_path):
#         print(f"File does not exist: {pickle_path}")
#         # List available factor graph files
#         factor_graphs_dir = os.path.join(project_root, "configs", "factor_graphs")
#         if os.path.exists(factor_graphs_dir):
#             print(f"Available factor graph files in {factor_graphs_dir}:")
#             for file in os.listdir(factor_graphs_dir):
#                 if file.startswith("factor-graph"):
#                     print(f"  - {file}")
#                     # Update pickle_path to use an existing file
#                     pickle_path = os.path.join(factor_graphs_dir, file)
#                     print(f"Using first available file: {pickle_path}")
#                     break
#
#     if os.path.exists(pickle_path):
#         # Load the factor graph
#         fg = load_pickle_safely(pickle_path)
#
#         if fg is not None:
#             print(f"Graph loaded. Type: {type(fg)}")
#
#             # Repair any issues with the loaded graph
#             fg = repair_factor_graph(fg)
#
#             # Check if the graph is usable
#             print("\nFactor graph details:")
#             try:
#                 print(f"Variables: {len(fg.variables)}")
#                 print(f"Factors: {len(fg.factors)}")
#                 print(f"Graph nodes: {len(fg.G.nodes())}")
#                 print(f"Graph edges: {len(fg.G.edges())}")
#
#                 # Print first few nodes
#                 print("\nFirst few nodes:")
#                 for i, node in enumerate(fg.G.nodes()):
#                     if i >= 5:  # Limit to 5 nodes
#                         break
#                     print(f"  - {node}")
#
#                 # Try to access a variable node's attributes
#                 if fg.variables:
#                     var = fg.variables[0]
#                     print(f"\nFirst variable: {var.name}, Domain: {var.domain}")
#
#                 # Try to access a factor node's attributes
#                 if fg.factors:
#                     factor = fg.factors[0]
#                     print(f"\nFirst factor: {factor.name}")
#                     if hasattr(factor, "cost_table") and factor.cost_table is not None:
#                         print(f"Cost table shape: {factor.cost_table.shape}")
#             except Exception as e:
#                 print(f"Error inspecting graph: {e}")
#
#             # Try to save the repaired graph
#             try:
#                 output_path = os.path.join(
#                     os.path.dirname(pickle_path),
#                     "repaired_" + os.path.basename(pickle_path),
#                 )
#                 with open(output_path, "wb") as f:
#                     pickle.dump(fg, f, protocol=pickle.HIGHEST_PROTOCOL)
#                 print(f"\nRepaired graph saved to: {output_path}")
#             except Exception as e:
#                 print(f"Error saving repaired graph: {e}")
#     else:
#         print("No factor graph files found.")
#
# except ImportError as e:
#     print(f"Import error: {e}")
#     print("Make sure all required modules are installed and in your Python path.")
# except Exception as e:
#     print(f"Unexpected error: {e}")


def get_bound(factor_graph: FactorGraph, reduce_func=np.min) -> float:
    """
    Get the lower bound of the factor graph by summing the minimum costs of each factor.

    Args:
        factor_graph (FactorGraph): The factor graph to analyze.
        reduce_func (callable): A function to reduce the cost table, default is np.min.

    Returns:
        float: The lower bound of the factor graph.
    """
    bound = 0.0
    for factor in factor_graph.factors:
        if hasattr(factor, "cost_table") and factor.cost_table is not None:
            bound += reduce_func(factor.cost_table)
    return bound
