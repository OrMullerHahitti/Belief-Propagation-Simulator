from typing import Dict, Any, Tuple, List

import numpy as np

from bp_base.agents import FactorAgent, VariableAgent
from bp_base.computators import BPComputator
from bp_base.factor_graph import FactorGraph
from config.create_factor_graph_config import create_randint_table, get_computator_by_type
from utils.ct_utils import create_uniform_table, create_normal_table, create_symmetric_cost_table


def create_ndarray(domain: int, num_connected: int) -> np.ndarray:
    """
    Create an ndarray with 'num_connected' dimensions,
    each dimension of size 'domain'.

    :param domain: The size of each dimension in the ndarray.
    :param num_connected: The number of dimensions in the ndarray.
    :return: An ndarray of shape (domain, domain, ..., domain)
             (num_connected times).
    """
    shape = (domain,) * num_connected  # e.g., (3,3) for domain=3, num_connected=2
    # Populate the array with random integers (0..9) for demonstration:
    return np.random.randint(0, 10, size=shape)


def get_cost_table_function(function_type: str):
    """
    Returns the appropriate cost table creation function based on the type.

    Parameters:
    -----------
    function_type: str
        Type of random function ('uniform', 'normal', 'randint', etc.)

    Returns:
    --------
    function: A function that creates random arrays
    """
    function_map = {
        'uniform': create_uniform_table,
        'normal': create_normal_table,
        'randint': create_randint_table,
        'symmetric': create_symmetric_cost_table
    }

    if function_type.lower() not in function_map:
        raise ValueError(f"Unknown random function type: {function_type}")

    return function_map[function_type.lower()]


def create_cycle_graph(num_variables: int, domain_size: int, computator,
                       ct_creation_func, ct_params: Dict[str, Any]) -> Tuple[
    List[VariableAgent], List[FactorAgent], Dict]:
    """
    Creates variables, factors and edges for a cycle graph with the specified number of variables.

    Parameters:
    -----------
    num_variables: int
        Number of variables in the cycle
    domain_size: int
        Size of the variable domain
    computator: BPComputator
        Computator instance to use
    ct_creation_func: function
        Function to create cost tables
    ct_params: dict
        Parameters for cost table creation

    Returns:
    --------
    Tuple[List[VariableAgent], List[FactorAgent], Dict]:
        Variables, factors, and edges for the graph
    """
    variables = [VariableAgent(f"x{i}", domain_size, computator) for i in range(num_variables)]
    factors = [FactorAgent(f"f{i}_{i + 1}", domain_size, computator, ct_creation_func, ct_params)
               for i in range(num_variables - 1)]
    # Add the closing factor for the cycle
    factors.append(FactorAgent(f"f{num_variables - 1}_0", domain_size, computator, ct_creation_func, ct_params))

    # Define edges (connections between variables and factors)
    edges = {}
    for i in range(num_variables - 1):
        edges[factors[i]] = [variables[i], variables[i + 1]]
    # Add the closing edge
    edges[factors[-1]] = [variables[-1], variables[0]]

    return variables, factors, edges


def create_factor_graph_from_config(config: dict) -> FactorGraph:
    """
    Creates a factor graph based on the provided configuration.

    Parameters:
    -----------
    config: dict
        Graph configuration dictionary with parameters

    Returns:
    --------
    FactorGraph: The created factor graph
    """
    # Extract configuration parameters
    graph_type = config.get('graph_type')

    # Support for both parameter naming conventions
    domain_size = config.get('domain_size', config.get('domain'))

    # Support both direct computator and string type
    computator = None
    if 'computator' in config:
        # Direct reference to computator class or instance .e.g. MaxSumComputator,MinSumComputator
        computator = config['computator']
        if not isinstance(computator, BPComputator) and callable(computator):
            # It's a class reference, instantiate it
            computator = computator()
    elif 'computator_type' in config:
        # String reference to computator type
        computator_type = config['computator_type']
        computator = get_computator_by_type(computator_type)
    else:
        raise ValueError("Config must contain either 'computator' or 'computator_type'")

    # Support both direct function reference and string type for cost table creation
    ct_creation_func = None
    ct_params = {}

    if 'ct_creation_func' in config:
        # Direct function reference
        ct_creation_func = config['ct_creation_func']
        ct_params = config.get('param', {})
    elif 'cost_table_params' in config:
        # String reference + params dict
        cost_table_params = config['cost_table_params']
        function_type = cost_table_params['function_type']
        ct_creation_func = get_cost_table_function(function_type)
        ct_params = cost_table_params.get('params', {})
    else:
        raise ValueError("Config must contain either 'ct_creation_func' or 'cost_table_params'")

    # Create the appropriate graph structure
    if not graph_type:
        raise ValueError("Config must specify 'graph_type'")

    if graph_type.startswith('cycle-'):
        try:
            num_variables = int(graph_type.split('-')[1])
            variables, factors, edges = create_cycle_graph(num_variables, domain_size,
                                                           computator, ct_creation_func, ct_params)
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid cycle graph type: {graph_type}. Expected format: 'cycle-N' where N is an integer.")
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    # Create and return the factor graph
    return FactorGraph(variables, factors, edges)

def create_and_save_factor_graph(config: dict, graphs_dir: str = "saved_configs/factor_graphs") -> str:
    """
    Creates a factor graph based on the configuration and saves it.

    Parameters:
    -----------
    config: dict
        Graph configuration dictionary
    graphs_dir: str, optional
        Directory to save the factor graph

    Returns:
    --------
    str: Path to the saved factor graph
    """
    # Save the configuration first
    config_path = save_graph_config(config)
    config_filename = os.path.basename(config_path)

    # Create the factor graph
    factor_graph = create_factor_graph_from_config(config)

    # Create the graphs directory if it doesn't exist
    os.makedirs(graphs_dir, exist_ok=True)

    # Save the factor graph with a similar filename to the config
    graph_filename = config_filename.replace('config-', 'graph-')
    graph_path = os.path.join(graphs_dir, graph_filename)

    save_object_as_pickle(factor_graph, graphs_dir, graph_filename)
    print(f"Factor graph saved as: {graph_path}")

    return graph_path