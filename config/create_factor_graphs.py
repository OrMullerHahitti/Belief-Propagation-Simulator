import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Any, Callable

from utils.config_utils import save_object_as_pickle, load_object_from_pickle
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.factor_graph import FactorGraph
from bp_base.computators import MinSumComputator, MaxSumComputator
from bp_base.components import BPComputator

# Define cost table creation functions at module level so they can be pickled
def create_uniform_table(n, d, **kwargs):
    """Create a table with uniformly distributed random values."""
    return np.random.uniform(**kwargs, size=tuple(d for _ in range(n)))

def create_normal_table(n, d, **kwargs):
    """Create a table with normally distributed random values."""
    return np.random.normal(**kwargs, size=tuple(d for _ in range(n)))

def create_randint_table(n, d, **kwargs):
    """Create a table with random integer values."""
    return np.random.randint(**kwargs, size=tuple(d for _ in range(n)))

def create_rand_table(n, d, **kwargs):
    """Create a table with random values in [0, 1)."""
    return np.random.rand(*tuple(d for _ in range(n)))

def save_graph_config(config: dict, base_dir: str = "saved_configs/graph_configs"):
    """
    Creates and saves a factor graph configuration as a pickle file.
    
    Parameters:
    -----------
    config : dict
        Dictionary containing factor graph configuration parameters including:
        - graph_type: str (e.g., 'cycle-3', 'cycle-4', 'variable-loop')
        - graph_name: str (optional, custom name for this graph)
        - computator_type: str (e.g., 'sum', 'product', 'min', 'max')
        - domain_size: int (size of the message domain)
        - cost_table_params: dict
            - function_type: str (e.g., 'uniform', 'normal', 'randint')
            - params: dict (parameters for the random function)
    
    base_dir : str, optional
        Directory to save the configuration, defaults to "saved_configs/graph_configs"
    
    Returns:
    --------
    str: Path to the saved configuration file
    """
    # Ensure required keys exist in the config
    required_keys = ['graph_type', 'computator_type', 'domain_size', 'cost_table_params']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration parameter: {key}")
    
    # Extract values for filename
    graph_type = config['graph_type']
    computator_type = config['computator_type']
    cost_table_params = config['cost_table_params']
    function_type = cost_table_params.get('function_type', 'unknown')
    
    # Use graph_name if provided, otherwise generate from graph_type
    if 'graph_name' in config and config['graph_name']:
        base_filename = config['graph_name']
    else:
        # Create parameter string for filename
        param_str = ""
        if 'params' in cost_table_params:
            params = cost_table_params['params']
            if isinstance(params, dict):
                # Format parameters for filename
                param_str = "_".join(f"{k}{v}" for k, v in params.items())
            else:
                param_str = str(params)
        
        # Base filename format: graph_type-computator_type-function_type-params
        base_filename = f"{graph_type}-{computator_type}-{function_type}"
        if param_str:
            base_filename += f"-{param_str}"
    
    # Check for existing files with the same base name and find the next index
    os.makedirs(base_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(base_dir, f"{base_filename}-*.pkl"))
    
    # Extract existing indices and find the next one
    indices = [int(f.split('-')[-1].split('.')[0]) for f in existing_files if f.split('-')[-1].split('.')[0].isdigit()]
    next_index = 1
    if indices:
        next_index = max(indices) + 1
    
    # Final filename
    filename = f"{base_filename}-{next_index}.pkl"
    
    # Save configuration
    save_object_as_pickle(config, base_dir, filename)
    
    full_path = os.path.join(base_dir, filename)
    print(f"Graph configuration saved as: {full_path}")
    return full_path

def load_graph_config(filepath):
    """
    Load a graph configuration from a pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
    
    Returns:
    --------
    dict: The loaded graph configuration
    """
    directory, filename = os.path.split(filepath)
    return load_object_from_pickle(directory, filename)

def list_available_configs(base_dir: str = "saved_configs/graph_configs"):
    """
    Lists all available graph configurations in the specified directory.
    
    Parameters:
    -----------
    base_dir : str, optional
        Directory to look for configurations
    
    Returns:
    --------
    list: Available configuration files
    """
    os.makedirs(base_dir, exist_ok=True)
    config_files = glob.glob(os.path.join(base_dir, "*.pkl"))
    
    if not config_files:
        print("No graph configurations found.")
        return []
    
    print("Available graph configurations:")
    for i, filepath in enumerate(config_files, 1):
        filename = os.path.basename(filepath)
        print(f"{i}. {filename}")
    
    return config_files

def get_computator_by_type(computator_type: str):
    """
    Returns the appropriate computator class based on the type.
    
    Parameters:
    -----------
    computator_type: str
        Type of computator ('min-sum', 'max-sum', etc.)
        
    Returns:
    --------
    BPComputator: An instance of the appropriate computator
    """
    computator_map = {
        'min-sum': MinSumComputator(),
        'max-sum': MaxSumComputator(),
        # Add more computator types as needed
    }
    
    if computator_type.lower() not in computator_map:
        raise ValueError(f"Unknown computator type: {computator_type}")
        
    return computator_map[computator_type.lower()]

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
        'rand': create_rand_table
    }
    
    if function_type.lower() not in function_map:
        raise ValueError(f"Unknown random function type: {function_type}")
        
    return function_map[function_type.lower()]

def create_cycle_graph(num_variables: int, domain_size: int, computator, 
                      ct_creation_func, ct_params: Dict[str, Any]) -> Tuple[List[VariableAgent], List[FactorAgent], Dict]:
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
    factors = [FactorAgent(f"f{i}_{i+1}", domain_size, computator, ct_creation_func, ct_params) 
               for i in range(num_variables-1)]
    # Add the closing factor for the cycle
    factors.append(FactorAgent(f"f{num_variables-1}_0", domain_size, computator, ct_creation_func, ct_params))
    
    # Define edges (connections between variables and factors)
    edges = {}
    for i in range(num_variables-1):
        edges[factors[i]] = [variables[i], variables[i+1]]
    # Add the closing edge
    edges[factors[-1]] = [variables[-1], variables[0]]
    
    return variables, factors, edges

def create_factor_graph(config: dict) -> FactorGraph:
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
        # Direct reference to computator class or instance
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
            raise ValueError(f"Invalid cycle graph type: {graph_type}. Expected format: 'cycle-N' where N is an integer.")
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
    factor_graph = create_factor_graph(config)
    
    # Create the graphs directory if it doesn't exist
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Save the factor graph with a similar filename to the config
    graph_filename = config_filename.replace('config-', 'graph-')
    graph_path = os.path.join(graphs_dir, graph_filename)
    
    save_object_as_pickle(factor_graph, graphs_dir, graph_filename)
    print(f"Factor graph saved as: {graph_path}")
    
    return graph_path

def load_factor_graph(filepath: str) -> FactorGraph:
    """
    Load a factor graph from a pickle file.
    
    Parameters:
    -----------
    filepath: str
        Path to the pickle file containing the factor graph
        
    Returns:
    --------
    FactorGraph: The loaded factor graph
    """
    directory, filename = os.path.split(filepath)
    return load_object_from_pickle(directory, filename)
