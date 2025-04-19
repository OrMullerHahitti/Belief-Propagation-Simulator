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
