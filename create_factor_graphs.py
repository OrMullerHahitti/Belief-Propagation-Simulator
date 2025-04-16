import os
from utils.config_utils import save_object_as_pickle
import glob

def save_graph_config(config: dict, base_dir: str = "saved_configs"):
    """
    Creates and saves a factor graph configuration as a pickle file.
    
    Parameters:
    -----------
    config : dict
        Dictionary containing all graph configuration parameters including:
        - engine_type: str (e.g., 'max-sum', 'min-sum')
        - graph_type: str (e.g., 'cycle-3', 'cycle-4')
        - cost_table_params: dict (parameters for cost table creation)
            - function_type: str (e.g., 'uniform', 'normal')
            - params: dict (parameters specific to the function)
        - domain_size: int (size of the message domain)
        - other optional configuration parameters
    
    base_dir : str, optional
        Directory to save the configuration, defaults to "saved_configs"
    
    Returns:
    --------
    str: Path to the saved configuration file
    """
    # Ensure required keys exist in the config
    required_keys = ['engine_type', 'graph_type', 'cost_table_params', 'domain_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration parameter: {key}")
    
    # Extract values for filename
    engine_type = config['engine_type']
    graph_type = config['graph_type']
    cost_table_params = config['cost_table_params']
    function_type = cost_table_params.get('function_type', 'unknown')
    
    # Create parameter string for filename
    param_str = ""
    if 'params' in cost_table_params:
        params = cost_table_params['params']
        if isinstance(params, dict):
            # Format parameters for filename
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
        else:
            param_str = str(params)
    
    # Base filename format: engine_type-graph_type-function_type-params
    base_filename = f"{engine_type}-{graph_type}-{function_type}"
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
    print(f"Configuration saved as: {full_path}")
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
    from utils.config_utils import load_object_from_pickle
    directory, filename = os.path.split(filepath)
    return load_object_from_pickle(directory, filename)
