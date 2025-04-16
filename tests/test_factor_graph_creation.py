import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from config.create_factor_graphs import (
    create_factor_graph, 
    create_and_save_factor_graph,
    load_factor_graph, 
    load_graph_config,
    list_available_configs,
    create_uniform_table  # Import the new function for the direct reference test
)
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.computators import MinSumComputator, MaxSumComputator
from utils.ct_utils import create_random_int_table  # Assume this exists for test compatibility


@pytest.fixture
def test_dir():
    """Create a test directory for saved configs and graphs."""
    test_dir_path = Path('test_configs')
    test_dir_path.mkdir(exist_ok=True)
    yield test_dir_path
    
    # Cleanup after tests
    import shutil
    if test_dir_path.exists():
        shutil.rmtree(test_dir_path)


@pytest.fixture
def cycle_3_config():
    """Generate a cycle-3 test configuration."""
    return {
        'graph_type': 'cycle-3',
        'graph_name': 'test-cycle-3',
        'computator_type': 'min-sum',
        'domain_size': 3,
        'cost_table_params': {
            'function_type': 'uniform',
            'params': {'low': -10, 'high': 10}
        }
    }


@pytest.fixture
def cycle_4_config():
    """Generate a cycle-4 test configuration."""
    return {
        'graph_type': 'cycle-4',
        'graph_name': 'test-cycle-4',
        'computator_type': 'max-sum',
        'domain_size': 4,
        'cost_table_params': {
            'function_type': 'normal',
            'params': {'loc': 0, 'scale': 5}
        }
    }


@pytest.fixture
def cycle_3_config_direct_refs():
    """Generate a cycle-3 test configuration with direct references to classes/functions."""
    param_dict = {'low': 0, 'high': 10}
    return {
        'graph_type': 'cycle-3',
        'graph_name': 'test-cycle-3-direct',
        'computator': MinSumComputator(),
        'domain': 3,
        'ct_creation_func': create_uniform_table,  # Use the module-level function
        'param': param_dict
    }


def test_create_cycle_3_graph(cycle_3_config):
    """Test creating a cycle-3 factor graph."""
    # Create factor graph from config
    graph = create_factor_graph(cycle_3_config)
    
    # Verify graph structure
    assert isinstance(graph, FactorGraph)
    
    # Count variable and factor nodes
    variable_count = 0
    factor_count = 0
    for node in graph.G.nodes():
        if isinstance(node, VariableAgent):
            variable_count += 1
        elif isinstance(node, FactorAgent):
            factor_count += 1
    
    # Cycle-3 should have 3 variables and 3 factors
    assert variable_count == 3
    assert factor_count == 3
    
    # Verify edges: each factor should connect to 2 variables
    for node in graph.G.nodes():
        if isinstance(node, FactorAgent):
            neighbors = list(graph.G.neighbors(node))
            assert len(neighbors) == 2
            for neighbor in neighbors:
                assert isinstance(neighbor, VariableAgent)


def test_save_and_load_graph(cycle_4_config, test_dir):
    """Test saving and loading a factor graph."""
    # Create and save graph
    graph_path = create_and_save_factor_graph(
        cycle_4_config, 
        graphs_dir=str(test_dir)
    )
    
    # Verify file was created
    assert os.path.exists(graph_path)
    
    # Load the graph
    loaded_graph = load_factor_graph(graph_path)
    
    # Verify loaded graph is a FactorGraph
    assert isinstance(loaded_graph, FactorGraph)
    
    # Count nodes in loaded graph
    variable_count = 0
    factor_count = 0
    for node in loaded_graph.G.nodes():
        if isinstance(node, VariableAgent):
            variable_count += 1
        elif isinstance(node, FactorAgent):
            factor_count += 1
    
    # Cycle-4 should have 4 variables and 4 factors
    assert variable_count == 4
    assert factor_count == 4


def test_cost_table_generation(cycle_3_config):
    """Test that cost tables are properly generated."""
    # Create graph
    graph = create_factor_graph(cycle_3_config)
    
    # Check each factor has a cost table of correct shape
    for node in graph.G.nodes():
        if isinstance(node, FactorAgent):
            # Cost table should exist
            assert node.cost_table is not None
            
            # Shape should be (domain_size, domain_size) since factors connect to 2 variables
            domain_size = cycle_3_config['domain_size']
            expected_shape = (domain_size, domain_size)
            assert node.cost_table.shape == expected_shape
            
            # Values should be within the specified range
            if cycle_3_config['cost_table_params']['function_type'] == 'uniform':
                low = cycle_3_config['cost_table_params']['params']['low']
                high = cycle_3_config['cost_table_params']['params']['high']
                assert np.all(node.cost_table >= low)
                assert np.all(node.cost_table < high)


def test_list_configs(cycle_3_config, cycle_4_config, test_dir):
    """Test listing available configurations."""
    # Save a couple of configurations
    create_and_save_factor_graph(cycle_3_config, graphs_dir=str(test_dir))
    create_and_save_factor_graph(cycle_4_config, graphs_dir=str(test_dir))
    
    # List configurations
    configs = list_available_configs(base_dir=str(test_dir))
    
    # Should find at least 2 config files
    assert len(configs) >= 2


def test_create_graph_with_direct_references(cycle_3_config_direct_refs):
    """Test creating a factor graph with direct class/function references."""
    # Create factor graph from config using direct references
    graph = create_factor_graph(cycle_3_config_direct_refs)
    
    # Verify graph structure
    assert isinstance(graph, FactorGraph)
    
    # Count variable and factor nodes
    variable_count = 0
    factor_count = 0
    for node in graph.G.nodes():
        if isinstance(node, VariableAgent):
            variable_count += 1
        elif isinstance(node, FactorAgent):
            factor_count += 1
    
    # Cycle-3 should have 3 variables and 3 factors
    assert variable_count == 3
    assert factor_count == 3


