
# Tests for loading components
import pytest
import os
from bp_base.factor_graph import FactorGraph # Assuming FactorGraph class
from utils.path_utils import  find_project_root
from utils.loading_utils import load_pickle


# Define the root directory of the project for constructing paths
# This might need adjustment based on where tests are run from
# For now, assuming tests are run from the workspace root or paths are relative to it in a known way.
TEST_WORKSPACE_ROOT = find_project_root()
# Or use a more robust way to find the root if available, e.g., based on __file__

@pytest.fixture
def factor_graph_pickle_path():
    # Assuming the test pickle file is in configs/factor_graphs/
    # Adjust if the actual location is different
    path = os.path.join(TEST_WORKSPACE_ROOT, "configs", "factor_graphs", "test-factor.pkl")
    # We can't check for os.path.exists(path) here directly in the thought process,
    # but the test will fail if it doesn't exist, which is the desired behavior.
    return path

def test_load_factor_graph_from_pickle(factor_graph_pickle_path):
    """
    Tests loading a FactorGraph object from a .pkl file.
    Assumes 'test_factor_graph.pkl' exists in the specified path.
    """
    # Check if the assumed loading function can be imported and used.
    # If load_factor_graph is a static method of FactorGraph, the call would be different.
    try:
        loaded_fg = load_pickle(factor_graph_pickle_path)
    except FileNotFoundError:
        pytest.fail(f"Test factor graph pickle file not found at {factor_graph_pickle_path}. " 
                    "Please ensure 'test_factor_graph.pkl' is in the correct location.")
    except Exception as e:
        pytest.fail(f"Loading factor graph failed with an unexpected error: {e}")

    assert loaded_fg is not None, "Loaded factor graph should not be None"
    assert isinstance(loaded_fg, FactorGraph), "Loaded object should be an instance of FactorGraph"
    
    # Add more specific assertions based on what 'test_factor_graph.pkl' is expected to contain
    # For example, if it's known to have a specific name or number of nodes/edges:
    #assert loaded_fg.name is not None, "Loaded factor graph should have a name" TODO - make it have a name
    # assert loaded_fg.name == "test_factor_graph" # Or whatever the expected name is
    assert len(loaded_fg.G.nodes) > 0, "Loaded factor graph should have nodes"
    # assert len(loaded_fg.get_variable_agents()) > 0 # Example assertion
    # assert len(loaded_fg.get_factor_agents()) > 0 # Example assertion

# TODO: Add test cases for loading other components or configurations if applicable
# For example, loading engine configurations, policy configurations etc.
