import logging
import pickle
import sys
import os
from pathlib import Path

import colorlog  # pip install colorlog
import pytest

# Create logs directory if it doesn't exist
log_dir = "test_logs"
os.makedirs(log_dir, exist_ok=True)

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers
if root_logger.handlers:
    root_logger.handlers.clear()

# Create console handler with colored formatting
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)
root_logger.addHandler(console_handler)

# Add file handler
file_handler = logging.FileHandler(os.path.join(log_dir, "debug_graph.log"))
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("Logging is now set up with colored console output")


# Function to find the project root directory
def find_project_root():
    """Find the project root directory by looking for a common marker like .git or a specific file"""
    current_dir = Path.cwd()
    while True:
        # Check if this is the project root (containing typical root markers)
        if any(
            (current_dir / marker).exists()
            for marker in [".git", "setup.py", "pyproject.toml", ".root"]
        ):
            return current_dir

        # Check if we've reached the filesystem root
        if current_dir == current_dir.parent:
            raise FileNotFoundError("Project root not found")

        # Move up one directory
        current_dir = current_dir.parent


# Make sure your project root is in the Python path
project_root = find_project_root()
sys.path.append(str(project_root))


# Safely load pickle by handling errors - MOVED OUTSIDE TRY BLOCK
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


# Import all the required classes before unpickling
try:
    from bp_base.factor_graph import FactorGraph
    from bp_base.agents import VariableAgent, FactorAgent
    from bp_base.components import Message
    import networkx as nx
    import inspect  # Add missing import

    print(f"NetworkX version: {nx.__version__}")
    print(f"FactorGraph class: {inspect.getmro(FactorGraph)}")

    # Try to load the pickle
    pickle_path = os.path.join(
        project_root,
        "configs",
        "factor_graphs",
        "factor-graph-cycle-3-random_intlow1,high100-number150.pkl",
    )
    print(f"Attempting to load: {pickle_path}")

    # Check if file exists
    if not os.path.exists(pickle_path):
        print(f"File does not exist: {pickle_path}")
        # List available factor graph files
        factor_graphs_dir = os.path.join(project_root, "configs", "factor_graphs")
        if os.path.exists(factor_graphs_dir):
            print(f"Available factor graph files in {factor_graphs_dir}:")
            for file in os.listdir(factor_graphs_dir):
                if file.startswith("factor-graph"):
                    print(f"  - {file}")
    else:
        fg = load_pickle(pickle_path)
        if fg is not None:
            print(f"Graph loaded successfully: {type(fg)}")

            # Inspect the graph object
            print("\nFactor Graph attributes:")
            for attr in dir(fg):
                if not attr.startswith("__"):
                    try:
                        value = getattr(fg, attr)
                        print(f"  - {attr}: {type(value)}")
                    except Exception as e:
                        print(f"  - {attr}: Error accessing attribute - {e}")

            # Check if G exists and inspect it
            if hasattr(fg, "G"):
                print("\nNetworkX Graph (G) exists")
                print(f"Type of G: {type(fg.G)}")

                try:
                    print(f"Number of nodes: {len(fg.G.nodes())}")
                    print(f"Number of edges: {len(fg.G.edges())}")

                    # Print some nodes if they exist
                    nodes = list(fg.G.nodes())
                    if nodes:
                        print("First few nodes:")
                        for node in nodes[:5]:  # Print up to 5 nodes
                            print(f"  - {node} (type: {type(node)})")
                    else:
                        print("Graph has no nodes")

                except Exception as e:
                    print(f"Error inspecting graph: {e}")
            else:
                print("\nNo 'G' attribute found in the factor graph object")

            # Check if the graph methods work
            print("\nTesting graph methods:")
            try:
                if hasattr(fg, "initialize_cost_tables"):
                    print("  - initialize_cost_tables: Found")
                    # Don't actually call it as it might modify the graph
                else:
                    print("  - initialize_cost_tables: Not found")

                if hasattr(fg, "initialize_mailbox"):
                    print("  - initialize_mailbox: Found")
                else:
                    print("  - initialize_mailbox: Not found")
            except Exception as e:
                print(f"Error testing methods: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are installed and in your Python path.")
except Exception as e:
    print(f"Unexpected error: {e}")


@pytest.fixture
def factor_graph():
    fg = load_pickle(pickle_path)
    assert fg is not None, "Failed to load factor graph"
    logger.info("Graph loaded successfully")
    return fg


def test_factor_graph_attributes(factor_graph):
    logger.info("Checking attributes of factor_graph")
    assert hasattr(factor_graph, "G"), "No 'G' attribute found"


def test_graph_nodes_edges(factor_graph):
    logger.info(f"Number of nodes: {len(factor_graph.G.nodes())}")
    logger.info(f"Number of edges: {len(factor_graph.G.edges())}")
    assert len(factor_graph.G.nodes()) >= 0, "Node count is negative?"
    assert len(factor_graph.G.edges()) >= 0, "Edge count is negative?"


def test_graph_methods(factor_graph):
    logger.info("Testing graph methods")
    assert hasattr(
        factor_graph, "initialize_cost_tables"
    ), "initialize_cost_tables method not found"
    assert hasattr(
        factor_graph, "initialize_messages"
    ), "initialize_mailbox method not found"


def test_graph_pickle(factor_graph):
    logger.info("Testing graph pickling")
    try:
        # Pickle the graph
        with open("test_factor_graph.pkl", "wb") as f:
            pickle.dump(factor_graph, f)

        # Unpickle the graph
        with open("test_factor_graph.pkl", "rb") as f:
            loaded_graph = pickle.load(f)

        assert loaded_graph is not None, logger.error("Failed to unpickle factor graph")
        logger.info("Graph pickled and unpickled successfully")
    except Exception as e:
        logger.error(f"Error during pickling: {e}")
        assert False, "Pickling failed"


def test_loading_factor_graph():
    logger.info("Testing loading of factor graph")
    try:
        fg = load_pickle(pickle_path)
        for node in fg.G.nodes():
            logger.info(f"Node: {nodes}")
            if isinstance(node, FactorAgent):
                logger.info(
                    f"  - FactorAgent: {node.name} and its table is {node.cost_table}"
                )
                assert node.cost_table is not None

        assert fg is not None, "Failed to load factor graph"
        assert isinstance(fg, FactorGraph), "Failed to load factor graph"
        logger.info("Factor graph loaded successfully")
    except Exception as e:
        logger.error(f"Error loading factor graph: {e}")
        assert False, "Loading failed"


def test_variable_agent_post_init(factor_graph):
    logger.info("Testing VariableAgent post-init")
    v1 = list(factor_graph.G.nodes())[0]
    assert isinstance(v1, VariableAgent), "Node is not a VariableAgent"
    assert hasattr(v1, "mailer"), "VariableAgent has no mailbox"
    assert isinstance(v1.inbox, list), "inbox is not a list"
    logger.info("VariableAgent messages inbox : %s", v1.inbox)
    logger.info("VariableAgent messages to send : %s", v1.mailer._outgoing)
    assert v1.domain == 3, "Domain is not 3"
    assert v1.name == "x1", "Name is not x1"
