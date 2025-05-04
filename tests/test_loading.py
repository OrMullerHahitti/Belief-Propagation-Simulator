import logging
import os
import pickle
import networkx as nx
from pathlib import Path
import sys
from matplotlib import pyplot as plt

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
            for marker in [".git", "setup.py", "pyproject.toml"]
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
        "factor-graph-random-50-random_intlow1,high1000.5-number49.pkl",
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


def test_graph_structure(factor_graph):
    """
    Test the structure of the factor graph.
    """
    assert isinstance(fg, FactorGraph), "Loaded object is not a FactorGraph"
    assert len(fg.G.nodes()) > 0, "Graph has no nodes"
    assert len(fg.G.edges()) > 0, "Graph has no edges"
    logger.info("Graph structure is valid")


def test_viz(factor_graph):
    """
    Test the visualization of the factor graph.
    """
    try:
        # Assume fg is your FactorGraph and fg.G is the NetworkX graph
        G = factor_graph.G

        # Identify variable and factor nodes
        variable_nodes = [n for n in G.nodes if isinstance(n, VariableAgent)]
        factor_nodes = [n for n in G.nodes if isinstance(n, FactorAgent)]

        # Create bipartite layout
        pos = nx.bipartite_layout(G, variable_nodes)

        plt.figure(figsize=(8, 6))
        # Draw variable nodes (circles)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=variable_nodes,
            node_shape="o",
            node_color="skyblue",
            label="Variables",
        )
        # Draw factor nodes (squares)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=factor_nodes,
            node_shape="s",
            node_color="salmon",
            label="Factors",
        )
        # Draw edges
        nx.draw_networkx_edges(G, pos)
        # Draw labels
        nx.draw_networkx_labels(G, pos)

        plt.legend(scatterpoints=1)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        plt.savefig("factor_graph_viz.png")
        plt.show()
        logger.info("Graph visualization successful")
    except Exception as e:
        logger.error(f"Graph visualization failed: {e}")
        assert False, "Graph visualization failed"
