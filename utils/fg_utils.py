import pickle
import sys
import os
from pathlib import Path
import networkx as nx
import numpy as np

from utils.path_utils import find_project_root  # Added import

# Make sure your project root is in the Python path
project_root = find_project_root()
sys.path.append(str(project_root))


# Custom unpickler to handle potential issues
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle potential module renames or reorganizations
        if module == 'bp_base.factor_graph' and name == 'FactorGraph':
            from bp_base.factor_graph import FactorGraph
            return FactorGraph
        elif module == 'bp_base.agents' and name == 'VariableAgent':
            from bp_base.agents import VariableAgent
            return VariableAgent
        elif module == 'bp_base.agents' and name == 'FactorAgent':
            from bp_base.agents import FactorAgent
            return FactorAgent
        elif module == 'bp_base.components' and name == 'Message':
            from bp_base.components import Message
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
        with open(file_path, 'rb') as f:
            return SafeUnpickler(f).load()
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


# Fix potential issues with the factor graph after loading
def repair_factor_graph(fg):
    """Attempt to repair any issues with the loaded factor graph"""

    # Ensure G exists
    if not hasattr(fg, 'G') or fg.G is None:
        print("Initializing missing NetworkX graph")
        fg.G = nx.Graph()

        # Reconstruct graph from variables and factors
        if hasattr(fg, 'variables') and hasattr(fg, 'factors'):
            # Add nodes
            fg.G.add_nodes_from(fg.variables)
            fg.G.add_nodes_from(fg.factors)

            # Try to reconstruct edges
            for factor in fg.factors:
                if hasattr(factor, 'connection_number'):
                    for var, dim in factor.connection_number.items():
                        fg.G.add_edge(factor, var, dim=dim)

    # Ensure all required attributes exist
    for node in fg.G.nodes():
        # Ensure mailbox exists
        if not hasattr(node, 'mailbox'):
            node.mailbox = []

        # Ensure other attributes exist based on node type
        if hasattr(node, 'type') and node.type == 'factor':
            if not hasattr(node, 'cost_table') or node.cost_table is None:
                try:
                    # Try to initialize cost table if missing
                    if hasattr(node, 'initiate_cost_table'):
                        node.initiate_cost_table()
                except Exception as e:
                    print(f"Could not initialize cost table for {node}: {e}")

    return fg


try:
    # Import all the required classes
    from bp_base.factor_graph import FactorGraph
    from bp_base.agents import VariableAgent, FactorAgent
    from bp_base.components import Message

    print(f"NetworkX version: {nx.__version__}")

    # Try to load the pickle
    pickle_path = os.path.join(project_root, 'configs', 'factor_graphs',
                               'factor-graph-cycle-3-random_intlow1,high100-number5.pkl')
    print(f"Attempting to load: {pickle_path}")

    # Check if file exists
    if not os.path.exists(pickle_path):
        print(f"File does not exist: {pickle_path}")
        # List available factor graph files
        factor_graphs_dir = os.path.join(project_root, 'configs', 'factor_graphs')
        if os.path.exists(factor_graphs_dir):
            print(f"Available factor graph files in {factor_graphs_dir}:")
            for file in os.listdir(factor_graphs_dir):
                if file.startswith('factor-graph'):
                    print(f"  - {file}")
                    # Update pickle_path to use an existing file
                    pickle_path = os.path.join(factor_graphs_dir, file)
                    print(f"Using first available file: {pickle_path}")
                    break

    if os.path.exists(pickle_path):
        # Load the factor graph
        fg = load_pickle_safely(pickle_path)

        if fg is not None:
            print(f"Graph loaded. Type: {type(fg)}")

            # Repair any issues with the loaded graph
            fg = repair_factor_graph(fg)

            # Check if the graph is usable
            print("\nFactor graph details:")
            try:
                print(f"Variables: {len(fg.variables)}")
                print(f"Factors: {len(fg.factors)}")
                print(f"Graph nodes: {len(fg.G.nodes())}")
                print(f"Graph edges: {len(fg.G.edges())}")

                # Print first few nodes
                print("\nFirst few nodes:")
                for i, node in enumerate(fg.G.nodes()):
                    if i >= 5:  # Limit to 5 nodes
                        break
                    print(f"  - {node}")

                # Try to access a variable node's attributes
                if fg.variables:
                    var = fg.variables[0]
                    print(f"\nFirst variable: {var.name}, Domain: {var.domain}")

                # Try to access a factor node's attributes
                if fg.factors:
                    factor = fg.factors[0]
                    print(f"\nFirst factor: {factor.name}")
                    if hasattr(factor, 'cost_table') and factor.cost_table is not None:
                        print(f"Cost table shape: {factor.cost_table.shape}")
            except Exception as e:
                print(f"Error inspecting graph: {e}")

            # Try to save the repaired graph
            try:
                output_path = os.path.join(os.path.dirname(pickle_path), "repaired_" + os.path.basename(pickle_path))
                with open(output_path, 'wb') as f:
                    pickle.dump(fg, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"\nRepaired graph saved to: {output_path}")
            except Exception as e:
                print(f"Error saving repaired graph: {e}")
    else:
        print("No factor graph files found.")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are installed and in your Python path.")
except Exception as e:
    print(f"Unexpected error: {e}")