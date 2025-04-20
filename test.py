import pickle
import sys
import os
from pathlib import Path


# Function to find the project root
def find_project_root():
    current_dir = Path.cwd()
    while True:
        if any((current_dir / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml']):
            return current_dir
        if current_dir == current_dir.parent:
            return Path.cwd()
        current_dir = current_dir.parent


# Add project root to path
project_root = find_project_root()
sys.path.append(str(project_root))

# Import required classes
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
import networkx as nx

# Try to load the pickle
pickle_path = os.path.join(project_root, 'configs', 'factor_graphs',
                           'factor-graph-cycle-3-random_intlow1,high100-number5.pkl')
print(f"Attempting to load: {pickle_path}")

try:
    with open(pickle_path, 'rb') as f:
        fg = pickle.load(f)
    print(f"Graph loaded successfully: {type(fg)}")
    print(f"Number of nodes: {len(fg.G.nodes())}")

    # Print variables and factors to verify
    print("\nVariables:")
    for var in fg.variables:
        print(f"  - {var.name}")

    print("\nFactors:")
    for factor in fg.factors:
        print(f"  - {factor.name}")

    # Test accessing node attributes
    print("\nTesting node attribute access:")
    for node in list(fg.G.nodes())[:2]:  # Just check a couple
        print(f"  - Node {node.name}, type: {node.type}")

    print("\nPickle loaded and verified successfully!")

except Exception as e:
    print(f"Error loading pickle: {e}")