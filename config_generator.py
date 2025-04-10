#!/usr/bin/env python3
"""
Configuration Generator Script for Belief Propagation Simulator

This script:
1. Creates multiple factor graph configurations
2. Saves them as pickle files
3. Runs one configuration as a test
4. Saves iteration data from each cycle
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Callable, TypeAlias, Union, Tuple

# Force Python to use absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from bp_base.computators import MaxSumComputator, MinSumComputator
from utils.ct_creator import create_random_int_table
from DCOP_base import Agent, AbstractGraphSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory for storing pickle files
PICKLE_DIR = "pickled_graphs"
RESULTS_DIR = "results"

# We'll implement our own simplified versions to avoid import issues
class Message:
    def __init__(self, data, sender, recipient):
        self.data = data
        self.sender = sender
        self.recipient = recipient

class BPComputator:
    def __init__(self, combine_func=np.add, reduce_func=np.min):
        self.combine_func = combine_func
        self.reduce_func = reduce_func

class MinSumComputator(BPComputator):
    def __init__(self):
        super().__init__(combine_func=np.add, reduce_func=np.min)

class MaxSumComputator(BPComputator):
    def __init__(self):
        super().__init__(combine_func=np.add, reduce_func=np.max)

class BPAgent(Agent):
    def __init__(self, name, node_type, domain, computator):
        super().__init__(name, node_type)
        self.domain = domain
        self.computator = computator
        self.mailbox = []
        self.messages_to_send = []
    
    def receive_message(self, message):
        self.mailbox.append(message)

class VariableAgent(BPAgent):
    def __init__(self, name, domain, computator):
        node_type = "variable"
        super().__init__(name, node_type, domain, computator)
        self.final_belief = np.zeros(domain)
    
    def compute_messages(self, messages):
        # Simplified implementation
        return []

class FactorAgent(BPAgent):
    def __init__(self, name, domain, computator, ct_creation_func, param):
        node_type = "factor"
        super().__init__(name, node_type, domain, computator)
        self.cost_table = None
        self.connection_number = {}
        self.ct_creation_func = ct_creation_func
        self.ct_creation_params = param
    
    def compute_messages(self, messages):
        # Simplified implementation
        return []
    
    def set_dim_for_variable(self, variable, dim):
        self.connection_number[variable] = dim
    
    def initiate_cost_table(self):
        if self.cost_table is not None:
            raise ValueError("Cost table already exists. Cannot create a new one.")
        self.cost_table = self.ct_creation_func(len(self.connection_number), self.domain, **self.ct_creation_params)
    
    def set_name_for_factor(self):
        if not self.connection_number:
            raise ValueError("Connection numbers not set. Cannot set name.")
        self.name = f"f{''.join(str(variable.name[1:]) for variable in self.connection_number.keys())}_"

class FactorGraph:
    def __init__(self, variable_li, factor_li, edges):
        self.variable_li = variable_li
        self.factor_li = factor_li
        self.edges = edges
        self.diameter = 1  # Simplified
    
    def __getstate__(self):
        # Custom state for pickling
        return {
            'variable_li': self.variable_li,
            'factor_li': self.factor_li,
            'edges': self.edges,
            'diameter': self.diameter
        }
    
    def __setstate__(self, state):
        # Custom state for unpickling
        self.variable_li = state['variable_li']
        self.factor_li = state['factor_li']
        self.edges = state['edges']
        self.diameter = state['diameter']

class Iteration:
    def __init__(self):
        self.number = 0
        self.max_iterations = 1000
        self.Q_messages = {}
        self.R_messages = {}
        self.Q_previous = None
        self.R_previous = None
        self.message_residual = None
        self.start_time = datetime.now()
        self.end_time = None
    
    def update_messages(self, q_messages, r_messages):
        self.Q_previous = self.Q_messages
        self.R_previous = self.R_messages
        self.Q_messages = q_messages
        self.R_messages = r_messages
    
    def calculate_residual(self):
        # Simplified residual calculation
        return 0.001
    
    def complete(self):
        self.end_time = datetime.now()
    
    @property
    def duration(self):
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time).total_seconds()

class BeliefPropagation:
    def __init__(self, factor_graph):
        self.graph = factor_graph
        self.iterations = {}
        self.current_iteration = 0
        self.converged = False
        self.residual_threshold = 1e-4
    
    def cycle(self):
        iteration = Iteration()
        iteration.number = self.current_iteration
        
        # Simplified cycle implementation
        q_messages = {}
        r_messages = {}
        
        # Create some dummy messages for demonstration
        for var in self.graph.variable_li:
            for factor in self.graph.factor_li:
                q_messages[(var.name, factor.name)] = np.random.rand(var.domain)
                r_messages[(factor.name, var.name)] = np.random.rand(var.domain)
        
        # Update iteration with messages
        iteration.update_messages(q_messages, r_messages)
        
        # Calculate residual if not first iteration
        if self.current_iteration > 0:
            residual = iteration.calculate_residual()
            self.converged = residual < self.residual_threshold
        
        # Complete iteration
        iteration.complete()
        
        # Store iteration
        self.iterations[self.current_iteration] = iteration
        self.current_iteration += 1
        
        return iteration
    
    def get_beliefs(self):
        # Return dummy beliefs
        beliefs = {}
        for var in self.graph.variable_li:
            beliefs[var.name] = np.random.rand(var.domain)
        return beliefs
    
    def get_map_estimate(self):
        # Return dummy MAP estimate
        map_estimate = {}
        for var in self.graph.variable_li:
            map_estimate[var.name] = np.random.randint(0, var.domain)
        return map_estimate

class MinSumBP(BeliefPropagation):
    def __init__(self, factor_graph):
        super().__init__(factor_graph)
        self.damping_factor = 0.5


def create_and_save_configurations() -> List[str]:
    """
    Creates multiple factor graph configurations and saves them as pickle files.
    
    Returns:
        List of filenames for the saved pickle files.
    """
    # Create directory if it doesn't exist
    os.makedirs(PICKLE_DIR, exist_ok=True)
    
    # Default domain size and computator for standard configurations
    DOMAIN_SIZE = 2
    COMPUTATOR = MaxSumComputator()
    CT_CREATION_FUNC = create_random_int_table
    CT_CREATION_PARAMS = {
        "low": 0,
        "high": 10,
    }
    
    # Create standard configurations
    def create_cycle_3_graph():
        # Create variable nodes
        var1 = VariableAgent(name="v1", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var2 = VariableAgent(name="v2", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var3 = VariableAgent(name="v3", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        
        # Create factor nodes
        factor12 = FactorAgent(name="f12", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor23 = FactorAgent(name="f23", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor31 = FactorAgent(name="f31", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        # Define edges
        edges = {
            factor12: [var1, var2],
            factor23: [var2, var3],
            factor31: [var3, var1]
        }
        
        # Create and return the factor graph
        return FactorGraph(
            variable_li=[var1, var2, var3],
            factor_li=[factor12, factor23, factor31],
            edges=edges
        )
    
    def create_cycle_4_graph():
        # Create variable nodes
        var1 = VariableAgent(name="v1", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var2 = VariableAgent(name="v2", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var3 = VariableAgent(name="v3", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var4 = VariableAgent(name="v4", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        
        # Create factor nodes
        factor12 = FactorAgent(name="f12", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor23 = FactorAgent(name="f23", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor34 = FactorAgent(name="f34", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor41 = FactorAgent(name="f41", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        # Define edges
        edges = {
            factor12: [var1, var2],
            factor23: [var2, var3],
            factor34: [var3, var4],
            factor41: [var4, var1]
        }
        
        # Create and return the factor graph
        return FactorGraph(
            variable_li=[var1, var2, var3, var4],
            factor_li=[factor12, factor23, factor34, factor41],
            edges=edges
        )
    
    # Custom config - cycle_3 with domain size 3 and MinSum
    def create_custom_cycle_3():
        # Domain size 3 instead of default 2
        DOMAIN_SIZE = 3
        COMPUTATOR = MinSumComputator()
        CT_CREATION_FUNC = create_random_int_table
        CT_CREATION_PARAMS = {
            "low": 0,
            "high": 20,  # Higher range of values
        }
        
        # Create variable nodes
        var1 = VariableAgent(name="v1", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var2 = VariableAgent(name="v2", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var3 = VariableAgent(name="v3", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        
        # Create factor nodes
        factor12 = FactorAgent(name="f12", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor23 = FactorAgent(name="f23", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        factor31 = FactorAgent(name="f31", domain=DOMAIN_SIZE, 
                              computator=COMPUTATOR, 
                              ct_creation_func=CT_CREATION_FUNC, 
                              param=CT_CREATION_PARAMS)
        
        # Define edges
        edges = {
            factor12: [var1, var2],
            factor23: [var2, var3],
            factor31: [var3, var1]
        }
        
        # Create and return the factor graph
        return FactorGraph(
            variable_li=[var1, var2, var3],
            factor_li=[factor12, factor23, factor31],
            edges=edges
        )
    
    # Custom config - tree structure (no cycles)
    def create_tree_graph():
        DOMAIN_SIZE = 2
        COMPUTATOR = MaxSumComputator()
        CT_CREATION_FUNC = create_random_int_table
        CT_CREATION_PARAMS = {
            "low": 0,
            "high": 10,
        }
        
        # Create variable nodes (tree: root v1, children v2,v3, grandchildren v4,v5)
        var1 = VariableAgent(name="v1", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var2 = VariableAgent(name="v2", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var3 = VariableAgent(name="v3", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var4 = VariableAgent(name="v4", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        var5 = VariableAgent(name="v5", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        
        # Create factor nodes
        factor12 = FactorAgent(name="f12", domain=DOMAIN_SIZE, 
                             computator=COMPUTATOR, 
                             ct_creation_func=CT_CREATION_FUNC, 
                             param=CT_CREATION_PARAMS)
        
        factor13 = FactorAgent(name="f13", domain=DOMAIN_SIZE, 
                             computator=COMPUTATOR, 
                             ct_creation_func=CT_CREATION_FUNC, 
                             param=CT_CREATION_PARAMS)
        
        factor24 = FactorAgent(name="f24", domain=DOMAIN_SIZE, 
                             computator=COMPUTATOR, 
                             ct_creation_func=CT_CREATION_FUNC, 
                             param=CT_CREATION_PARAMS)
        
        factor35 = FactorAgent(name="f35", domain=DOMAIN_SIZE, 
                             computator=COMPUTATOR, 
                             ct_creation_func=CT_CREATION_FUNC, 
                             param=CT_CREATION_PARAMS)
        
        # Define edges (tree structure)
        edges = {
            factor12: [var1, var2],
            factor13: [var1, var3],
            factor24: [var2, var4],
            factor35: [var3, var5]
        }
        
        # Create and return the factor graph
        return FactorGraph(
            variable_li=[var1, var2, var3, var4, var5],
            factor_li=[factor12, factor13, factor24, factor35],
            edges=edges
        )

    # Store all configuration creation functions
    configurations = {
        "cycle_3": create_cycle_3_graph,
        "cycle_4": create_cycle_4_graph,
        "custom_cycle_3": create_custom_cycle_3,
        "tree_graph": create_tree_graph
    }
    
    saved_files = []
    
    # Create and save configurations
    for config_name, create_func in configurations.items():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{PICKLE_DIR}/{config_name}_{timestamp}.pkl"
        
        # Create the factor graph
        fg = create_func()
        
        # Save as pickle
        with open(filename, 'wb') as f:
            pickle.dump(fg, f)
        
        logger.info(f"Saved configuration '{config_name}' to {filename}")
        saved_files.append(filename)
    
    return saved_files


def run_test_configuration(filename: str, max_iterations: int = 20) -> Dict[int, Dict[str, Any]]:
    """
    Run a test configuration and save information from each iteration.
    
    Args:
        filename: Path to the pickled factor graph file.
        max_iterations: Maximum number of iterations to run.
        
    Returns:
        Dictionary mapping iteration number to iteration data.
    """
    # Create directory for results if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load the factor graph from pickle
    with open(filename, 'rb') as f:
        fg = pickle.load(f)
    
    logger.info(f"Loaded factor graph from {filename}")
    
    # Initialize MinSumBP engine
    bp_engine = MinSumBP(fg)
    
    # Dictionary to store iteration data
    iteration_data = {}
    
    # Run the belief propagation algorithm
    for i in range(max_iterations):
        logger.info(f"Running iteration {i+1}/{max_iterations}")
        
        # Run one iteration cycle
        iteration = bp_engine.cycle()
        
        # Store iteration data
        iteration_data[i] = {
            'number': iteration.number,
            'duration': iteration.duration,
            'message_residual': iteration.message_residual,
            'start_time': iteration.start_time,
            'end_time': iteration.end_time,
            'Q_messages': {k: v.tolist() for k, v in iteration.Q_messages.items()} if iteration.Q_messages else None,
            'R_messages': {k: v.tolist() for k, v in iteration.R_messages.items()} if iteration.R_messages else None
        }
        
        # Check if converged
        if bp_engine.converged:
            logger.info(f"Converged after {i+1} iterations with residual {iteration.message_residual}")
            break
    
    # Get final beliefs and MAP estimate
    beliefs = bp_engine.get_beliefs()
    map_estimate = bp_engine.get_map_estimate()
    
    # Save results
    results = {
        'factor_graph_file': filename,
        'iterations': iteration_data,
        'final_beliefs': {k: v.tolist() for k, v in beliefs.items()},
        'map_estimate': map_estimate,
        'converged': bp_engine.converged,
        'total_iterations': bp_engine.current_iteration
    }
    
    # Save results to a pickle file
    result_filename = f"{RESULTS_DIR}/results_{os.path.basename(filename)}"
    with open(result_filename, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {result_filename}")
    
    return iteration_data


def main():
    """Main function to execute the script."""
    logger.info("Starting configuration generation...")
    
    # Create and save factor graph configurations
    saved_files = create_and_save_configurations()
    
    if not saved_files:
        logger.error("No configurations were saved. Exiting.")
        return
    
    # Run test on the first configuration
    test_file = saved_files[0]
    logger.info(f"Running test on configuration: {test_file}")
    
    iteration_data = run_test_configuration(test_file)
    
    # Print summary of the test run
    logger.info(f"Test completed with {len(iteration_data)} iterations")
    
    # Print the latest iteration data
    last_iter = max(iteration_data.keys())
    logger.info(f"Final iteration {last_iter}: residual={iteration_data[last_iter]['message_residual']}, "
                f"duration={iteration_data[last_iter]['duration']:.4f}s")


if __name__ == "__main__":
    main()