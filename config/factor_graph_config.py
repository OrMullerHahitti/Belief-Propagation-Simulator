# Configuration for different factor graph topologies
from typing import Dict, List, Callable
import numpy as np

from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import CostTable
from bp_base.factor_graph import FactorGraph, Edges
from bp_base.computators import MaxSumComputator, MinSumComputator
from utils.ct_creator import create_random_int_table

# Common parameters
# Domain size for all variables (e.g., binary variables with values 0 or 1)
DOMAIN_SIZE = 2
# Computator to use (MaxSum or MinSum)
COMPUTATOR = MaxSumComputator()
# Cost table creation function and parameters
CT_CREATION_FUNC = create_random_int_table
CT_CREATION_PARAMS = {
    "low": 0,
    "high": 10,
}


def create_cycle_3_graph() -> FactorGraph:
    """
    Create a factor graph with a cycle of 3 variables and 3 factors in a ring structure.
    
    Structure:
        v1 -- f12 -- v2
        |            |
        f31          f23
        |            |
        v3 ----------+
        
    Returns:
        A FactorGraph instance with the cycle configuration.
    """
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


def create_cycle_4_graph() -> FactorGraph:
    """
    Create a factor graph with a cycle of 4 variables and 4 factors in a square structure.
    
    Structure:
        v1 -- f12 -- v2
        |            |
        f41          f23
        |            |
        v4 -- f34 -- v3
        
    Returns:
        A FactorGraph instance with the cycle configuration.
    """
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


def create_loop_8_graph() -> FactorGraph:
    """
    Create a factor graph with 8 variables and 8 factors in a ring structure.
    
    Structure:
        v1 -- f12 -- v2
        |            |
        f81          f23
        |            |
        v8           v3
        |            |
        f78          f34
        |            |
        v7 -- f67 -- v6 -- f56 -- v5
        
    Returns:
        A FactorGraph instance with the 8-node loop configuration.
    """
    # Create variable nodes
    variables = [
        VariableAgent(name=f"v{i}", domain=DOMAIN_SIZE, computator=COMPUTATOR)
        for i in range(1, 9)
    ]
    
    # Create factor nodes
    factors = [
        FactorAgent(
            name=f"f{i}{i%8+1}", 
            domain=DOMAIN_SIZE, 
            computator=COMPUTATOR, 
            ct_creation_func=CT_CREATION_FUNC, 
            param=CT_CREATION_PARAMS
        )
        for i in range(1, 9)
    ]
    
    # Define edges (connecting each factor to its adjacent variables)
    edges = {}
    for i in range(8):
        edges[factors[i]] = [variables[i], variables[(i+1)%8]]
    
    # Create and return the factor graph
    return FactorGraph(
        variable_li=variables,
        factor_li=factors,
        edges=edges
    )


# Choose the factor graph to use - used in other scripts
DEFAULT_GRAPH_CONFIG = "cycle_3"

# Factory function to get the specified graph
def get_factor_graph(config_name: str = DEFAULT_GRAPH_CONFIG) -> FactorGraph:
    """
    Factory function to get a factor graph based on its configuration name.
    
    Args:
        config_name: Name of the graph configuration.
            Options: 'cycle_3', 'cycle_4', 'loop_8'
    
    Returns:
        A FactorGraph instance with the specified configuration.
    
    Raises:
        ValueError: If the configuration name is not recognized.
    """
    if config_name == "cycle_3":
        return create_cycle_3_graph()
    elif config_name == "cycle_4":
        return create_cycle_4_graph()
    elif config_name == "loop_8":
        return create_loop_8_graph()
    else:
        raise ValueError(f"Unknown factor graph configuration: {config_name}")