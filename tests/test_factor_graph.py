import pytest
import numpy as np
import os
import logging
import networkx as nx
from datetime import datetime
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.computators import MaxSumComputator
from utils.ct_utils import create_random_int_table

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"factor_graph_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logger
logger = logging.getLogger("factor_graph_tests")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_format)
logger.addHandler(console_handler)

global_config = {
    'domain': 2,
    'computator': MaxSumComputator,
}

dict = {
    'low': 0,
    'high': 10
}
global_factor={
    'domain': 2,
    'computator': MaxSumComputator,
    'ct_creation_func':  create_random_int_table,
    'param': dict
}

def log_graph_structure(fg, test_name):
    """Log the structure of the factor graph"""
    logger.info(f"==== {test_name} - Factor Graph Structure ====")
    
    # Log nodes
    variable_nodes = [n for n in fg.G.nodes() if isinstance(n, VariableAgent)]
    factor_nodes = [n for n in fg.G.nodes() if isinstance(n, FactorAgent)]
    
    logger.info(f"Variable nodes: {[v.name for v in variable_nodes]}")
    logger.info(f"Factor nodes: {[f.name for f in factor_nodes]}")
    
    # Log edges
    logger.info("Edges:")
    for e in fg.G.edges():
        logger.info(f"  {e[0].name} <-> {e[1].name}")
    
    # Log cost tables
    for f in factor_nodes:
        logger.info(f"Cost table for {f.name}:")
        logger.info(f"  Shape: {f.cost_table.shape}")
        logger.info(f"  Content:\n{f.cost_table}")

def test_factor_graph_initialization():
    logger.info("=== Starting test_factor_graph_initialization ===")
    
    # Create variable nodes
    var1 = VariableAgent(name="v1",**global_config)
    var2 = VariableAgent(name="v2",**global_config)
    var3 = VariableAgent(name="v3",**global_config)

    # Create factor nodes
    factor12 = FactorAgent(name="f12", **global_factor)
    factor23 = FactorAgent(name="f23", **global_factor)

    # Create edges
    edges = {
        factor12: [var1, var2],
        factor23: [var2, var3]
    }

    # Initialize FactorGraph
    fg = FactorGraph(variable_li=[var1, var2, var3], factor_li=[factor12, factor23], edges=edges)
    
    # Log graph structure
    log_graph_structure(fg, "test_factor_graph_initialization")
    
    # Assert nodes and edges
    assert set(fg.G.nodes()) == {var1, var2, var3, factor12, factor23}

    # Assert edges without considering order
    expected_edges = {(factor12, var1), (factor12, var2), (factor23, var2), (factor23, var3)}
    actual_edges = set(fg.G.edges())
    assert all(edge in actual_edges or tuple(reversed(edge)) in actual_edges for edge in expected_edges)
    
    logger.info("test_factor_graph_initialization: PASSED")

def test_add_edges():
    logger.info("=== Starting test_add_edges ===")
    
    var1 = VariableAgent(name="v1", **global_config)
    var2 = VariableAgent(name="v2", **global_config)
    factor12 = FactorAgent(name="f12", **global_factor)
    edges = {factor12: [var1, var2]}

    fg = FactorGraph(variable_li=[var1, var2], factor_li=[factor12], edges=edges)
    
    # Log graph structure
    log_graph_structure(fg, "test_add_edges")

    # Check if edges are added correctly
    assert fg.G.has_edge(factor12, var1)
    assert fg.G.has_edge(factor12, var2)
    
    logger.info("test_add_edges: PASSED")

def test_initialize_mailbox():
    logger.info("=== Starting test_initialize_mailbox ===")
    
    var1 = VariableAgent(name="v1", **global_config)
    var2 = VariableAgent(name="v2", **global_config)
    factor12 = FactorAgent(name="f12", **global_factor)
    edges = {factor12: [var1, var2]}

    fg = FactorGraph(variable_li=[var1, var2], factor_li=[factor12], edges=edges)
    
    # Log graph structure
    log_graph_structure(fg, "test_initialize_mailbox")

    # Check if mailboxes are initialized as empty lists
    for node in fg.G.nodes():
        logger.info(f"Checking mailbox for node {node.name}")
        assert hasattr(node, 'mailbox')
        assert isinstance(node.mailbox, list)
        assert len(node.mailbox) > 0  # Should have initialized messages
    
    # Check that messages_to_send is initialized as an empty list
    for node in fg.G.nodes():
        logger.info(f"Checking messages_to_send for node {node.name}")
        assert hasattr(node, 'messages_to_send')
        assert isinstance(node.messages_to_send, list)  # Fixed typo here
        assert node.messages_to_send == []
    
    logger.info("test_initialize_mailbox: PASSED")

def test_initialize_cost_table():
    logger.info("=== Starting test_initialize_cost_table ===")
    
    var1 = VariableAgent(name="v1", **global_config)
    var2 = VariableAgent(name="v2", **global_config)
    factor12 = FactorAgent(name="f12", **global_factor)
    edges = {factor12: [var1, var2]}

    fg = FactorGraph(variable_li=[var1, var2], factor_li=[factor12], edges=edges)
    
    # Log graph structure
    log_graph_structure(fg, "test_initialize_cost_table")

    # Check if cost table is initialized
    assert factor12.cost_table is not None
    assert factor12.cost_table.shape == (2, 2)  # Based on 2 variables with domain size 2
    
    logger.info(f"Cost table for {factor12.name}: \n{factor12.cost_table}")
    logger.info("test_initialize_cost_table: PASSED")
