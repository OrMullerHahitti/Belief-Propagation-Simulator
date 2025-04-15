import pytest
import numpy as np
import logging
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.factor_graph import FactorGraph
from bp_base.computators import MinSumComputator
from utils.ct_utils import create_random_int_table

# Configure logger
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logging.getLogger().setLevel(logging.INFO)
    yield

@pytest.fixture
def cycle_factor_graph():
    """Create a cycle factor graph with 3 variables and 3 factors."""
    # Create a computator for all agents
    computator = MinSumComputator()
    
    # Create 3 variable agents with domain size 3
    domain_size = 3
    x1 = VariableAgent(name="x1", domain=domain_size, computator=computator)
    x2 = VariableAgent(name="x2", domain=domain_size, computator=computator)
    x3 = VariableAgent(name="x3", domain=domain_size, computator=computator)
    
    # Create 3 factor agents
    # Each factor connects two variables in a cycle: x1-f1-x2-f2-x3-f3-x1
    ct_creation_func = create_random_int_table
    ct_params = {"low": 0, "high": 10}
    
    f1 = FactorAgent(name="f1", domain=domain_size, computator=computator, 
                    ct_creation_func=ct_creation_func, param=ct_params)
    f2 = FactorAgent(name="f2", domain=domain_size, computator=computator, 
                    ct_creation_func=ct_creation_func, param=ct_params)
    f3 = FactorAgent(name="f3", domain=domain_size, computator=computator, 
                    ct_creation_func=ct_creation_func, param=ct_params)
    
    # Define edges: each factor is connected to two variables in a cycle
    edges = {
        f1: [x1, x2],
        f2: [x2, x3],
        f3: [x3, x1]
    }
    
    # Create the factor graph
    factor_graph = FactorGraph(
        variable_li=[x1, x2, x3],
        factor_li=[f1, f2, f3],
        edges=edges
    )
    
    return factor_graph

def test_factor_graph_initialization(cycle_factor_graph):
    """Test that the factor graph is correctly initialized."""
    # Extract components from the factor graph
    G = cycle_factor_graph.G
    
    # Check the number of nodes
    assert len(G.nodes()) == 6, "Factor graph should have 6 nodes (3 variables, 3 factors)"
    
    # Check the number of edges
    assert len(G.edges()) == 6, "Factor graph should have 6 edges (each factor connected to 2 variables)"
    
    # Check that variables have proper connections
    var_nodes = [node for node in G.nodes() if isinstance(node, VariableAgent)]
    factor_nodes = [node for node in G.nodes() if isinstance(node, FactorAgent)]
    
    assert len(var_nodes) == 3, "There should be 3 variable nodes"
    assert len(factor_nodes) == 3, "There should be 3 factor nodes"
    
    # Check that each variable is connected to 2 factors
    for var in var_nodes:
        connected_factors = list(G.neighbors(var))
        assert len(connected_factors) == 2, f"Variable {var.name} should be connected to 2 factors"
    
    # Check that each factor is connected to 2 variables
    for factor in factor_nodes:
        connected_vars = list(G.neighbors(factor))
        assert len(connected_vars) == 2, f"Factor {factor.name} should be connected to 2 variables"
        
        # Check that cost tables are initialized
        assert factor.cost_table is not None, f"Cost table for factor {factor.name} should be initialized"
        assert factor.cost_table.shape == (3, 3), f"Cost table shape should be (3, 3), got {factor.cost_table.shape}"

def test_message_initialization(cycle_factor_graph):
    """Test that the mailbox is correctly initialized."""
    G = cycle_factor_graph.G
    
    # Check each edge has mailbox entries
    for node in G.nodes():
        assert hasattr(node, 'mailbox'), f"Node {node.name} should have a mailbox"
        assert len(node.mailbox) == 2, f"Node {node.name} should have 2 messages in mailbox (one for each connection)"

def test_one_step_message_passing(cycle_factor_graph):
    """Test one step of message passing in the factor graph."""
    # Extract variables and factors from the graph
    G = cycle_factor_graph.G
    var_nodes = [node for node in G.nodes() if isinstance(node, VariableAgent)]
    factor_nodes = [node for node in G.nodes() if isinstance(node, FactorAgent)]
    
    # Manually assign some concrete cost tables for clearer testing
    for factor in factor_nodes:
        # Override the random cost tables with known values
        factor.cost_table = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        # Log the cost table
        logger.info(f"Factor {factor.name} cost table:\n{factor.cost_table}")
    
    # Step 1: Compute messages for all factor nodes first
    logger.info("Computing messages for factor nodes")
    for factor in factor_nodes:
        # Factor agents need to use compute_R with the cost table
        factor.messages_to_send = factor.computator.compute_R(
            factor.cost_table, 
            factor.mailbox
        )
        
        # Log the computed messages
        logger.info(f"Factor {factor.name} computed {len(factor.messages_to_send)} messages:")
        for i, msg in enumerate(factor.messages_to_send):
            logger.info(f"  Message {i+1}: {msg}")
    
    # Step 2: Send factor messages to variables
    logger.info("Sending factor messages to variables")
    for factor in factor_nodes:
        for message in factor.messages_to_send:
            message.recipient.receive_message(message)
    
    # Step 3: Compute messages for all variable nodes
    logger.info("Computing messages for variable nodes")
    for var in var_nodes:
        # Variable agents use compute_Q with their mailbox
        var.messages_to_send = var.computator.compute_Q(var.mailbox)
        
        # Log the computed messages
        logger.info(f"Variable {var.name} computed {len(var.messages_to_send)} messages:")
        for i, msg in enumerate(var.messages_to_send):
            logger.info(f"  Message {i+1}: {msg}")
    
    # Step 4: Send variable messages to factors
    logger.info("Sending variable messages to factors")
    for var in var_nodes:
        for message in var.messages_to_send:
            message.recipient.receive_message(message)
    
    # Step 5: Verify messages were received correctly
    for node in G.nodes():
        logger.info(f"Node {node.name} mailbox now contains {len(node.mailbox)} messages")
        for i, msg in enumerate(node.mailbox):
            logger.info(f"  Message {i+1}: {msg}")
        
        # Each node should have exactly 2 messages in mailbox after all the sends
        assert len(node.mailbox) == 2, f"Node {node.name} should have 2 messages in mailbox after message passing"

    # Step 6: Verify specific message contents for a sample variable and factor
    var1 = next(var for var in var_nodes if var.name == "x1")
    factor1 = next(fac for fac in factor_nodes if fac.name == "f1")
    
    # Check that var1 received a message from factor1
    var1_msgs = [msg for msg in var1.mailbox if msg.sender == factor1]
    assert len(var1_msgs) == 1, "Should find exactly 1 message from factor1 to var1"
    
    # Check shape of messages
    assert var1_msgs[0].data.shape == (3,), f"Message data should have shape (3,), got {var1_msgs[0].data.shape}"
    
    # Check that factor1 received a message from var1
    factor1_msgs = [msg for msg in factor1.mailbox if msg.sender == var1]
    assert len(factor1_msgs) == 1, "Should find exactly 1 message from var1 to factor1"
    
    # Check shape of messages
    assert factor1_msgs[0].data.shape == (3,), f"Message data should have shape (3,), got {factor1_msgs[0].data.shape}"
    
    logger.info("One step of message passing completed and verified")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Running tests directly")
    pytest.main(["-xvs", "--log-cli-level=INFO", __file__])
