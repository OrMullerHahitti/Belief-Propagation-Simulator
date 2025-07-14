"""
Tests for search-based bp (DSA and MGM).
"""

import numpy as np
import sys

from src.propflow.bp.factor_graph import FactorGraph
from src.propflow.core import VariableAgent, FactorAgent
from src.propflow.search import DSAEngine, MGMEngine, DSAComputator, MGMComputator

# Flag to enable verbose output during tests
VERBOSE = True


def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)
        sys.stdout.flush()


def create_simple_search_factor_graph():
    """Create a simple factor graph for search testing."""
    # Create variable agents
    var1 = VariableAgent(name="var1", domain=2)
    var2 = VariableAgent(name="var2", domain=2)

    # Create factor agent with a simple cost table
    def create_test_cost_table(num_vars=None, domain_size=None, **kwargs):
        # Simple cost table that encourages var1=0, var2=1
        return np.array([[2.0, 1.0], [3.0, 4.0]])  # [var1][var2]

    factor = FactorAgent(
        name="factor",
        domain=2,
        ct_creation_func=create_test_cost_table,
    )

    # Set up connection numbers
    factor.connection_number = {"var1": 0, "var2": 1}

    # Initialize the cost table
    factor.initiate_cost_table()

    # Create factor graph
    edges = {factor: [var1, var2]}
    factor_graph = FactorGraph(edges)

    return factor_graph


def test_dsa_computator():
    """Test DSA computator functionality."""
    verbose_print("\n=== Testing DSAComputator ===")

    computator = DSAComputator(probability=1.0)  # Always change if improvement found

    # Create a mock agent
    class MockAgent:
        def __init__(self, name, domain, assignment=0):
            self.name = name
            self.domain = domain
            self.curr_assignment = assignment
            self._connected_factors = []

    # Create mock factor with cost table
    class MockFactor:
        def __init__(self):
            self.cost_table = np.array(
                [[2.0, 1.0], [3.0, 4.0]]
            )  # Prefer var1=0, var2=1
            self.connection_number = {"agent1": 0, "agent2": 1}

    factor = MockFactor()
    agent = MockAgent("agent1", 2, 0)
    agent._connected_factors = [factor]

    # Test cost evaluation
    neighbors = {"agent2": 0}  # neighbor has value 0

    cost_0 = computator.evaluate_cost(agent, 0, neighbors)  # Cost of agent1=0, agent2=0
    cost_1 = computator.evaluate_cost(agent, 1, neighbors)  # Cost of agent1=1, agent2=0

    verbose_print(f"Cost for agent1=0: {cost_0}")
    verbose_print(f"Cost for agent1=1: {cost_1}")

    assert cost_0 == 2.0  # From cost_table[0][0]
    assert cost_1 == 3.0  # From cost_table[1][0]

    # Test decision making
    decision = computator.compute_decision(agent, neighbors)
    # With probability=1.0, should prefer the better value (0 over 1)
    # Since agent is already at 0 and that's better, should stay at 0
    assert decision == 0

    verbose_print("✓ DSAComputator tests passed")


def test_mgm_computator():
    """Test MGM computator functionality."""
    verbose_print("\n=== Testing MGMComputator ===")

    computator = MGMComputator()

    # Create a mock agent
    class MockAgent:
        def __init__(self, name, domain, assignment=0):
            self.name = name
            self.domain = domain
            self.curr_assignment = assignment
            self._connected_factors = []
            self.neighbor_gains = {}

    # Create mock factor
    class MockFactor:
        def __init__(self):
            self.cost_table = np.array([[2.0, 1.0], [3.0, 4.0]])
            self.connection_number = {"agent1": 0, "agent2": 1}

    factor = MockFactor()
    agent = MockAgent("agent1", 2, 1)  # Start at suboptimal value
    agent._connected_factors = [factor]

    # Test gain calculation phase
    neighbors = {"agent2": 0}

    # Phase 1: Gain calculation
    assert computator.phase == "gain_calculation"
    decision = computator.compute_decision(agent, neighbors)
    assert decision is None  # No decision in gain calculation phase

    # Check that gain was calculated correctly
    assert "agent1" in computator.agent_gains
    gain_info = computator.agent_gains["agent1"]
    expected_gain = 3.0 - 2.0  # Cost of value 1 - cost of value 0 = improvement
    assert gain_info["gain"] == expected_gain
    assert gain_info["best_value"] == 0

    # Phase 2: Decision phase with maximum gain
    computator.move_to_decision_phase()
    agent.neighbor_gains = {}  # No neighbor competition
    decision = computator.compute_decision(agent, neighbors)
    assert decision == 0  # Should choose the better value

    verbose_print("✓ MGMComputator tests passed")


def test_dsa_engine():
    """Test DSA engine initialization and basic functionality."""
    verbose_print("\n=== Testing DSAEngine ===")

    # Create factor graph
    fg = create_simple_search_factor_graph()

    # Create DSA engine
    computator = DSAComputator(probability=0.5)
    engine = DSAEngine(factor_graph=fg, computator=computator)

    # Verify engine is initialized correctly
    assert engine is not None
    assert engine.graph == fg
    assert isinstance(engine.computator, DSAComputator)

    # Verify variable agents were extended
    for var in engine.var_nodes:
        assert hasattr(var, "compute_search_step")
        assert hasattr(var, "update_assignment")
        assert hasattr(var, "get_neighbor_values")
        assert hasattr(var, "_connected_factors")

    verbose_print("✓ DSAEngine initialization tests passed")


def test_mgm_engine():
    """Test MGM engine initialization and basic functionality."""
    verbose_print("\n=== Testing MGMEngine ===")

    # Create factor graph
    fg = create_simple_search_factor_graph()

    # Create MGM engine
    computator = MGMComputator()
    engine = MGMEngine(factor_graph=fg, computator=computator)

    # Verify engine is initialized correctly
    assert engine is not None
    assert engine.graph == fg
    assert isinstance(engine.computator, MGMComputator)

    # Verify variable agents were extended
    for var in engine.var_nodes:
        assert hasattr(var, "compute_search_step")
        assert hasattr(var, "update_assignment")
        assert hasattr(var, "get_neighbor_values")
        assert hasattr(var, "_connected_factors")
        assert hasattr(var, "neighbor_gains")

    verbose_print("✓ MGMEngine initialization tests passed")


def test_dsa_step_execution():
    """Test DSA engine step execution."""
    verbose_print("\n=== Testing DSA Step Execution ===")

    # Create factor graph
    fg = create_simple_search_factor_graph()

    # Set initial assignments
    fg.variables[0].curr_assignment = 1  # Suboptimal
    fg.variables[1].curr_assignment = 0  # Suboptimal

    # Create DSA engine with high probability to ensure changes
    computator = DSAComputator(probability=1.0)
    engine = DSAEngine(factor_graph=fg, computator=computator)

    # Record initial cost
    initial_cost = engine.global_cost
    verbose_print(f"Initial cost: {initial_cost}")

    # Execute one step
    step = engine.step(0)

    # Verify step was created
    assert step is not None

    # Check if cost improved (may not always improve due to randomness)
    final_cost = engine.global_cost
    verbose_print(f"Final cost: {final_cost}")

    # At minimum, should track costs
    assert len(engine.history.costs) > 0

    verbose_print("✓ DSA step execution tests passed")


def test_mgm_step_execution():
    """Test MGM engine step execution."""
    verbose_print("\n=== Testing MGM Step Execution ===")

    # Create factor graph
    fg = create_simple_search_factor_graph()

    # Set initial assignments to suboptimal values
    fg.variables[0].curr_assignment = 1  # Should prefer 0
    fg.variables[1].curr_assignment = 0  # Should prefer 1

    # Create MGM engine
    computator = MGMComputator()
    engine = MGMEngine(factor_graph=fg, computator=computator)

    # Record initial cost
    initial_cost = engine.global_cost
    verbose_print(f"Initial cost: {initial_cost}")

    # Execute one step
    step = engine.step(0)

    # Verify step was created
    assert step is not None

    # Check final cost
    final_cost = engine.global_cost
    verbose_print(f"Final cost: {final_cost}")

    # Should track costs
    assert len(engine.history.costs) > 0

    verbose_print("✓ MGM step execution tests passed")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_dsa_computator()
    test_mgm_computator()
    test_dsa_engine()
    test_mgm_engine()
    test_dsa_step_execution()
    test_mgm_step_execution()
    verbose_print("\n🎉 All search engine tests passed!")
