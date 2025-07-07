"""
Basic tests for search computators without external dependencies.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search.search_computator import DSAComputator, MGMComputator

# Flag to enable verbose output during tests
VERBOSE = True


def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)
        sys.stdout.flush()


def test_dsa_computator_basic():
    """Test DSA computator basic functionality."""
    verbose_print("\n=== Testing DSAComputator Basic ===")
    
    computator = DSAComputator(probability=1.0)  # Always change if improvement found
    assert computator.probability == 1.0
    
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
            self.cost_table = np.array([[2.0, 1.0], [3.0, 4.0]])  # Prefer (0,1)
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
    
    verbose_print("✓ DSAComputator basic tests passed")


def test_mgm_computator_basic():
    """Test MGM computator basic functionality."""
    verbose_print("\n=== Testing MGMComputator Basic ===")
    
    computator = MGMComputator()
    assert computator.phase == "gain_calculation"
    
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
    assert gain_info['gain'] == expected_gain
    assert gain_info['best_value'] == 0
    
    verbose_print("✓ MGMComputator basic tests passed")


def test_phase_transitions():
    """Test MGM phase transitions."""
    verbose_print("\n=== Testing MGM Phase Transitions ===")
    
    computator = MGMComputator()
    
    # Start in gain calculation
    assert computator.phase == "gain_calculation"
    
    # Move to decision phase
    computator.move_to_decision_phase()
    assert computator.phase == "decision"
    
    # Reset phase
    computator.reset_phase()
    assert computator.phase == "gain_calculation"
    assert len(computator.agent_gains) == 0
    
    verbose_print("✓ MGM phase transition tests passed")


def test_cost_evaluation_edge_cases():
    """Test cost evaluation with edge cases."""
    verbose_print("\n=== Testing Cost Evaluation Edge Cases ===")
    
    computator = DSAComputator()
    
    class MockAgent:
        def __init__(self, name, domain, assignment=0):
            self.name = name
            self.domain = domain
            self.curr_assignment = assignment
            self._connected_factors = []
    
    agent = MockAgent("agent1", 2, 0)
    
    # Test with no connected factors
    cost = computator.evaluate_cost(agent, 0, {})
    assert cost == 0.0
    
    # Test with empty factors list
    agent._connected_factors = []
    cost = computator.evaluate_cost(agent, 1, {"other": 0})
    assert cost == 0.0
    
    verbose_print("✓ Cost evaluation edge case tests passed")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_dsa_computator_basic()
    test_mgm_computator_basic()
    test_phase_transitions()
    test_cost_evaluation_edge_cases()
    verbose_print("\n🎉 All basic search tests passed!")