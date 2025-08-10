"""
Demo script showing DSA and MGM algorithms in action.

This script demonstrates the use of the search-based algorithms DSA and MGM
on a simple constraint optimization problem.
"""

import numpy as np
import sys
import os

# Add the project root to the path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    # Try to import full functionality
    from ..bp.factor_graph import FactorGraph
    from ..core.agents import VariableAgent, FactorAgent
    from . import DSAEngine, MGMEngine, DSAComputator, MGMComputator

    FULL_FUNCTIONALITY = True
except ImportError:
    # Fallback to just computators for testing
    from . import DSAComputator, MGMComputator

    FULL_FUNCTIONALITY = False
    print("Note: Full engine functionality not available due to missing dependencies")
    print("This demo will show computator functionality only.\n")


def demo_computators_only():
    """Demo the search computators without full engine functionality."""
    print("=== DSA and MGM Computator Demo ===\n")

    # Create mock agents and factors for demonstration
    class MockAgent:
        def __init__(self, name, domain, assignment=0):
            self.name = name
            self.domain = domain
            self.curr_assignment = assignment
            self._connected_factors = []

    class MockFactor:
        def __init__(self, cost_table, connection_map):
            self.cost_table = np.array(cost_table)
            self.connection_number = connection_map

    # Create a simple binary constraint problem
    # Two variables (x1, x2) with domain {0, 1}
    # Cost table prefers (0,1) combination
    cost_table = [
        [2.0, 1.0],  # x1=0, x2=0 -> cost=2; x1=0, x2=1 -> cost=1
        [3.0, 4.0],
    ]  # x1=1, x2=0 -> cost=3; x1=1, x2=1 -> cost=4

    factor = MockFactor(cost_table, {"x1": 0, "x2": 1})

    x1 = MockAgent("x1", 2, 1)  # Start at suboptimal value
    x2 = MockAgent("x2", 2, 0)  # Start at suboptimal value

    x1._connected_factors = [factor]
    x2._connected_factors = [factor]

    print("Problem Setup:")
    print(f"Variables: x1 ∈ {{0,1}}, x2 ∈ {{0,1}}")
    print(f"Cost table: {cost_table}")
    print(f"Initial assignment: x1={x1.curr_assignment}, x2={x2.curr_assignment}")

    current_cost = cost_table[x1.curr_assignment][x2.curr_assignment]
    print(f"Initial cost: {current_cost}")
    print(f"Optimal solution: x1=0, x2=1 (cost=1)\n")

    # Demo DSA
    print("--- DSA (Distributed Stochastic Algorithm) Demo ---")
    dsa = DSAComputator(probability=1.0)  # Always take improving moves

    print("DSA behavior: Each agent independently decides whether to change")
    print("with probability p when an improvement is found.\n")

    # Simulate a few DSA steps
    for step in range(3):
        print(f"DSA Step {step + 1}:")

        # Each agent makes independent decision
        neighbors_x1 = {"x2": x2.curr_assignment}
        neighbors_x2 = {"x1": x1.curr_assignment}

        new_x1 = dsa.compute_decision(x1, neighbors_x1)
        new_x2 = dsa.compute_decision(x2, neighbors_x2)

        print(f"  x1: {x1.curr_assignment} -> {new_x1}")
        print(f"  x2: {x2.curr_assignment} -> {new_x2}")

        # Update assignments
        x1.curr_assignment = new_x1
        x2.curr_assignment = new_x2

        new_cost = cost_table[x1.curr_assignment][x2.curr_assignment]
        print(f"  New cost: {new_cost}")

        if new_cost == 1:
            print("  ✓ Optimal solution found!")
            break
        print()

    print("\n--- MGM (Maximum Gain Message) Demo ---")

    # Reset to initial state
    x1.curr_assignment = 1
    x2.curr_assignment = 0

    mgm = MGMComputator()

    print("MGM behavior: Agents coordinate to ensure only the agent")
    print("with maximum gain in its neighborhood changes.\n")

    # Simulate MGM steps
    for step in range(3):
        print(f"MGM Step {step + 1}:")

        # Phase 1: Gain calculation
        mgm.reset_phase()
        neighbors_x1 = {"x2": x2.curr_assignment}
        neighbors_x2 = {"x1": x1.curr_assignment}

        mgm.compute_decision(x1, neighbors_x1)
        mgm.compute_decision(x2, neighbors_x2)

        # Show calculated gains
        if "x1" in mgm.agent_gains:
            print(f"  x1 gain: {mgm.agent_gains['x1']['gain']}")
        if "x2" in mgm.agent_gains:
            print(f"  x2 gain: {mgm.agent_gains['x2']['gain']}")

        # Phase 2: Decision (simulate coordination)
        mgm.move_to_decision_phase()

        # Simple coordination: each agent knows its own gain and can compare
        x1_gain = mgm.agent_gains.get("x1", {}).get("gain", 0)
        x2_gain = mgm.agent_gains.get("x2", {}).get("gain", 0)

        # Determine who has maximum gain
        if x1_gain > x2_gain:
            x1.neighbor_gains = {"x2": x2_gain}
            x2.neighbor_gains = {"x1": x1_gain}
        else:
            x1.neighbor_gains = {"x2": x2_gain}
            x2.neighbor_gains = {"x1": x1_gain}

        new_x1 = mgm.compute_decision(x1, neighbors_x1)
        new_x2 = mgm.compute_decision(x2, neighbors_x2)

        print(f"  x1: {x1.curr_assignment} -> {new_x1}")
        print(f"  x2: {x2.curr_assignment} -> {new_x2}")

        # Update assignments
        x1.curr_assignment = new_x1
        x2.curr_assignment = new_x2

        new_cost = cost_table[x1.curr_assignment][x2.curr_assignment]
        print(f"  New cost: {new_cost}")

        if new_cost == 1:
            print("  ✓ Optimal solution found!")
            break
        print()


def demo_full_engines():
    """Demo the full DSA and MGM bp with factor graphs."""
    print("=== Full DSA and MGM Engine Demo ===\n")

    # Create a simple factor graph
    def create_demo_factor_graph():
        # Create variable agents
        var1 = VariableAgent(name="x1", domain=2)
        var2 = VariableAgent(name="x2", domain=2)

        # Create factor agent with cost table that prefers (0,1)
        def create_cost_table(num_vars=None, domain_size=None, **kwargs):
            return np.array([[2.0, 1.0], [3.0, 4.0]])  # Prefer x1=0, x2=1

        factor = FactorAgent(
            name="constraint_x1_x2",
            domain=2,
            ct_creation_func=create_cost_table,
        )

        # Set up connection numbers
        factor.connection_number = {"x1": 0, "x2": 1}
        factor.initiate_cost_table()

        # Create factor graph
        edges = {factor: [var1, var2]}
        fg = FactorGraph(edges)

        # Set initial suboptimal assignments
        var1.curr_assignment = 1
        var2.curr_assignment = 0

        return fg

    # Test DSA Engine
    print("--- DSA Engine Demo ---")
    fg_dsa = create_demo_factor_graph()

    print(
        f"Initial assignment: x1={fg_dsa.variables[0].curr_assignment}, x2={fg_dsa.variables[1].curr_assignment}"
    )
    print(f"Initial cost: {fg_dsa.global_cost}")

    dsa_computator = DSAComputator(probability=0.8)
    dsa_engine = DSAEngine(factor_graph=fg_dsa, computator=dsa_computator)

    print("\nRunning DSA for 5 iterations...")
    for i in range(5):
        step = dsa_engine.step(i)
        cost = dsa_engine.global_cost
        assignments = [var.curr_assignment for var in dsa_engine.var_nodes]
        print(f"  Step {i+1}: assignments={assignments}, cost={cost}")

        if cost == 1:
            print("  ✓ Optimal solution found!")
            break

    # Test MGM Engine
    print("\n--- MGM Engine Demo ---")
    fg_mgm = create_demo_factor_graph()

    print(
        f"Initial assignment: x1={fg_mgm.variables[0].curr_assignment}, x2={fg_mgm.variables[1].curr_assignment}"
    )
    print(f"Initial cost: {fg_mgm.global_cost}")

    mgm_computator = MGMComputator()
    mgm_engine = MGMEngine(factor_graph=fg_mgm, computator=mgm_computator)

    print("\nRunning MGM for 5 iterations...")
    for i in range(5):
        step = mgm_engine.step(i)
        cost = mgm_engine.global_cost
        assignments = [var.curr_assignment for var in mgm_engine.var_nodes]
        print(f"  Step {i+1}: assignments={assignments}, cost={cost}")

        if cost == 1:
            print("  ✓ Optimal solution found!")
            break


def main():
    """Main demo function."""
    print("DSA and MGM Search Algorithms Demo")
    print("=" * 50)
    print()

    if FULL_FUNCTIONALITY:
        demo_full_engines()
        print("\n" + "=" * 50)

    demo_computators_only()

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nKey differences:")
    print("- DSA: Independent, probabilistic decisions")
    print("- MGM: Coordinated decisions, only max gain agent moves")
    print("- Both aim to minimize cost through local search")


if __name__ == "__main__":
    main()
