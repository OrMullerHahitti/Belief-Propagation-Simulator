"""
Test script for DiffusionEngine.

This script demonstrates the DiffusionEngine with different alpha values,
comparing it against pure BP and DampingEngine.

Diffusion smooths messages spatially (across neighbors), unlike damping
which smooths messages temporally (across iterations).
"""

import numpy as np
from propflow import (
    FactorGraph,
    VariableAgent,
    FactorAgent,
    DiffusionEngine,
    DampingEngine,
    BPEngine,
    MinSumComputator,
)
from propflow.configs import create_random_int_table


def create_cycle_graph():
    """Create a simple 3-node cycle graph for testing."""
    # Create variables
    var1 = VariableAgent(name="x1", domain=3)
    var2 = VariableAgent(name="x2", domain=3)
    var3 = VariableAgent(name="x3", domain=3)

    # Create factors with random cost tables
    factor12 = FactorAgent(
        name="f12",
        domain=3,
        ct_creation_func=create_random_int_table,
        param={"low": 0, "high": 10},
    )
    factor23 = FactorAgent(
        name="f23",
        domain=3,
        ct_creation_func=create_random_int_table,
        param={"low": 0, "high": 10},
    )
    factor31 = FactorAgent(
        name="f31",
        domain=3,
        ct_creation_func=create_random_int_table,
        param={"low": 0, "high": 10},
    )

    # Define edges (factor -> variables)
    edges = {
        factor12: [var1, var2],
        factor23: [var2, var3],
        factor31: [var3, var1],
    }

    # Build factor graph
    fg = FactorGraph(
        variable_li=[var1, var2, var3], factor_li=[factor12, factor23, factor31], edges=edges
    )

    return fg


def run_engine_comparison():
    """Compare different engines on the same graph."""
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 70)
    print("DiffusionEngine Test - Cycle Graph (3 variables, 3 factors)")
    print("=" * 70)
    print()

    # Create factor graph
    fg = create_cycle_graph()

    # Test configurations
    configs = [
        {"name": "Pure BP (no diffusion)", "engine": BPEngine, "params": {}},
        {"name": "Light Diffusion (α=0.1)", "engine": DiffusionEngine, "params": {"alpha": 0.1}},
        {"name": "Medium Diffusion (α=0.3)", "engine": DiffusionEngine, "params": {"alpha": 0.3}},
        {"name": "Heavy Diffusion (α=0.7)", "engine": DiffusionEngine, "params": {"alpha": 0.7}},
        {"name": "Damping (0.9) for comparison", "engine": DampingEngine, "params": {"damping_factor": 0.9}},
    ]

    results = []

    for config in configs:
        print(f"\n{'─' * 70}")
        print(f"Testing: {config['name']}")
        print('─' * 70)

        # Create fresh graph for each test
        fg_test = create_cycle_graph()

        # Create engine
        engine = config["engine"](factor_graph=fg_test, computator=MinSumComputator(), **config["params"])

        # Run
        engine.run(max_iter=100)

        # Collect results
        snapshot = engine.latest_snapshot()
        final_cost = snapshot.global_cost if snapshot and snapshot.global_cost is not None else 0.0
        # Check if converged by seeing if it stopped before max_iter
        converged = engine.iteration_count < 100
        result = {
            "name": config["name"],
            "converged": converged,
            "iterations": engine.iteration_count,
            "final_cost": final_cost,
            "assignment": snapshot.assignments if snapshot else {},
        }
        results.append(result)

        # Print results
        print(f"  Converged: {result['converged']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final Cost: {result['final_cost']:.2f}")
        print(f"  Assignment: {result['assignment']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Engine':<40} {'Converged':<12} {'Iters':<8} {'Cost':<10}")
    print("-" * 70)
    for r in results:
        conv = "✓" if r["converged"] else "✗"
        print(f"{r['name']:<40} {conv:<12} {r['iterations']:<8} {r['final_cost']:<10.2f}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("• α=0: No diffusion (equivalent to pure BP)")
    print("• α=0.1-0.3: Recommended range - balances local info with neighbor smoothing")
    print("• α=0.7: Heavy smoothing - may slow convergence but could help on hard problems")
    print("• Damping vs Diffusion:")
    print("    - Damping: temporal smoothing (current vs previous iteration)")
    print("    - Diffusion: spatial smoothing (local vs neighbors at same iteration)")
    print()


if __name__ == "__main__":
    run_engine_comparison()
