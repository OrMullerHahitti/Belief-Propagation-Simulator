"""
Example: Training differentiable BP to optimize cost tables.

This example demonstrates how to use TrainableBPModule to learn cost tables
via gradient descent, minimizing the total cost of the BP solution.
"""

import numpy as np
from propflow import FactorGraph, VariableAgent, FactorAgent
from propflow.nn.trainable_bp import TrainableBPModule, BPTrainer

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_graph():
    """Create a simple 3-variable, 2-factor graph."""
    # Variables with domain size 3
    v1 = VariableAgent("v1", domain=3)
    v2 = VariableAgent("v2", domain=3)
    v3 = VariableAgent("v3", domain=3)

    # Factor 1: connects v1 and v2 (random costs)
    cost_table_1 = np.random.randint(0, 20, size=(3, 3)).astype(float)
    f1 = FactorAgent("f1", domain=3)
    f1.set_cost_table(cost_table_1)

    # Factor 2: connects v2 and v3 (random costs)
    cost_table_2 = np.random.randint(0, 20, size=(3, 3)).astype(float)
    f2 = FactorAgent("f2", domain=3)
    f2.set_cost_table(cost_table_2)

    # Build factor graph
    edges = {
        f1: [v1, v2],
        f2: [v2, v3],
    }
    fg = FactorGraph([v1, v2, v3], [f1, f2], edges)

    return fg, cost_table_1, cost_table_2


def main():
    print("=" * 70)
    print("Trainable BP Example: Learning Cost Tables via Gradient Descent")
    print("=" * 70)

    # Create factor graph
    fg, original_ct1, original_ct2 = create_sample_graph()

    print("\nOriginal Cost Tables:")
    print("Factor f1 (v1, v2):")
    print(original_ct1)
    print("\nFactor f2 (v2, v3):")
    print(original_ct2)

    # Create trainable model
    print("\n" + "-" * 70)
    print("Initializing trainable BP module...")
    model = TrainableBPModule(fg, tau=0.1, device="cpu")

    # Create trainer
    trainer = BPTrainer(model, learning_rate=0.05)

    # Get initial assignments and cost
    print("\nRunning initial BP (before training)...")
    initial_assignments = trainer.get_final_assignments(max_iter=30)
    initial_cost = float(trainer.compute_cost(initial_assignments))
    print(f"Initial assignments: {initial_assignments}")
    print(f"Initial total cost: {initial_cost:.2f}")

    # Train the model
    print("\n" + "-" * 70)
    print("Training to minimize total cost...")
    learned_tables = trainer.train(
        num_epochs=200,
        bp_iterations=20,
        verbose=True,
        convergence_threshold=1e-4,
    )

    # Get final assignments and cost
    print("\n" + "-" * 70)
    print("Running final BP (after training)...")
    final_assignments = trainer.get_final_assignments(max_iter=30)
    final_cost = float(trainer.compute_cost(final_assignments))
    print(f"Final assignments: {final_assignments}")
    print(f"Final total cost: {final_cost:.2f}")
    print(f"Cost reduction: {initial_cost - final_cost:.2f} ({100 * (initial_cost - final_cost) / initial_cost:.1f}%)")

    # Show learned cost tables
    print("\n" + "-" * 70)
    print("Learned Cost Tables:")
    print("Factor f1 (v1, v2):")
    print(learned_tables["f1"])
    print("\nFactor f2 (v2, v3):")
    print(learned_tables["f2"])

    # Plot loss history
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(trainer.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Total Cost)")
        plt.title("Training Progress: Cost Table Optimization")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("/tmp/trainable_bp_loss.png", dpi=150)
        print("\n" + "-" * 70)
        print("Loss plot saved to: /tmp/trainable_bp_loss.png")
    except ImportError:
        print("\nMatplotlib not available - skipping plot")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
