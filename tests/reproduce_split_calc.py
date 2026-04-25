import numpy as np
from propflow import FactorAgent, VariableAgent
from propflow.policies import split_specific_factors
from propflow.utils.fg_utils import FGBuilder
from propflow.engines import RDampingEngine


def run_repro():
    # Setup from notebook
    C12_split = np.array([[4, 10], [16, 8]]).T  # [[4, 16], [10, 8]]
    C23_unary = np.array([[10, 10], [0, 10]])

    X1 = VariableAgent("X1", domain=2)
    X2 = VariableAgent("X2", domain=2)
    X3 = VariableAgent("X3", domain=2)

    # Note: notebook multiplies C12_split * 2 before creating factor?
    # Cell 5: "create factor for X1-X2 ... cost_table=C12_split * 2  # full cost before split"
    # Then calls split_specific_factors.
    # So effectively the split factors will have C12_split.

    F12 = FactorAgent.create_from_cost_table("F12", cost_table=C12_split * 2)
    F23 = FactorAgent.create_from_cost_table("F23", cost_table=C23_unary)

    fg = FGBuilder.build_from_edges(
        variables=[X1, X2, X3],
        factors=[F12, F23],
        edges={F12: [X1, X2], F23: [X2, X3]},
    )

    split_specific_factors(fg, [F12])

    # Engine with damping
    engine = RDampingEngine(factor_graph=fg, damping_factor=0.5)

    # Run Step 0 (Iteration 1)
    engine.step(0)

    # Get beliefs
    snapshot = engine.snapshots[0]
    b_x2 = snapshot.beliefs["X2"]

    print("--- Step 0 Results ---")
    print(f"Belief X2: {b_x2}")
    if b_x2 is not None:
        delta = b_x2[1] - b_x2[0]
        print(f"X2 Delta (b[1]-b[0]): {delta}")

    # Check messages
    print("\nR Messages to X2:")
    for (f, v), val in snapshot.R.items():
        if v == "X2":
            print(f"{f}->{v}: {val}")


if __name__ == "__main__":
    run_repro()
