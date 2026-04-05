
import numpy as np
from propflow import FactorAgent, VariableAgent
from propflow.policies import split_specific_factors
from propflow.utils.fg_utils import FGBuilder
from propflow.engines import RDampingEngine

def run_all_scenarios():
    # --- Common setup ---
    num_iterations = 15
    damping_factor = 0.5
    
    C12_split = np.array([[4, 10], [16, 8]]).T  # [[4, 16], [10, 8]]
    C12_full = np.array([[8, 20], [32, 16]]).T  # [[8, 32], [20, 16]]
    C23_unary = np.array([[10, 10], [0, 10]])

    # --- Scenario 1: Splitting + Damping ---
    X1_s1 = VariableAgent("X1", domain=2)
    X2_s1 = VariableAgent("X2", domain=2)
    X3_s1 = VariableAgent("X3", domain=2)
    F12_s1 = FactorAgent.create_from_cost_table("F12", cost_table=C12_split * 2)
    F23_s1 = FactorAgent.create_from_cost_table("F23", cost_table=C23_unary)
    fg_s1 = FGBuilder.build_from_edges(
        variables=[X1_s1, X2_s1, X3_s1],
        factors=[F12_s1, F23_s1],
        edges={F12_s1: [X1_s1, X2_s1], F23_s1: [X2_s1, X3_s1]},
    )
    split_specific_factors(fg_s1, [F12_s1])
    engine_s1 = RDampingEngine(factor_graph=fg_s1, damping_factor=damping_factor)

    # --- Scenario 2: Damping Only ---
    X1_s2 = VariableAgent("X1", domain=2)
    X2_s2 = VariableAgent("X2", domain=2)
    X3_s2 = VariableAgent("X3", domain=2)
    F12_s2 = FactorAgent.create_from_cost_table("F12", cost_table=C12_full)
    F23_s2 = FactorAgent.create_from_cost_table("F23", cost_table=C23_unary)
    fg_s2 = FGBuilder.build_from_edges(
        variables=[X1_s2, X2_s2, X3_s2],
        factors=[F12_s2, F23_s2],
        edges={F12_s2: [X1_s2, X2_s2], F23_s2: [X2_s2, X3_s2]},
    )
    engine_s2 = RDampingEngine(factor_graph=fg_s2, damping_factor=damping_factor)

    # --- Scenario 3: Splitting Only ---
    X1_s3 = VariableAgent("X1", domain=2)
    X2_s3 = VariableAgent("X2", domain=2)
    X3_s3 = VariableAgent("X3", domain=2)
    F12_s3 = FactorAgent.create_from_cost_table("F12", cost_table=C12_split * 2)
    F23_s3 = FactorAgent.create_from_cost_table("F23", cost_table=C23_unary)
    fg_s3 = FGBuilder.build_from_edges(
        variables=[X1_s3, X2_s3, X3_s3],
        factors=[F12_s3, F23_s3],
        edges={F12_s3: [X1_s3, X2_s3], F23_s3: [X2_s3, X3_s3]},
    )
    split_specific_factors(fg_s3, [F12_s3])
    engine_s3 = RDampingEngine(factor_graph=fg_s3, damping_factor=0.0)

    # --- Scenario 4: Baseline ---
    X1_s4 = VariableAgent("X1", domain=2)
    X2_s4 = VariableAgent("X2", domain=2)
    X3_s4 = VariableAgent("X3", domain=2)
    F12_s4 = FactorAgent.create_from_cost_table("F12", cost_table=C12_full)
    F23_s4 = FactorAgent.create_from_cost_table("F23", cost_table=C23_unary)
    fg_s4 = FGBuilder.build_from_edges(
        variables=[X1_s4, X2_s4, X3_s4],
        factors=[F12_s4, F23_s4],
        edges={F12_s4: [X1_s4, X2_s4], F23_s4: [X2_s4, X3_s4]},
    )
    engine_s4 = RDampingEngine(factor_graph=fg_s4, damping_factor=0.0)

    engines = [
        ("1. Splitting+Damping", engine_s1),
        ("2. Damping Only", engine_s2),
        ("3. Splitting Only", engine_s3),
        ("4. Baseline", engine_s4),
    ]

    # Run all engines
    for name, engine in engines:
        for i in range(num_iterations):
            engine.step(i)

    # Print X1 belief iteration by iteration
    print(f"{'Iter':>4} | {'1. Split+Damp':>25} | {'2. Damp Only':>25} | {'3. Split Only':>25} | {'4. Baseline':>25}")
    print("-" * 115)

    for i in range(num_iterations):
        row = [f"{i:>4}"]
        for name, engine in engines:
            snap = engine.snapshots[i]
            belief = snap.beliefs.get('X1')
            if belief is not None:
                # show delta = b[1] - b[0]
                delta = belief[1] - belief[0]
                row.append(f"{delta:>25.2f}")
            else:
                row.append(f"{'N/A':>25}")
        print(" | ".join(row))

if __name__ == "__main__":
    run_all_scenarios()
