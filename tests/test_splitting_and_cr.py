import numpy as np
from propflow.bp.engines_realizations import DampingSCFGEngine, DampingCROnceEngine
from propflow.core import VariableAgent, FactorAgent
from propflow.bp.factor_graph import FactorGraph


def create_simple_factor_graph():
    # Two variables, one factor
    var1 = VariableAgent(name="var1", domain=2)
    var2 = VariableAgent(name="var2", domain=2)
    # Cost table: just a 2x2 matrix
    cost_table = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor = FactorAgent(
        name="factor", domain=2, ct_creation_func=lambda **kw: cost_table.copy()
    )
    factor.connection_number = {"var1": 0, "var2": 1}
    fg = FactorGraph(
        variable_li=[var1, var2], factor_li=[factor], edges={factor: [var1, var2]}
    )
    return fg


def extract_var_messages(variable):
    # Returns a dict: {recipient: message data (as tuple)}
    return {msg.recipient.name: tuple(msg.data) for msg in variable.mailer.outbox}


def extract_factor_messages(factor):
    # Returns a dict: {recipient: message data (as tuple)}
    return {msg.recipient.name: tuple(msg.data) for msg in factor.mailer.outbox}


def test_equivalence_split_vs_costreduction():
    fg1 = create_simple_factor_graph()
    fg2 = create_simple_factor_graph()
    # Initialize bp
    engine_split = DampingSCFGEngine(
        factor_graph=fg1,
        split_factor=0.5,
        damping_factor=0.9,
        normalize=False,
        monitor_performance=False,
    )
    engine_costred = DampingCROnceEngine(
        factor_graph=fg2,
        reduction_factor=0.5,
        damping_factor=0.9,
        normalize=False,
        monitor_performance=False,
    )
    # Run one iteration
    engine_split.run(max_iter=1)
    engine_costred.run(max_iter=1)
    # Compare factor messages (outbox)
    split_factor_msgs = []
    for factor in fg1.factors:
        split_factor_msgs.append(extract_factor_messages(factor))
    costred_factor_msgs = []
    for factor in fg2.factors:
        costred_factor_msgs.append(extract_factor_messages(factor))
    # Flatten and sort for comparison
    split_msgs_flat = sorted([(k, v) for d in split_factor_msgs for k, v in d.items()])
    costred_msgs_flat = sorted(
        [(k, v) for d in costred_factor_msgs for k, v in d.items()]
    )
    assert (
        split_msgs_flat == costred_msgs_flat
    ), f"Factor messages differ: {split_msgs_flat} vs {costred_msgs_flat}"
    # Compare variable beliefs
    for i, (v1, v2) in enumerate(zip(fg1.variables, fg2.variables)):
        np.testing.assert_allclose(
            v1.belief, v2.belief, err_msg=f"Belief mismatch at variable {i}"
        )


if __name__ == "__main__":
    test_equivalence_split_vs_costreduction()
    print(
        "Test passed: The two bp produce identical messages and beliefs after one iteration."
    )
