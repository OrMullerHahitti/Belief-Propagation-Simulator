import numpy as np

from src.propflow.bp.factor_graph import FactorGraph
from src.propflow.core.agents import FactorAgent, VariableAgent
from src.propflow.search.search_computator import (
    DSAComputator,
    MGMComputator,
    MGM2Computator,
)
from src.propflow.search.search_engine import DSAEngine, MGMEngine, MGM2Engine


def _build_two_variable_graph() -> FactorGraph:
    var_x = VariableAgent("x1", domain=2)
    var_y = VariableAgent("x2", domain=2)

    def cost_table(_, domain_size: int, **__) -> np.ndarray:
        # Simple binary constraint that prefers the assignment (0, 0).
        return np.array([[0.0, 2.0], [2.0, 4.0]], dtype=float)

    factor = FactorAgent(
        name="f_xy",
        domain=2,
        ct_creation_func=cost_table,
    )

    edges = {factor: [var_x, var_y]}
    return FactorGraph(variable_li=[var_x, var_y], factor_li=[factor], edges=edges)


def _build_pair_only_graph() -> FactorGraph:
    var_x = VariableAgent("x1", domain=2)
    var_y = VariableAgent("x2", domain=2)

    def cost_table(_, domain_size: int, **__) -> np.ndarray:
        # Only the joint move to (0, 0) reduces the cost.
        return np.array([[0.0, 4.0], [4.0, 3.0]], dtype=float)

    factor = FactorAgent(
        name="f_xy",
        domain=2,
        ct_creation_func=cost_table,
    )

    edges = {factor: [var_x, var_y]}
    return FactorGraph(variable_li=[var_x, var_y], factor_li=[factor], edges=edges)


def test_dsa_engine_finds_optimum_assignment():
    factor_graph = _build_two_variable_graph()
    computator = DSAComputator(probability=1.0, seed=0)
    engine = DSAEngine(factor_graph=factor_graph, computator=computator, max_iterations=5)

    # Start from a deliberately suboptimal assignment.
    for variable in engine.var_nodes:
        variable.curr_assignment = 1

    result = engine.run(max_iter=3, save_csv=False, save_json=False)

    assert result["best_assignment"] == {"x1": 0, "x2": 0}
    assert result["best_cost"] == 0.0
    assert engine.assignments == {"x1": 0, "x2": 0}


def test_mgm_engine_coordinates_single_winner_per_iteration():
    factor_graph = _build_two_variable_graph()
    computator = MGMComputator(seed=0)
    engine = MGMEngine(factor_graph=factor_graph, computator=computator, max_iterations=5)

    for variable in engine.var_nodes:
        variable.curr_assignment = 1

    result = engine.run(max_iter=4, save_json=False, save_csv=False)

    assert result["best_assignment"] == {"x1": 0, "x2": 0}
    assert result["best_cost"] == 0.0
    assert engine.assignments == {"x1": 0, "x2": 0}


def test_mgm2_engine_executes_pair_move_when_needed():
    factor_graph = _build_pair_only_graph()
    computator = MGM2Computator(seed=0)
    engine = MGM2Engine(factor_graph=factor_graph, computator=computator, max_iterations=5)

    for variable in engine.var_nodes:
        variable.curr_assignment = 1

    result = engine.run(max_iter=3, save_json=False, save_csv=False)

    assert result["best_assignment"] == {"x1": 0, "x2": 0}
    assert result["best_cost"] == 0.0
    assert engine.assignments == {"x1": 0, "x2": 0}
