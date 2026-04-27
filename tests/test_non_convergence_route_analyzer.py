import numpy as np

from experiments.other.non_convergence_chain.code.route_analyzer import analyze_routes_from_graph
from propflow import FactorAgent, FGBuilder, VariableAgent


def _factor(name, table):
    return FactorAgent.create_from_cost_table(name, np.asarray(table, dtype=float))


def _triangle_graph(*tables):
    x1 = VariableAgent("X1", 2)
    x2 = VariableAgent("X2", 2)
    x3 = VariableAgent("X3", 2)
    f12 = _factor("F12", tables[0])
    f23 = _factor("F23", tables[1])
    f31 = _factor("F31", tables[2])
    return FGBuilder.build_from_edges(
        [x1, x2, x3],
        [f12, f23, f31],
        {f12: [x1, x2], f23: [x2, x3], f31: [x3, x1]},
    )


def _chain_graph():
    x1 = VariableAgent("X1", 2)
    x2 = VariableAgent("X2", 2)
    x3 = VariableAgent("X3", 2)
    f12 = _factor("F12", [[0, 4], [5, 6]])
    f23 = _factor("F23", [[0, 4], [5, 6]])
    return FGBuilder.build_from_edges(
        [x1, x2, x3],
        [f12, f23],
        {f12: [x1, x2], f23: [x2, x3]},
    )


def test_single_cycle_minimal_route_consistent():
    graph = _triangle_graph(
        [[0, 5], [5, 6]],
        [[0, 5], [5, 6]],
        [[0, 5], [5, 6]],
    )

    result = analyze_routes_from_graph(graph)

    assert result["status"] == "ok"
    assert result["best_route"]["consistent"]
    assert not result["best_route"]["inconsistent"]


def test_single_cycle_minimal_route_inconsistent():
    graph = _triangle_graph(
        [[0, 5], [5, 6]],
        [[0, 5], [5, 6]],
        [[5, 5], [0, 6]],  # F31 variables are [X3, X1], so min implies X3=1, X1=0.
    )

    result = analyze_routes_from_graph(graph)

    assert result["status"] == "ok"
    assert result["best_route"]["inconsistent"]


def test_non_single_cycle_graph_skips_exact_route_classification():
    result = analyze_routes_from_graph(_chain_graph())

    assert result["status"] == "skipped"
    assert "single-cycle" in result["warning"]
    assert len(result["local_factor_diagnostics"]) == 2
