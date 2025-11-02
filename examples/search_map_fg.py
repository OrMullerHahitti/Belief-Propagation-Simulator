"""Simple demonstration of the search runtime solving a MAP assignment."""

from __future__ import annotations

import numpy as np

from propflow.bp.factor_graph import FactorGraph
from propflow.core.agents import FactorAgent, VariableAgent
from propflow.search.adapters.factor_graph import FactorGraphView
from propflow.search.algorithms import a_star_factor_graph


def build_demo_factor_graph() -> FactorGraph:
    """Create a minimal chain factor graph with a well-defined MAP solution."""
    vars_list = [
        VariableAgent("x1", domain=2),
        VariableAgent("x2", domain=2),
        VariableAgent("x3", domain=2),
    ]

    cost_xy = np.array([[0.0, 1.5], [1.5, 3.0]], dtype=float)
    cost_yz = np.array([[0.0, 1.0], [1.0, 2.5]], dtype=float)

    factor_xy = FactorAgent.create_from_cost_table("f_xy", cost_xy)
    factor_yz = FactorAgent.create_from_cost_table("f_yz", cost_yz)

    edges = {
        factor_xy: [vars_list[0], vars_list[1]],
        factor_yz: [vars_list[1], vars_list[2]],
    }
    return FactorGraph(
        variable_li=vars_list, factor_li=[factor_xy, factor_yz], edges=edges
    )


def main() -> None:
    fg = build_demo_factor_graph()
    engine = a_star_factor_graph(fg)
    goal = engine.run({})

    if goal is None:
        print("Search failed to find a MAP assignment.")
        return

    view = FactorGraphView(fg)
    print("MAP assignment:", goal.state)
    print("Total cost:", view.assignment_cost(goal.state))


if __name__ == "__main__":
    main()
