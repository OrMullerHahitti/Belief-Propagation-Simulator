import copy

from propflow.bp.computators import MinSumComputator
from propflow.bp.engine_base import BPEngine
from propflow.search.algorithms import a_star_factor_graph


def test_map_equals_bp_on_tree_fg(tree_fg):
    search_fg = copy.deepcopy(tree_fg)

    # Run BP (min-sum) to obtain the MAP assignment on a tree factor graph.
    bp_engine = BPEngine(
        factor_graph=tree_fg,
        computator=MinSumComputator(),
        name="TreeBP",
    )
    bp_engine.run(max_iter=5, save_csv=False, save_json=False)
    bp_assignment = {var.name: int(var.curr_assignment) for var in tree_fg.variables}

    # Run A* search on a fresh copy of the factor graph.
    search_engine = a_star_factor_graph(search_fg)
    goal = search_engine.run({})
    assert goal is not None, "A* should find a MAP assignment on tree factor graphs."

    assert goal.state == bp_assignment
