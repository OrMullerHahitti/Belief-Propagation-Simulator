import itertools

from propflow.search.adapters.factor_graph import FactorGraphView
from propflow.search.algorithms import a_star_factor_graph


def _brute_force_optimal(view: FactorGraphView):
    variables = list(view.variables())
    best_cost = float("inf")
    best_assignment = None
    domains = [view.domain(var) for var in variables]
    for values in itertools.product(*domains):
        assignment = dict(zip(variables, values))
        cost = view.assignment_cost(assignment)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    return best_assignment, best_cost


def test_a_star_fg_finds_min_cost_assignment(simple_fg):
    engine = a_star_factor_graph(simple_fg)
    goal = engine.run(start_state={})

    assert goal is not None, "A* should find a solution on a finite factor graph."
    assignment = goal.state
    view = FactorGraphView(simple_fg)

    assert set(assignment.keys()) == set(view.variables())

    optimal_assignment, optimal_cost = _brute_force_optimal(view)
    goal_cost = view.assignment_cost(assignment)

    assert goal_cost <= optimal_cost + 1e-9
    assert assignment == optimal_assignment
