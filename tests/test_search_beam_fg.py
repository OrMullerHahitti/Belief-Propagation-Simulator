from propflow.search.adapters.factor_graph import FactorGraphView
from propflow.search.algorithms import beam_search_factor_graph


def test_beam_fg_monotone_with_width(simple_fg):
    narrow_engine = beam_search_factor_graph(simple_fg, beam_width=2)
    wide_engine = beam_search_factor_graph(simple_fg, beam_width=8)

    narrow_goal = narrow_engine.run({})
    wide_goal = wide_engine.run({})

    view = FactorGraphView(simple_fg)

    assert wide_goal is not None, "Beam search with wider beam should find a solution."
    narrow_cost = (
        view.assignment_cost(narrow_goal.state) if narrow_goal is not None else float("inf")
    )
    wide_cost = view.assignment_cost(wide_goal.state)

    assert wide_cost <= narrow_cost + 1e-9
