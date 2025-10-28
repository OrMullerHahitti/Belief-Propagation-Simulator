from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .adapters.factor_graph import (
    Assignment,
    FGCost,
    FGDuplicate,
    FGExpansion,
    FGGoal,
    FGHeuristic,
    FGStateKey,
)
from .frontier import BeamFrontier, FIFOFrontier, PriorityFrontier
from .policies import DefaultCostAcc, DefaultHeuristic
from .search_engine import EngineHistory, EngineHooks, SearchEngine


def _state_key() -> FGStateKey:
    return FGStateKey()


def a_star_factor_graph(
    factor_graph: Any,
    var_order: Optional[Sequence[Any]] = None,
    *,
    hooks: Optional[EngineHooks] = None,
    history: Optional[EngineHistory] = None,
) -> SearchEngine[Assignment, tuple[object, object]]:
    """Construct an A*-style engine for MAP search on factor graphs."""
    return SearchEngine(
        expander=FGExpansion(factor_graph, var_order),
        heuristic=FGHeuristic(factor_graph),
        goal=FGGoal(factor_graph),
        cost=FGCost(),
        duplicate=FGDuplicate(),
        state_key=_state_key(),
        frontier=PriorityFrontier(),
        hooks=hooks,
        history=history,
    )


def greedy_best_first_factor_graph(
    factor_graph: Any,
    var_order: Optional[Sequence[Any]] = None,
    *,
    hooks: Optional[EngineHooks] = None,
    history: Optional[EngineHistory] = None,
) -> SearchEngine[Assignment, tuple[object, object]]:
    """Greedy best-first variant using ``f = h``."""

    class GreedyCost(DefaultCostAcc):
        def f_score(self, g: float, h: float) -> float:
            return h

    engine = SearchEngine(
        expander=FGExpansion(factor_graph, var_order),
        heuristic=FGHeuristic(factor_graph),
        goal=FGGoal(factor_graph),
        cost=GreedyCost(),
        duplicate=FGDuplicate(),
        state_key=_state_key(),
        frontier=PriorityFrontier(),
        hooks=hooks,
        history=history,
    )
    return engine


def beam_search_factor_graph(
    factor_graph: Any,
    beam_width: int = 16,
    var_order: Optional[Sequence[Any]] = None,
    *,
    hooks: Optional[EngineHooks] = None,
    history: Optional[EngineHistory] = None,
) -> SearchEngine[Assignment, tuple[object, object]]:
    """Beam search retaining the best ``beam_width`` frontier nodes."""
    engine = a_star_factor_graph(
        factor_graph,
        var_order,
        hooks=hooks,
        history=history,
    )
    engine.frontier = BeamFrontier(beam_width)
    return engine


def iddfs_factor_graph(
    factor_graph: Any,
    depth_limit: int,
    var_order: Optional[Sequence[Any]] = None,
    *,
    hooks: Optional[EngineHooks] = None,
    history: Optional[EngineHistory] = None,
) -> SearchEngine[Assignment, tuple[object, object]]:
    """
    Iterative deepening DFS wrapper using a FIFO frontier.

    Depth control is expected to be handled externally by re-running the
    engine with increasing limits.
    """
    # depth_limit stored for downstream consumers
    engine = SearchEngine(
        expander=FGExpansion(factor_graph, var_order),
        heuristic=DefaultHeuristic(),
        goal=FGGoal(factor_graph),
        cost=FGCost(),
        duplicate=FGDuplicate(),
        state_key=_state_key(),
        frontier=FIFOFrontier(),
        hooks=hooks,
        history=history,
    )
    engine.depth_limit = depth_limit  # type: ignore[attr-defined]
    return engine


__all__ = [
    "a_star_factor_graph",
    "greedy_best_first_factor_graph",
    "beam_search_factor_graph",
    "iddfs_factor_graph",
]

