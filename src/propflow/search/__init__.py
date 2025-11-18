"""High-level search module exposing both legacy local search and new runtimes."""

from .search_computator import (
    DSAComputator,
    MGM2Computator,
    MGMComputator,
    SearchComputator,
)
from .search_engine import (
    EngineHistory,
    EngineHooks,

    SearchEngine,
)
from .legacy_engines import (
    DSAEngine,
    MGM2Engine,
    MGMEngine,
)
from .algorithms import (
    a_star_factor_graph,
    beam_search_factor_graph,
    greedy_best_first_factor_graph,
    iddfs_factor_graph,
)
from .policies import (
    ClosedList,
    DefaultCostAcc,
    DefaultHeuristic,
    DefaultStateKey,
)
from .adapters.factor_graph import (
    Assignment,
    FGCost,
    FGDuplicate,
    FGExpansion,
    FGGoal,
    FGHeuristic,
    FGStateKey,
    FactorGraphView,
)
from .search_agents import SearchVariableAgent, extend_variable_agent_for_search

__all__ = [
    # Runtime engine
    "SearchEngine",
    "EngineHooks",
    "EngineHistory",
    # Convenience policies & adapters
    "DefaultStateKey",
    "DefaultHeuristic",
    "DefaultCostAcc",
    "ClosedList",
    "FGStateKey",
    "FGExpansion",
    "FGHeuristic",
    "FGGoal",
    "FGCost",
    "FGDuplicate",
    "FactorGraphView",
    "Assignment",
    # Algorithms
    "a_star_factor_graph",
    "beam_search_factor_graph",
    "greedy_best_first_factor_graph",
    "iddfs_factor_graph",
# Legacy local-search support

    "DSAEngine",
    "MGMEngine",
    "MGM2Engine",
    "SearchComputator",
    "DSAComputator",
    "MGMComputator",
    "MGM2Computator",
    "SearchVariableAgent",
    "extend_variable_agent_for_search",
]
