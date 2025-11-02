from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

from ...bp.factor_graph import FactorGraph
from ...core.agents import FactorAgent, VariableAgent
from ..protocols import (
    CostAccumulationPolicy,
    DuplicateDetectionPolicy,
    ExpansionPolicy,
    FactorGraphProtocol,
    GoalPolicy,
    HeuristicPolicy,
    StateKeyFn,
)

Assignment = Mapping[Hashable, Any]
MutableAssignment = Dict[Hashable, Any]


def _normalise_order(order: Optional[Sequence[Hashable]], variables: Sequence[Hashable]) -> Sequence[Hashable]:
    if order is None:
        return list(variables)
    normalised: list[Hashable] = []
    for entry in order:
        if isinstance(entry, VariableAgent):
            normalised.append(entry.name)
        else:
            normalised.append(entry)
    return normalised


@dataclass(slots=True)
class FactorGraphView(FactorGraphProtocol):
    """Adapts :class:`FactorGraph` to the generic protocol surface."""

    factor_graph: FactorGraph

    def __post_init__(self) -> None:
        self._var_lookup: Dict[Hashable, VariableAgent] = {
            var.name: var for var in self.factor_graph.variables
        }
        self._factor_lookup: Dict[Hashable, FactorAgent] = {
            factor.name: factor for factor in self.factor_graph.factors
        }

    # ------------------------------------------------------------------#
    # Protocol interface
    # ------------------------------------------------------------------#
    def variables(self) -> Sequence[Hashable]:
        return [var.name for var in self.factor_graph.variables]

    def domain(self, var: Hashable) -> Sequence[Any]:
        agent = self._var_lookup[var]
        return range(int(agent.domain))

    def factors_for(self, var: Hashable) -> Sequence[FactorAgent]:
        agent = self._var_lookup[var]
        return [
            factor
            for factor in self.factor_graph.G.neighbors(agent)
            if isinstance(factor, FactorAgent)
        ]

    def factors(self) -> Sequence[FactorAgent]:
        return list(self.factor_graph.factors)

    def factor_cost(self, factor: FactorAgent, assignment: Assignment) -> float:
        table = getattr(factor, "cost_table", None)
        if table is None:
            return 0.0
        mapping = getattr(factor, "connection_number", {})
        indices: list[int] = [0] * len(mapping)
        for var_name, dim in mapping.items():
            if var_name not in assignment:
                return 0.0
            indices[dim] = int(assignment[var_name])
        return float(table[tuple(indices)])

    def is_complete(self, assignment: Assignment) -> bool:
        return all(var in assignment for var in self.variables())

    def assignment_cost(self, assignment: Assignment) -> float:
        return sum(self.factor_cost(factor, assignment) for factor in self.factor_graph.factors)


def _coerce_view(
    factor_graph: FactorGraph | FactorGraphProtocol,
) -> FactorGraphProtocol:
    if isinstance(factor_graph, FactorGraphView):
        return factor_graph
    if isinstance(factor_graph, FactorGraphProtocol):
        return factor_graph
    if isinstance(factor_graph, FactorGraph):
        return FactorGraphView(factor_graph)
    raise TypeError(f"Unsupported factor graph type {type(factor_graph)!r}")


class FGStateKey(StateKeyFn[Assignment]):
    """Deterministic key for assignment dictionaries."""

    def __call__(self, state: Assignment) -> Hashable:
        return tuple(sorted(state.items()))


class FGExpansion(ExpansionPolicy[Assignment, Tuple[Hashable, Any], float]):
    """Expands partial assignments by assigning the next variable in the order."""

    def __init__(
        self,
        factor_graph: FactorGraph | FactorGraphProtocol,
        var_order: Optional[Sequence[Hashable]] = None,
    ) -> None:
        self.view = _coerce_view(factor_graph)
        self.order = _normalise_order(var_order, self.view.variables())

    def expand(self, state: Assignment) -> Iterable[Tuple[Tuple[Hashable, Any], Assignment, float]]:
        current = dict(state)
        for var in self.order:
            if var not in current:
                neighbours = self.view.factors_for(var)
                base_cost = sum(self.view.factor_cost(factor, current) for factor in neighbours)
                for value in self.view.domain(var):
                    candidate: MutableAssignment = dict(current)
                    candidate[var] = value
                    new_cost = sum(self.view.factor_cost(factor, candidate) for factor in neighbours)
                    yield (var, value), candidate, float(new_cost - base_cost)
                return
        return []


class FGHeuristic(HeuristicPolicy[Assignment, float]):
    """Baseline admissible heuristic based on optimistic factor completions."""

    def __init__(self, factor_graph: FactorGraph | FactorGraphProtocol) -> None:
        self.view = _coerce_view(factor_graph)

    def _factor_min_completion(self, factor: FactorAgent, state: Assignment) -> float:
        mapping = getattr(factor, "connection_number", {})
        domains: list[Sequence[int]] = []
        for var_name in mapping:
            if var_name in state:
                domains.append([int(state[var_name])])
            else:
                domains.append(self.view.domain(var_name))
        table = getattr(factor, "cost_table", None)
        if table is None:
            return 0.0
        best = None
        for combo in itertools.product(*domains):
            cost = float(table[combo])
            if best is None or cost < best:
                best = cost
        return float(best or 0.0)

    def h(self, state: Assignment) -> float:
        seen: set[int] = set()
        optimistic = 0.0
        accounted = 0.0
        for var in self.view.variables():
            for factor in self.view.factors_for(var):
                ident = id(factor)
                if ident in seen:
                    continue
                seen.add(ident)
                optimistic += self._factor_min_completion(factor, state)
                accounted += self.view.factor_cost(factor, state)
        estimate = max(optimistic - accounted, 0.0)
        return float(estimate)


class FGGoal(GoalPolicy[Assignment]):
    """Goal condition: all variables assigned."""

    def __init__(self, factor_graph: FactorGraph | FactorGraphProtocol) -> None:
        self.view = _coerce_view(factor_graph)

    def is_goal(self, state: Assignment) -> bool:
        return self.view.is_complete(state)


class FGCost(CostAccumulationPolicy):
    """Standard additive MAP-style cost accumulation."""

    def g_update(self, g_so_far: float, step_cost: float) -> float:
        return g_so_far + float(step_cost)

    def f_score(self, g: float, h: float) -> float:
        return g + h


class FGDuplicate(DuplicateDetectionPolicy[Assignment]):
    """Closed list keyed by sorted assignment tuples."""

    def __init__(self) -> None:
        self.best: Dict[Tuple[Tuple[Hashable, Any], ...], float] = {}

    def better_path(self, state_key: Tuple[Tuple[Hashable, Any], ...], g: float) -> bool:
        prev = self.best.get(state_key)
        return prev is None or g < prev - 1e-12

    def record(self, state_key: Tuple[Tuple[Hashable, Any], ...], g: float) -> None:
        self.best[state_key] = g


__all__ = [
    "Assignment",
    "FGStateKey",
    "FGExpansion",
    "FGHeuristic",
    "FGGoal",
    "FGCost",
    "FGDuplicate",
    "FactorGraphView",
]
