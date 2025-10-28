from __future__ import annotations

from typing import (
    Any,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

S = TypeVar("S")  # Search state
A = TypeVar("A")  # Action
C = TypeVar("C")  # Cost type (float-compatible)


@runtime_checkable
class StateKeyFn(Protocol[S]):
    def __call__(self, state: S) -> Hashable:
        ...


@runtime_checkable
class ExpansionPolicy(Protocol[S, A, C]):
    """Generate neighbor states from ``state`` with step costs."""

    def expand(self, state: S) -> Iterable[Tuple[A, S, C]]:
        ...


@runtime_checkable
class HeuristicPolicy(Protocol[S, C]):
    """Admissible or domain heuristic ``h(s)``."""

    def h(self, state: S) -> C:
        ...


@runtime_checkable
class GoalPolicy(Protocol[S]):
    def is_goal(self, state: S) -> bool:
        ...


@runtime_checkable
class CostAccumulationPolicy(Protocol):
    """Combine path-cost ``g`` and step-cost ``c`` -> new ``g``; compute ``f`` for priority."""

    def g_update(self, g_so_far: float, step_cost: float) -> float:
        ...

    def f_score(self, g: float, h: float) -> float:
        ...


@runtime_checkable
class DuplicateDetectionPolicy(Protocol[S]):
    """Track visited/closed or superior g-values."""

    def better_path(self, state_key: Hashable, g: float) -> bool:
        ...

    def record(self, state_key: Hashable, g: float) -> None:
        ...


@runtime_checkable
class TerminationPolicy(Protocol[S]):
    """Global stop conditions (max_expansions, time, target f, etc.)."""

    def should_stop(self, current_best: Optional[S]) -> bool:
        ...


@runtime_checkable
class BacktrackPolicy(Protocol[S, A]):
    """Reconstruct path/solution from a terminal state key."""

    def path(self, terminal_key: Hashable) -> Sequence[Tuple[S, Optional[A]]]:
        ...


@runtime_checkable
class FactorGraphProtocol(Protocol):
    """Minimal surface required by the search adapters."""

    def variables(self) -> Sequence[Hashable]:
        ...

    def domain(self, var: Hashable) -> Sequence[Any]:
        ...

    def factors_for(self, var: Hashable) -> Sequence[Any]:
        ...

    def factor_cost(self, factor: Any, assignment: Mapping[Hashable, Any]) -> float:
        ...

    def is_complete(self, assignment: Mapping[Hashable, Any]) -> bool:
        ...

    def assignment_cost(self, assignment: Mapping[Hashable, Any]) -> float:
        ...

