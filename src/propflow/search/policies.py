from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, Mapping, Sequence, Tuple

from .protocols import (
    CostAccumulationPolicy,
    DuplicateDetectionPolicy,
    ExpansionPolicy,
    GoalPolicy,
    HeuristicPolicy,
    StateKeyFn,
)


class DefaultStateKey(StateKeyFn[Mapping[Hashable, Any]]):
    """Robust keying strategy for dict-like assignments."""

    def __call__(self, state: Mapping[Hashable, Any]) -> Hashable:
        return tuple(sorted(state.items()))


class DefaultHeuristic(HeuristicPolicy[Any, float]):
    """Zero heuristic baseline."""

    def h(self, state: Any) -> float:  # pragma: no cover - simple
        return 0.0


class DefaultCostAcc(CostAccumulationPolicy):
    """Standard additive path cost with ``f = g + h``."""

    def g_update(self, g_so_far: float, step_cost: float) -> float:
        return g_so_far + float(step_cost)

    def f_score(self, g: float, h: float) -> float:
        return g + h


class ClosedList(DuplicateDetectionPolicy):
    """Classic closed-list for dominance checking."""

    def __init__(self) -> None:
        self.best: Dict[Hashable, float] = {}

    def better_path(self, state_key: Hashable, g: float) -> bool:
        prev = self.best.get(state_key)
        return prev is None or g < prev - 1e-12

    def record(self, state_key: Hashable, g: float) -> None:
        self.best[state_key] = g


# Convenience alias for typing clarity when policies need mappings
Assignment = Mapping[Hashable, Any]
ExpansionOutput = Iterable[Tuple[Any, Assignment, float]]

