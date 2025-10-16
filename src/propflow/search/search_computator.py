"""Computator classes for search-based DCOP algorithms.

This module provides computator implementations that reuse the belief
propagation `Computator` interface while operating with local-search style
decisions on factor graphs.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..bp.factor_graph import FactorGraph
from ..core.components import Message
from ..core.dcop_base import Agent, Computator


class SearchComputator(Computator, ABC):
    """Base computator for local-search style algorithms.

    The computator keeps a reference to the factor graph so it can evaluate
    local costs directly from factor cost tables while still presenting the
    synchronous ``compute_Q`` / ``compute_R`` interface expected by the runtime.
    """

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__()
        self.iteration: int = 0
        self._graph: FactorGraph | None = None
        self._var_lookup: Dict[str, Agent] = {}
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------#
    # Lifecycle helpers
    # ------------------------------------------------------------------#
    def bind_factor_graph(self, factor_graph: FactorGraph) -> None:
        """Attach the factor graph so local costs can be computed."""
        self._graph = factor_graph
        self._var_lookup = {var.name: var for var in factor_graph.variables}

    def next_iteration(self) -> None:
        """Advance to the next iteration (override in subclasses as needed)."""
        self.iteration += 1

    # ------------------------------------------------------------------#
    # Message interface compatibility
    # ------------------------------------------------------------------#
    async def compute_Q(self, messages: List[Message]) -> List[Message]:
        """Send the current assignment to neighbouring factors.

        Local search does not rely on these messages, but keeping the method
        implemented allows the algorithms to coexist with inference engines.
        """
        if not messages:
            return []
        variable = messages[0].recipient
        assignment = float(getattr(variable, "curr_assignment", 0))
        payload = np.array([assignment], dtype=float)
        return [
            Message(data=payload, sender=variable, recipient=msg.sender)
            for msg in messages
        ]

    async def compute_R(
        self, cost_table: np.ndarray, incoming_messages: List[Message]
    ) -> List[Message]:
        """Return placeholder cost messages from factors to variables."""
        if not incoming_messages:
            return []
        factor = incoming_messages[0].recipient
        payload = np.array([0.0], dtype=float)
        return [
            Message(data=payload, sender=factor, recipient=msg.sender)
            for msg in incoming_messages
        ]

    # ------------------------------------------------------------------#
    # Local cost helpers
    # ------------------------------------------------------------------#
    def _ensure_graph(self) -> FactorGraph:
        if self._graph is None:
            raise RuntimeError(
                "SearchComputator must be bound to a factor graph before use."
            )
        return self._graph

    def _factor_neighbours(self, agent: Agent) -> Iterable[Agent]:
        graph = self._ensure_graph()
        return (
            neighbour
            for neighbour in graph.G.neighbors(agent)
            if getattr(neighbour, "type", None) == "factor"
        )

    def _variable_for_name(self, name: str) -> Agent:
        try:
            return self._var_lookup[name]
        except KeyError as exc:
            raise KeyError(f"Unknown variable '{name}' in factor graph.") from exc

    def evaluate_cost(
        self,
        agent: Agent,
        value: Any,
        neighbours_values: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate the local cost contribution of ``agent`` taking ``value``."""
        neighbours_values = neighbours_values or {}
        total_cost = 0.0
        for factor in self._factor_neighbours(agent):
            cost_table = getattr(factor, "cost_table", None)
            if cost_table is None:
                continue
            indices: list[int] = [0] * len(factor.connection_number)
            for var_name, dim in factor.connection_number.items():
                if var_name == agent.name:
                    indices[dim] = int(value)
                    continue
                if var_name in neighbours_values:
                    indices[dim] = int(neighbours_values[var_name])
                    continue
                neighbour_agent = self._variable_for_name(var_name)
                indices[dim] = int(getattr(neighbour_agent, "curr_assignment", 0))
            total_cost += float(cost_table[tuple(indices)])
        return total_cost

    def best_improvement(
        self, agent: Agent, neighbours_values: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, float, float]:
        """Return the best value, gain, and current cost for ``agent``."""
        neighbours_values = neighbours_values or {}
        current_value = int(getattr(agent, "curr_assignment", 0))
        current_cost = self.evaluate_cost(agent, current_value, neighbours_values)
        best_value = current_value
        best_gain = 0.0
        domain_size = int(getattr(agent, "domain", 2))

        for value in range(domain_size):
            if value == current_value:
                continue
            candidate_cost = self.evaluate_cost(agent, value, neighbours_values)
            gain = current_cost - candidate_cost
            if gain > best_gain:
                best_gain = gain
                best_value = value

        return best_value, best_gain, current_cost

    # ------------------------------------------------------------------#
    # Algorithm contract
    # ------------------------------------------------------------------#
    @abstractmethod
    def compute_decision(
        self, agent: Agent, neighbours_values: Dict[str, Any] | None
    ) -> Any:
        """Return the next assignment this agent should adopt."""


class DSAComputator(SearchComputator):
    """Distributed Stochastic Algorithm (DSA)."""

    def __init__(self, probability: float = 0.7, *, seed: Optional[int] = None) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in the range [0, 1].")
        super().__init__(seed=seed)
        self.probability = probability

    def compute_decision(
        self, agent: Agent, neighbours_values: Dict[str, Any] | None
    ) -> Any:
        neighbours_values = neighbours_values or {}
        current_value = int(getattr(agent, "curr_assignment", 0))
        best_value, best_gain, _ = self.best_improvement(agent, neighbours_values)

        if best_gain <= 0.0:
            return current_value
        if self._rng.random() <= self.probability:
            return best_value
        return current_value


class MGMComputator(SearchComputator):
    """Maximum Gain Messaging (MGM) computator."""

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)
        self.phase = "gain"
        self.agent_gains: Dict[str, Dict[str, Any]] = {}
        self._neighbour_gains: Dict[str, Dict[str, float]] = {}

    def begin_gain_phase(self) -> None:
        self.phase = "gain"
        self.agent_gains.clear()
        self._neighbour_gains.clear()

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def set_neighbour_gains(self, agent_name: str, gains: Dict[str, float]) -> None:
        self._neighbour_gains[agent_name] = gains

    def compute_decision(
        self, agent: Agent, neighbours_values: Dict[str, Any] | None
    ) -> Any:
        neighbours_values = neighbours_values or {}
        current_value = int(getattr(agent, "curr_assignment", 0))

        if self.phase == "gain":
            best_value, best_gain, current_cost = self.best_improvement(
                agent, neighbours_values
            )
            self.agent_gains[agent.name] = {
                "gain": best_gain,
                "best_value": best_value,
                "current_value": current_value,
                "current_cost": current_cost,
            }
            return None

        if self.phase != "decision":
            return current_value

        info = self.agent_gains.get(agent.name)
        if not info:
            return current_value

        gain = info["gain"]
        if gain <= 0.0:
            return current_value

        neighbour_gains = self._neighbour_gains.get(agent.name, {})
        for neighbour_name, neighbour_gain in neighbour_gains.items():
            if neighbour_gain > gain:
                return current_value
            if neighbour_gain == gain and neighbour_name < agent.name:
                return current_value

        return info["best_value"]
