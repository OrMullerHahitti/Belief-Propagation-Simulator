"""Engine classes for search-based DCOP algorithms."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import networkx as nx

from ..bp.engine_base import BPEngine
from ..bp.engine_components import Cycle, Step, History
from ..bp.factor_graph import FactorGraph
from ..core.agents import VariableAgent
from .search_agents import SearchVariableAgent, extend_variable_agent_for_search
from .search_computator import (
    DSAComputator,
    MGMComputator,
    MGM2Computator,
    SearchComputator,
)

logger = logging.getLogger(__name__)


class SearchEngine(BPEngine):
    """Base engine for local-search algorithms operating on factor graphs."""

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        *,
        name: str = "SearchEngine",
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            factor_graph=factor_graph, computator=computator, name=name, **kwargs
        )
        self.history = History(
            engine_type=self.__class__.__name__,
            computator=computator,
            factor_graph=factor_graph,
            use_bct_history=self._use_bct_history,
        )
        self.max_iterations = max_iterations
        self.best_assignment: Dict[str, Any] | None = None
        self.best_cost: float = float("inf")
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------#
    # Hooks & helpers
    # ------------------------------------------------------------------#
    def post_init(self) -> None:
        """Convert BP variable agents to search-capable agents."""
        super().post_init()
        self._prepare_variable_agents()
        if isinstance(self.computator, SearchComputator):
            self.computator.bind_factor_graph(self.graph)
        self.best_assignment = self.assignments.copy()
        self.best_cost = self.calculate_global_cost()

    def _prepare_variable_agents(self) -> None:
        mapping: Dict[VariableAgent, SearchVariableAgent] = {}
        for var in list(self.graph.variables):
            mapping[var] = extend_variable_agent_for_search(var)

        if not mapping:
            return

        # Relabel nodes in-place so the factor graph keeps the same connectivity.
        nx.relabel_nodes(self.graph.G, mapping, copy=False)
        self.graph.variables = [mapping.get(var, var) for var in self.graph.variables]
        self.var_nodes = tuple(self.graph.variables)
        self.factor_nodes = tuple(self.graph.factors)
        # Re-assign computator now that new agents are in place.
        self.graph.set_computator(self.computator)

    def _neighbour_assignments(self, variable: SearchVariableAgent) -> Dict[str, Any]:
        assignments: Dict[str, Any] = {}
        for factor in self.graph.G.neighbors(variable):
            if getattr(factor, "type", None) != "factor":
                continue
            for neighbour in self.graph.G.neighbors(factor):
                if (
                    neighbour is variable
                    or getattr(neighbour, "type", None) != "variable"
                ):
                    continue
                assignments.setdefault(neighbour.name, neighbour.curr_assignment)
        return assignments

    @property
    def global_cost(self) -> float:
        return self.calculate_global_cost()

    def _record_iteration(self, iteration: int, step: Step, changes: int) -> None:
        cycle = Cycle(iteration)
        cycle.add(step)
        self.history[iteration] = cycle
        self.history.assignments[iteration] = dict(self.assignments)
        self.history.beliefs[iteration] = {}

        current_cost = self.calculate_global_cost()
        self.history.costs.append(current_cost)

        self.stats["iterations"] += 1
        self.stats["changes"] = changes
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_assignment = dict(self.assignments)
            self.stats["improvements"] += 1

        if isinstance(self.computator, SearchComputator):
            self.computator.next_iteration()

    # ------------------------------------------------------------------#
    # Engine API
    # ------------------------------------------------------------------#
    def step(self, i: int = 0) -> Tuple[Step, int]:
        raise NotImplementedError

    def run(
        self,
        max_iter: Optional[int] = None,
        save_json: bool = False,
        save_csv: bool = True,
        filename: Optional[str] = None,
        config_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        iterations = max_iter if max_iter is not None else self.max_iterations
        self.stats = {
            "iterations": 0,
            "improvements": 0,
            "changes": 0,
            "final_cost": None,
        }
        self.history.costs = []
        self.history.beliefs = {}
        self.history.assignments = {}
        self.history.cycles = {}

        for iteration in range(iterations):
            step, changes = self.step(iteration)
            self._record_iteration(iteration, step, changes)
            if self._is_converged():
                logger.info("Local search converged after %s iterations", iteration + 1)
                break

        self.stats["final_cost"] = self.best_cost

        if save_json:
            self.history.save_results(filename or "local_search_results.json")
        if save_csv:
            cfg_name = config_name or self._generate_config_name()
            self.history.save_csv(cfg_name)

        return {
            "best_assignment": self.best_assignment,
            "best_cost": self.best_cost,
            **self.stats,
        }


class DSAEngine(SearchEngine):
    """Engine for the Distributed Stochastic Algorithm (DSA)."""

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: DSAComputator,
        *,
        name: str = "DSAEngine",
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            max_iterations=max_iterations,
            **kwargs,
        )

    def step(self, i: int = 0) -> Tuple[Step, int]:
        step = Step(i)
        proposals: Dict[SearchVariableAgent, int] = {}

        for var in self.var_nodes:
            if not isinstance(var, SearchVariableAgent):
                continue
            neighbours = self._neighbour_assignments(var)
            decision = self.computator.compute_decision(var, neighbours)
            if decision is None:
                decision = var.curr_assignment
            proposals[var] = int(decision)

        changes = 0
        for var, new_value in proposals.items():
            current_value = int(var.curr_assignment)
            if new_value != current_value:
                var.curr_assignment = int(new_value)
                changes += 1

        return step, changes


class MGMEngine(SearchEngine):
    """Engine for the Maximum Gain Messaging (MGM) algorithm."""

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: MGMComputator,
        *,
        name: str = "MGMEngine",
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            max_iterations=max_iterations,
            **kwargs,
        )

    def _gather_neighbour_gains(self, agent: SearchVariableAgent) -> Dict[str, float]:
        gains: Dict[str, float] = {}
        computator: MGMComputator = self.computator  # type: ignore[assignment]
        for factor in self.graph.G.neighbors(agent):
            if getattr(factor, "type", None) != "factor":
                continue
            for neighbour in self.graph.G.neighbors(factor):
                if neighbour is agent or getattr(neighbour, "type", None) != "variable":
                    continue
                info = computator.agent_gains.get(neighbour.name, {})
                gains[neighbour.name] = float(info.get("gain", 0.0))
        return gains

    def step(self, i: int = 0) -> Tuple[Step, int]:
        step = Step(i)
        computator = self.computator
        if not isinstance(computator, MGMComputator):
            raise TypeError("MGMEngine requires an MGMComputator instance.")

        computator.begin_gain_phase()
        for var in self.var_nodes:
            if not isinstance(var, SearchVariableAgent):
                continue
            neighbours = self._neighbour_assignments(var)
            computator.compute_decision(var, neighbours)

        for var in self.var_nodes:
            if not isinstance(var, SearchVariableAgent):
                continue
            neighbour_gains = self._gather_neighbour_gains(var)
            computator.set_neighbour_gains(var.name, neighbour_gains)

        computator.set_phase("decision")
        proposals: Dict[SearchVariableAgent, int] = {}
        for var in self.var_nodes:
            if not isinstance(var, SearchVariableAgent):
                continue
            neighbours = self._neighbour_assignments(var)
            decision = computator.compute_decision(var, neighbours)
            if decision is None:
                decision = var.curr_assignment
            proposals[var] = int(decision)

        changes = 0
        for var, new_value in proposals.items():
            current_value = int(var.curr_assignment)
            if new_value != current_value:
                var.curr_assignment = new_value
                changes += 1

        return step, changes


class MGM2Engine(SearchEngine):
    """Engine for the MGM2 algorithm allowing coordinated pair moves."""

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: MGM2Computator,
        *,
        name: str = "MGM2Engine",
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            max_iterations=max_iterations,
            **kwargs,
        )

    def _variable_lookup(self) -> Dict[str, SearchVariableAgent]:
        return {
            var.name: var
            for var in self.var_nodes
            if isinstance(var, SearchVariableAgent)
        }

    def step(self, i: int = 0) -> Tuple[Step, int]:
        step = Step(i)
        computator = self.computator
        if not isinstance(computator, MGM2Computator):
            raise TypeError("MGM2Engine requires an MGM2Computator instance.")

        computator.begin_iteration()
        var_lookup = self._variable_lookup()

        for var in var_lookup.values():
            neighbours = self._neighbour_assignments(var)
            computator.compute_decision(var, neighbours)

        best_single_gain = 0.0
        best_single_agent: Optional[str] = None
        for name, info in computator.single_gains.items():
            gain = float(info["gain"])
            if gain > best_single_gain or (
                gain == best_single_gain
                and best_single_agent
                and name < best_single_agent
            ):
                best_single_gain = gain
                best_single_agent = name

        best_pair_gain = 0.0
        best_pair_data: Optional[
            Tuple[SearchVariableAgent, SearchVariableAgent, Tuple[int, int]]
        ] = None
        best_pair_names: Optional[Tuple[str, str]] = None
        seen_pairs: set[Tuple[str, str]] = set()

        for var in var_lookup.values():
            for factor in self.graph.G.neighbors(var):
                if getattr(factor, "type", None) != "factor":
                    continue
                for neighbour in self.graph.G.neighbors(factor):
                    if (
                        neighbour is var
                        or getattr(neighbour, "type", None) != "variable"
                    ):
                        continue
                    if not isinstance(neighbour, SearchVariableAgent):
                        continue
                    key = tuple(sorted((var.name, neighbour.name)))
                    if key in seen_pairs:
                        continue
                    pair_info = computator.evaluate_pair_gain(var, neighbour)
                    seen_pairs.add(key)
                    gain = float(pair_info["gain"])
                    if gain > best_pair_gain or (
                        gain == best_pair_gain
                        and best_pair_names
                        and key < best_pair_names
                    ):
                        best_pair_gain = gain
                        best_pair_data = (
                            var,
                            neighbour,
                            tuple(int(v) for v in pair_info["values"]),
                        )
                        best_pair_names = key

        changes = 0
        if best_pair_data and best_pair_gain > max(best_single_gain, 0.0):
            var_a, var_b, (val_a, val_b) = best_pair_data
            if int(var_a.curr_assignment) != val_a:
                var_a.curr_assignment = val_a
                changes += 1
            if int(var_b.curr_assignment) != val_b:
                var_b.curr_assignment = val_b
                changes += 1
        elif best_single_agent and best_single_gain > 0.0:
            info = computator.single_gains[best_single_agent]
            target_value = int(info["best_value"])
            agent = var_lookup[best_single_agent]
            if int(agent.curr_assignment) != target_value:
                agent.curr_assignment = target_value
                changes += 1

        return step, changes
