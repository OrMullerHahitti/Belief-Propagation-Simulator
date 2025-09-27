"""Engine Classes for Search-Based DCOP Algorithms.

This module provides the orchestrating engine classes for running various
distributed local search algorithms like DSA and MGM. These engines adapt the
`BPEngine` interface to the specific needs of search-based, rather than
message-passing, algorithms.
"""

from typing import Dict, Optional, Any, Tuple
import logging

from ..bp.engine_base import BPEngine
from ..bp.factor_graph import FactorGraph
from ..bp.engine_components import Step, Cycle
from .search_computator import SearchComputator, KOptMGMComputator

logger = logging.getLogger(__name__)


class SearchEngine(BPEngine):
    """An abstract base class for engines that run search-based algorithms.

    This class adapts the `BPEngine` for local search algorithms. It tracks
    the best assignment found so far and provides a basic synchronous `step`
    method where agents compute and then apply their decisions.

    Attributes:
        max_iterations (int): The maximum number of iterations to run.
        best_assignment (dict): The best assignment found so far during the run.
        best_cost (float): The cost of the best assignment found so far.
        stats (dict): A dictionary to store statistics about the run.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        name: str = "SearchEngine",
        max_iterations: int = 100,
        **kwargs,
    ):
        """Initializes the search engine.

        Args:
            factor_graph: The factor graph representing the DCOP.
            computator: The search computator defining the algorithm's logic.
            name: The name of the engine instance.
            max_iterations: The default maximum number of iterations.
            **kwargs: Additional keyword arguments for the base `BPEngine`.
        """
        super().__init__(
            factor_graph=factor_graph, computator=computator, name=name, **kwargs
        )
        self.max_iterations = max_iterations
        self.best_assignment = None
        self.best_cost = float("inf")
        self.stats = {"iterations": 0, "improvements": 0, "changes": 0, "final_cost": None}

    def step(self, i: int = 0) -> Step:
        """Executes one synchronous step of a generic search algorithm.

        This implementation follows a simple two-phase process:
        1. All variable agents compute their potential new values.
        2. All variable agents update their assignments simultaneously.

        Args:
            i: The current step number.

        Returns:
            A `Step` object (used for history tracking, less detailed than in BP).
        """
        step = Step(i)
        for var in self.var_nodes:
            var.compute_search_step()
            self.post_var_compute(var)
        for var in self.var_nodes:
            var.mailer.send()
        for var in self.var_nodes:
            var.update_assignment()
            var.empty_mailbox()
            var.mailer.prepare()

        current_cost = self.global_cost
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_assignment = self.assignments.copy()
            self.stats["improvements"] += 1
        self.history.costs.append(current_cost)
        return step

    def cycle(self, j: int) -> Cycle:
        """Runs one complete cycle of the search algorithm.

        For most synchronous search algorithms, a cycle is equivalent to a single step.

        Args:
            j: The current cycle number.

        Returns:
            The completed `Cycle` object.
        """
        cy = Cycle(j)
        step_result = self.step(j)
        cy.add(step_result)
        self.post_cycle()
        self.history.beliefs[j] = self.beliefs()
        self.history.assignments[j] = self.assignments
        return cy

    def run(
        self,
        max_iter: Optional[int] = None,
        save_json: bool = False,
        save_csv: bool = True,
        filename: str = None,
        config_name: str = None,
    ) -> Dict[str, Any]:
        """Runs the search algorithm until `max_iter` is reached or it converges.

        Args:
            max_iter: The maximum number of iterations to run.
            save_json: Whether to save the full history as a JSON file.
            save_csv: Whether to save the cost history as a CSV file.
            filename: The base name for the output files.
            config_name: A name for the configuration being run.

        Returns:
            A dictionary containing the best assignment found, its cost, and statistics.
        """
        if config_name is None:
            config_name = self._generate_config_name()
        iterations = max_iter if max_iter is not None else self.max_iterations
        self.stats = {"iterations": 0, "improvements": 0, "changes": 0, "final_cost": None}

        for i in range(iterations):
            self.history[i] = self.cycle(i)
            self.stats["iterations"] += 1
            if self._is_converged():
                logger.info(f"Converged after {i + 1} iterations")
                break

        self.stats["final_cost"] = self.best_cost
        if save_json:
            self.history.save_results(filename or "results.json")
        if save_csv:
            self.history.save_csv(config_name)

        return {
            "best_assignment": self.best_assignment,
            "best_cost": self.best_cost,
            **self.stats,
        }

    def post_cycle(self) -> None:
        """A hook for logic to be executed after each cycle.

        This is used by algorithms with multiple phases to advance their state.
        """
        if isinstance(self.graph.variables[0].computator, SearchComputator):
            self.graph.variables[0].computator.next_iteration()


class DSAEngine(SearchEngine):
    """An engine for the Distributed Stochastic Algorithm (DSA).

    This engine implements the logic for DSA, where all agents act
    simultaneously and independently in each step.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        name: str = "DSAEngine",
        **kwargs,
    ):
        """Initializes the DSA engine.

        Args:
            factor_graph: The factor graph representing the DCOP.
            computator: The `DSAComputator` to use.
            name: The name of the engine instance.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            factor_graph=factor_graph, computator=computator, name=name, **kwargs
        )
        self._setup_search_agents()

    def _setup_search_agents(self) -> None:
        """Converts base variable agents to search-capable agents."""
        from .search_agents import extend_variable_agent_for_search

        extended_vars = []
        for var in self.var_nodes:
            search_var = extend_variable_agent_for_search(var)
            search_var.computator = self.computator
            connected_factors = [
                factor for factor in self.factor_nodes if var.name in factor.connection_number
            ]
            search_var.set_connected_factors(connected_factors)
            extended_vars.append(search_var)

        for old_var, new_var in zip(self.var_nodes, extended_vars):
            neighbors = list(self.graph.G.neighbors(old_var))
            for neighbor in neighbors:
                edge_data = self.graph.G.get_edge_data(old_var, neighbor)
                self.graph.G.remove_edge(old_var, neighbor)
                self.graph.G.add_edge(new_var, neighbor, **edge_data)
            self.graph.G.remove_node(old_var)
            self.graph.G.add_node(new_var)
        self.graph.variables = extended_vars

    def step(self, i: int = 0) -> Step:
        """Executes one synchronous step of the DSA algorithm.

        Args:
            i: The current step number.

        Returns:
            A `Step` object for history tracking.
        """
        step = Step(i)
        decisions = {}
        for var in self.var_nodes:
            neighbors_values = var.get_neighbor_values(self.graph)
            decisions[var.name] = var.compute_search_step(neighbors_values)

        changes = 0
        for var in self.var_nodes:
            old_value = var.curr_assignment
            var.update_assignment()
            if var.curr_assignment != old_value:
                changes += 1
        self.stats["changes"] = changes

        current_cost = self.global_cost
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_assignment = self.assignments.copy()
            self.stats["improvements"] += 1
        self.history.costs.append(current_cost)
        return step


class MGMEngine(SearchEngine):
    """An engine for the Maximum Gain Message (MGM) algorithm.

    This engine orchestrates the multi-phase process of MGM:
    1. Gain calculation: Each agent calculates its best possible local improvement.
    2. Gain exchange: Agents share their gains with neighbors.
    3. Decision: Only the agent with the maximum gain in its neighborhood moves.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        name: str = "MGMEngine",
        **kwargs,
    ):
        """Initializes the MGM engine.

        Args:
            factor_graph: The factor graph representing the DCOP.
            computator: The `MGMComputator` to use.
            name: The name of the engine instance.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            factor_graph=factor_graph, computator=computator, name=name, **kwargs
        )
        self._setup_search_agents()

    def _setup_search_agents(self) -> None:
        """Converts base variable agents to search-capable agents."""
        from .search_agents import extend_variable_agent_for_search

        extended_vars = []
        for var in self.var_nodes:
            search_var = extend_variable_agent_for_search(var)
            search_var.computator = self.computator
            connected_factors = [
                factor for factor in self.factor_nodes if var.name in factor.connection_number
            ]
            search_var.set_connected_factors(connected_factors)
            extended_vars.append(search_var)

        for old_var, new_var in zip(self.var_nodes, extended_vars):
            neighbors = list(self.graph.G.neighbors(old_var))
            for neighbor in neighbors:
                edge_data = self.graph.G.get_edge_data(old_var, neighbor)
                self.graph.G.remove_edge(old_var, neighbor)
                self.graph.G.add_edge(new_var, neighbor, **edge_data)
            self.graph.G.remove_node(old_var)
            self.graph.G.add_node(new_var)
        self.graph.variables = extended_vars

    def step(self, i: int = 0) -> Step:
        """Executes one full step of the MGM algorithm, including all phases.

        Args:
            i: The current step number.

        Returns:
            A `Step` object for history tracking.
        """
        step = Step(i)
        if hasattr(self.computator, "reset_phase"):
            self.computator.reset_phase()

        for var in self.var_nodes:
            neighbors_values = var.get_neighbor_values(self.graph)
            var.compute_search_step(neighbors_values)

        self._exchange_gains()

        if hasattr(self.computator, "move_to_decision_phase"):
            self.computator.move_to_decision_phase()

        changes = 0
        for var in self.var_nodes:
            neighbors_values = var.get_neighbor_values(self.graph)
            old_value = var.curr_assignment
            decision = var.compute_search_step(neighbors_values)
            if decision is not None and decision != old_value:
                var.curr_assignment = decision
                changes += 1
        self.stats["changes"] = changes

        current_cost = self.global_cost
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_assignment = self.assignments.copy()
            self.stats["improvements"] += 1
        self.history.costs.append(current_cost)
        return step

    def _exchange_gains(self) -> None:
        """Implements the gain exchange phase of the MGM algorithm."""
        var_neighbors = {}
        for var in self.var_nodes:
            var_neighbors[var.name] = [
                other_var_name
                for factor in self.factor_nodes if var.name in factor.connection_number
                for other_var_name in factor.connection_number if other_var_name != var.name
            ]

        for var in self.var_nodes:
            neighbor_gains = {}
            if hasattr(self.computator, "agent_gains"):
                for neighbor_name in var_neighbors.get(var.name, []):
                    if neighbor_name in self.computator.agent_gains:
                        neighbor_gains[neighbor_name] = self.computator.agent_gains[neighbor_name]["gain"]
            var.neighbor_gains = neighbor_gains


class KOptMGMEngine(SearchEngine):
    """An engine for the K-Optimal Maximum Gain Message (k-opt MGM) algorithm.

    This engine orchestrates the complex, multi-phase k-opt MGM algorithm,
    which allows coalitions of up to `k` agents to move simultaneously.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: KOptMGMComputator,
        name: str = "KOptMGMEngine",
        **kwargs,
    ):
        """Initializes the K-Opt MGM engine.

        Args:
            factor_graph: The factor graph representing the DCOP.
            computator: The `KOptMGMComputator` to use.
            name: The name of the engine instance.
            **kwargs: Additional keyword arguments.

        Raises:
            TypeError: If the provided computator is not a `KOptMGMComputator`.
        """
        if not isinstance(computator, KOptMGMComputator):
            raise TypeError("KOptMGMEngine requires a KOptMGMComputator")
        super().__init__(
            factor_graph=factor_graph, computator=computator, name=name, **kwargs
        )
        self.constraints = self._extract_constraints()

    def _extract_constraints(
        self,
    ) -> Dict[Tuple[str, str], Dict[Tuple[Any, Any], float]]:
        """Extracts binary constraints from the factor graph for coalition evaluation.

        Returns:
            A dictionary mapping pairs of agent names to their constraint costs.
        """
        constraints = {}
        for factor in self.factor_nodes:
            if len(factor.connection_number) == 2:
                var_names = list(factor.connection_number.keys())
                var_pair = tuple(sorted(var_names))
                constraints.setdefault(var_pair, {})
                domain_size = factor.domain
                for i in range(domain_size):
                    for j in range(domain_size):
                        indices = [None, None]
                        indices[factor.connection_number[var_names[0]]] = i
                        indices[factor.connection_number[var_names[1]]] = j
                        constraints[var_pair][(i, j)] = factor.cost_table[tuple(indices)]
        return constraints

    def step(self, i: int = 0) -> Step:
        """Executes one step of the k-opt MGM algorithm, progressing its phase.

        Args:
            i: The current step number.

        Returns:
            A `Step` object for history tracking.
        """
        step = Step(i)
        k_opt_computator = self.graph.variables[0].computator

        if k_opt_computator.phase == "exploration":
            for var in self.var_nodes:
                neighbors_values = self._get_neighbor_values(var)
                var.compute_search_step(neighbors_values)
            k_opt_computator.phase = "coordination"
        elif k_opt_computator.phase == "coordination":
            k_opt_computator.coalitions = k_opt_computator.form_coalitions(
                list(self.var_nodes), self.constraints
            )
            k_opt_computator.coalition_attempts += 1
            if k_opt_computator.coalition_attempts >= k_opt_computator.coalition_timeout:
                k_opt_computator.phase = "execution"
                k_opt_computator.coalition_attempts = 0
        elif k_opt_computator.phase == "execution":
            for var in self.var_nodes:
                neighbors_values = self._get_neighbor_values(var)
                new_value = var.compute_search_step(neighbors_values)
                if new_value is not None:
                    var.curr_assignment = new_value
            k_opt_computator.phase = "exploration"
            k_opt_computator.current_gains.clear()
            k_opt_computator.coalitions.clear()
            k_opt_computator.best_values.clear()

        self.history.costs.append(self.global_cost)
        return step

    def _get_neighbor_values(self, variable: Any) -> Dict[str, Any]:
        """Gets the current values of all neighboring variables.

        Args:
            variable: The variable agent whose neighbors are being queried.

        Returns:
            A dictionary mapping neighbor names to their current assignments.
        """
        neighbors_values = {}
        for neighbor in self.graph.G.neighbors(variable):
            if hasattr(neighbor, "curr_assignment"):
                neighbors_values[neighbor.name] = neighbor.curr_assignment
        return neighbors_values
