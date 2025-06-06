"""
Engine classes for search-based algorithms like DSA and MGM.
These extend the base BPEngine interface to fit search problems.
"""

from typing import Dict, Optional, Any, Tuple
import logging

from base_all.bp_engine_base import BPEngine
from bp_base.factor_graph import FactorGraph
from bp_base.engine_components import Step, Cycle
from search.search_computator import SearchComputator

logger = logging.getLogger(__name__)


class SearchEngine(BPEngine):
    """
    Abstract base class for search-based algorithm engines.
    This class adapts the BP Engine interface for search algorithms.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        name: str = "SearchEngine",
        normalize: bool = False,
        max_iterations: int = 100,
        **kwargs
    ):
        """
        Initialize the search engine.

        Args:
            factor_graph: The factor graph representing the problem
            computator: The search computator to use
            name: Name of the engine
            normalize: Whether to normalize costs
            max_iterations: Maximum number of iterations
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            normalize=normalize,
            **kwargs
        )
        self.max_iterations = max_iterations

        # Track best assignment found so far
        self.best_assignment = None
        self.best_cost = float('inf')

        # Engine statistics
        self.stats = {
            "iterations": 0,
            "improvements": 0,
            "changes": 0,
            "final_cost": None
        }

    def step(self, i: int = 0) -> Step:
        """
        Execute one step of the search algorithm.
        This is different from BP as search algorithms may have different phases.

        Args:
            i: Step number

        Returns:
            The completed step
        """
        step = Step(i)

        # Phase 1: Variable agents compute their potential new values
        for var in self.var_nodes:
            var.compute_search_step()
            self.post_var_compute(var)

        # Phase 2: Variables exchange information
        for var in self.var_nodes:
            var.mailer.send()

        # Phase 3: Variables update their values based on the exchanged information
        for var in self.var_nodes:
            var.update_assignment()
            var.empty_mailbox()
            var.mailer.prepare()

        # Update best assignment if current is better
        current_cost = self.calculate_global_cost()
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_assignment = self.assignments.copy()
            self.stats["improvements"] += 1

        # Calculate costs and track metrics
        self.history.costs.append(current_cost)

        return step

    def cycle(self, j) -> Cycle:
        """
        Run one complete cycle of the search algorithm.
        For search algorithms, a cycle is typically a single step.

        Args:
            j: Cycle number

        Returns:
            The completed cycle
        """
        cy = Cycle(j)

        # For search algorithms, a cycle is typically just one step
        step_result = self.step(j)
        cy.add(step_result)

        # Update algorithm-specific state
        self.post_cycle()

        # Update beliefs and assignments
        self.history.beliefs[j] = self.get_beliefs()
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
        """
        Run the search algorithm for a maximum number of iterations.

        Args:
            max_iter: Maximum number of iterations (overrides self.max_iterations)
            save_json: Whether to save results as JSON
            save_csv: Whether to save results as CSV
            filename: Base filename for saved results
            config_name: Configuration name for saved results

        Returns:
            Dictionary with results
        """
        if config_name is None:
            config_name = self._generate_config_name()

        iterations = max_iter if max_iter is not None else self.max_iterations

        # Reset statistics
        self.stats = {
            "iterations": 0,
            "improvements": 0,
            "changes": 0,
            "final_cost": None
        }

        for i in range(iterations):
            self.history[i] = self.cycle(i)
            self.stats["iterations"] += 1

            # Check for convergence
            if self._is_converged():
                logger.info(f"Converged after {i + 1} iterations")
                break

        # Record final statistics
        self.stats["final_cost"] = self.best_cost

        # Save results
        if save_json:
            self.history.save_results(filename or "results.json")
        if save_csv:
            self.history.save_csv(config_name)

        return {
            "best_assignment": self.best_assignment,
            "best_cost": self.best_cost,
            "iterations": self.stats["iterations"],
            "improvements": self.stats["improvements"],
            "changes": self.stats["changes"]
        }

    def post_cycle(self):
        """
        Actions to perform after a cycle.
        This is a hook for subclasses to implement.
        """
        # Move to the next iteration in the computator
        if isinstance(self.graph.variables[0].computator, SearchComputator):
            self.graph.variables[0].computator.next_iteration()


class DSAEngine(SearchEngine):
    """
    Engine for the Distributed Stochastic Algorithm (DSA).
    DSA is a local search algorithm where agents probabilistically decide to change their values.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        name: str = "DSAEngine",
        **kwargs
    ):
        """
        Initialize the DSA engine.

        Args:
            factor_graph: The factor graph representing the problem
            computator: The search computator to use
            name: Name of the engine
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            **kwargs
        )


class MGMEngine(SearchEngine):
    """
    Engine for the Maximum Gain Message (MGM) algorithm.
    MGM is a local search algorithm where agents coordinate to make the move with maximum gain.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: SearchComputator,
        name: str = "MGMEngine",
        **kwargs
    ):
        """
        Initialize the MGM engine.

        Args:
            factor_graph: The factor graph representing the problem
            computator: The search computator to use
            name: Name of the engine
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            **kwargs
        )


class KOptMGMEngine(SearchEngine):
    """
    Engine for the K-Opt Maximum Gain Message (MGM) algorithm.

    K-Opt MGM extends standard MGM by allowing groups of up to k agents to
    coordinate their value changes simultaneously, potentially escaping local
    optima that standard MGM would get stuck in.

    The algorithm proceeds in three phases:
    1. Exploration: Agents calculate potential gains from changing their values
    2. Coordination: Agents form coalitions of size <= k to maximize overall gain
    3. Execution: Coalitions implement their coordinated value changes
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: KOptMGMComputator,
        name: str = "KOptMGMEngine",
        **kwargs
    ):
        """
        Initialize the K-Opt MGM engine.

        Args:
            factor_graph: The factor graph representing the problem
            computator: The K-Opt MGM computator to use
            name: Name of the engine
            **kwargs: Additional keyword arguments
        """
        if not isinstance(computator, KOptMGMComputator):
            raise TypeError("KOptMGMEngine requires a KOptMGMComputator")

        super().__init__(
            factor_graph=factor_graph,
            computator=computator,
            name=name,
            **kwargs
        )

        # Extract constraints from factor graph for coalition formation
        self.constraints = self._extract_constraints()

    def _extract_constraints(self) -> Dict[Tuple[str, str], Dict[Tuple[Any, Any], float]]:
        """
        Extract binary constraints from the factor graph.

        Returns:
            Dictionary mapping pairs of agent names to their constraint costs
        """
        constraints = {}

        # Extract constraints from factor nodes
        for factor in self.factor_nodes:
            if len(factor.connection_number) == 2:
                # Binary constraint
                var_names = list(factor.connection_number.keys())
                var_pair = tuple(sorted(var_names))

                # Initialize constraint dictionary if needed
                if var_pair not in constraints:
                    constraints[var_pair] = {}

                # Extract costs for all value combinations
                domain_size = factor.domain
                for i in range(domain_size):
                    for j in range(domain_size):
                        # Get indices in the correct order for the cost table
                        indices = [None, None]
                        indices[factor.connection_number[var_names[0]]] = i
                        indices[factor.connection_number[var_names[1]]] = j

                        # Store the cost
                        val_pair = (i, j)
                        constraints[var_pair][val_pair] = factor.cost_table[tuple(indices)]

        return constraints

    def step(self, i: int = 0) -> Step:
        """
        Execute one step of the K-Opt MGM algorithm.

        Args:
            i: Step number

        Returns:
            The completed step
        """
        step = Step(i)
        k_opt_computator = self.graph.variables[0].computator

        # Handle different phases of the algorithm
        if k_opt_computator.phase == "exploration":
            # Exploration phase: Agents calculate their potential gains
            for var in self.var_nodes:
                # Get neighbor values
                neighbors_values = self._get_neighbor_values(var)

                # Compute potential decision and gain (stored in computator)
                var.compute_search_step(neighbors_values)

            # Move to coordination phase
            k_opt_computator.phase = "coordination"

        elif k_opt_computator.phase == "coordination":
            # Coordination phase: Form coalitions of agents
            k_opt_computator.coalitions = k_opt_computator.form_coalitions(
                list(self.var_nodes),
                self.constraints
            )

            # Check if coordination is complete
            k_opt_computator.coalition_attempts += 1
            if k_opt_computator.coalition_attempts >= k_opt_computator.coalition_timeout:
                k_opt_computator.phase = "execution"
                k_opt_computator.coalition_attempts = 0

        elif k_opt_computator.phase == "execution":
            # Execution phase: Implement the coordinated changes
            for var in self.var_nodes:
                neighbors_values = self._get_neighbor_values(var)
                new_value = var.compute_search_step(neighbors_values)

                if new_value is not None:
                    var.curr_assignment = new_value

            # Reset for next cycle
            k_opt_computator.phase = "exploration"
            k_opt_computator.current_gains.clear()
            k_opt_computator.coalitions.clear()
            k_opt_computator.best_values.clear()

        # Calculate costs and track metrics
        global_cost = self.calculate_global_cost()
        self.history.costs.append(global_cost)

        return step

    def _get_neighbor_values(self, variable):
        """
        Get current values of all neighboring variables.

        Args:
            variable: The variable agent

        Returns:
            Dictionary mapping neighbor names to their current values
        """
        neighbors_values = {}
        for neighbor in self.graph.G.neighbors(variable):
            if hasattr(neighbor, "curr_assignment"):
                neighbors_values[neighbor.name] = neighbor.curr_assignment
        return neighbors_values
