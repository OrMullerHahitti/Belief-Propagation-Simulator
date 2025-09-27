"""Computator Classes for Search-Based DCOP Algorithms.

This module defines the `Computator` logic for various distributed local
search algorithms like DSA (Distributed Stochastic Algorithm) and MGM (Maximum
Gain Message). These computators are designed to be used with `SearchAgent`
and `SearchEngine` classes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Set
import numpy as np

from ..core.dcop_base import Computator, Agent
from ..core.components import Message


class SearchComputator(Computator, ABC):
    """Abstract base class for computators used in search-based algorithms.

    This class adapts the `Computator` interface, which is originally designed
    for belief propagation, to the needs of local search algorithms. It
    introduces methods for making decisions and evaluating costs based on the
    local state of an agent and its neighbors.
    """

    def __init__(self):
        """Initializes the search computator."""
        super().__init__()
        self.iteration = 0
        self.is_decision_phase = True

    @abstractmethod
    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """Computes a decision for an agent based on its neighbors' current values.

        Args:
            agent: The agent making the decision.
            neighbors_values: A dictionary mapping neighbor agent names to their
                current assignment values.

        Returns:
            The computed decision value for the agent.
        """
        pass

    @abstractmethod
    def evaluate_cost(
        self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]
    ) -> float:
        """Evaluates the local cost of a potential value assignment for an agent.

        Args:
            agent: The agent being evaluated.
            value: The potential value to assign to the agent.
            neighbors_values: A dictionary of neighbor agent assignments.

        Returns:
            The resulting local cost (lower is better).
        """
        pass

    async def compute_Q(self, messages: List[Message]) -> List[Message]:
        """Adapts the `compute_Q` method for search algorithms.

        In search, this method is primarily for compatibility and may be used
        to exchange current value assignments.

        Args:
            messages: A list of incoming messages.

        Returns:
            A list of outgoing messages, typically containing the agent's
            current assignment.
        """
        if not messages:
            return []
        variable = messages[0].recipient
        return [
            Message(
                data=np.array([getattr(variable, "curr_assignment", 0)]),
                sender=variable,
                recipient=msg.sender,
            )
            for msg in messages
        ]

    async def compute_R(
        self, cost_table: np.ndarray, incoming_messages: List[Message]
    ) -> List[Message]:
        """Adapts the `compute_R` method for search algorithms.

        In search, this method is primarily for compatibility and may be used
        to exchange cost or gain information.

        Args:
            cost_table: The factor's cost table.
            incoming_messages: A list of incoming messages.

        Returns:
            A list of outgoing messages, typically containing placeholder cost data.
        """
        if not incoming_messages:
            return []
        factor = incoming_messages[0].recipient
        return [
            Message(data=np.array([0.0]), sender=factor, recipient=msg.sender)
            for msg in incoming_messages
        ]

    def next_iteration(self) -> None:
        """Moves to the next iteration and toggles the decision phase flag."""
        self.iteration += 1
        self.is_decision_phase = not self.is_decision_phase


class DSAComputator(SearchComputator):
    """A computator for the Distributed Stochastic Algorithm (DSA).

    DSA is a simple local search algorithm where each agent decides to change
    its value to the one that yields the best local cost improvement with a
    certain probability.
    """

    def __init__(self, probability: float = 0.7):
        """Initializes the DSA computator.

        Args:
            probability: The probability of changing value when a local
                improvement is found.
        """
        super().__init__()
        self.probability = probability

    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """Computes the DSA decision for an agent.

        The agent evaluates the cost of all possible values in its domain. If a
        value with a lower cost is found, the agent decides to switch to the
        best new value with a probability `p`.

        Args:
            agent: The agent making the decision.
            neighbors_values: A dictionary of neighbor assignments.

        Returns:
            The new value to assign, or the current value if no change is made.
        """
        import random

        curr_value = getattr(agent, "curr_assignment", 0)
        curr_cost = self.evaluate_cost(agent, curr_value, neighbors_values)

        best_value = curr_value
        best_cost = curr_cost

        domain_size = getattr(agent, "domain", 2)
        for value in range(domain_size):
            if value != curr_value:
                cost = self.evaluate_cost(agent, value, neighbors_values)
                if cost < best_cost:
                    best_cost = cost
                    best_value = value

        if best_value != curr_value and random.random() < self.probability:
            return best_value

        return curr_value

    def evaluate_cost(
        self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]
    ) -> float:
        """Evaluates the local cost for an agent given a potential value.

        The cost is the sum of the costs from all factors connected to this agent,
        calculated based on the proposed `value` for the agent and the current
        values of its neighbors.

        Args:
            agent: The agent being evaluated.
            value: The potential value for the agent.
            neighbors_values: A dictionary of neighbor assignments.

        Returns:
            The total local cost.
        """
        total_cost = 0.0
        factors = getattr(agent, "_connected_factors", [])
        for factor in factors:
            if factor.cost_table is not None:
                indices = [None] * len(factor.connection_number)
                valid_lookup = True
                for var_name, dim in factor.connection_number.items():
                    if var_name == agent.name:
                        indices[dim] = value
                    elif var_name in neighbors_values:
                        indices[dim] = neighbors_values[var_name]
                    else:
                        valid_lookup = False
                        break
                if valid_lookup and None not in indices:
                    try:
                        total_cost += factor.cost_table[tuple(indices)]
                    except (IndexError, TypeError):
                        pass
        return total_cost


class MGMComputator(SearchComputator):
    """A computator for the Maximum Gain Message (MGM) algorithm.

    MGM is a local search algorithm where agents coordinate to allow only the
    agent with the maximum local gain to change its value in each iteration.

    The algorithm operates in two phases:
    1.  **Gain Calculation**: Each agent calculates its best possible gain.
    2.  **Decision**: Agents exchange gain values, and only the one with the
        highest gain (with deterministic tie-breaking) makes a move.
    """

    def __init__(self):
        """Initializes the MGM computator."""
        super().__init__()
        self.agent_gains = {}
        self.phase = "gain_calculation"

    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """Computes the MGM decision for an agent based on the current phase.

        Args:
            agent: The agent making the decision.
            neighbors_values: A dictionary of neighbor assignments.

        Returns:
            The new value to assign, or the current value if no change is made.
        """
        if self.phase == "gain_calculation":
            curr_value = getattr(agent, "curr_assignment", 0)
            curr_cost = self.evaluate_cost(agent, curr_value, neighbors_values)
            best_value, best_gain = curr_value, 0.0

            domain_size = getattr(agent, "domain", 2)
            for value in range(domain_size):
                if value != curr_value:
                    cost = self.evaluate_cost(agent, value, neighbors_values)
                    gain = curr_cost - cost
                    if gain > best_gain:
                        best_gain, best_value = gain, value

            self.agent_gains[agent.name] = {
                "gain": best_gain, "best_value": best_value, "current_value": curr_value
            }
            return None  # No decision in this phase
        elif self.phase == "decision":
            agent_info = self.agent_gains.get(agent.name, {})
            agent_gain = agent_info.get("gain", 0.0)
            current_value = agent_info.get("current_value", getattr(agent, "curr_assignment", 0))

            if agent_gain <= 0:
                return current_value

            has_max_gain = True
            neighbor_gains = getattr(agent, "neighbor_gains", {})
            for neighbor_name, neighbor_gain in neighbor_gains.items():
                if neighbor_gain > agent_gain or (
                    neighbor_gain == agent_gain and neighbor_name < agent.name
                ):
                    has_max_gain = False
                    break

            return agent_info.get("best_value", current_value) if has_max_gain else current_value

    def evaluate_cost(
        self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]
    ) -> float:
        """Evaluates the local cost for an agent given a potential value.

        Args:
            agent: The agent being evaluated.
            value: The potential value for the agent.
            neighbors_values: A dictionary of neighbor assignments.

        Returns:
            The total local cost.
        """
        total_cost = 0.0
        factors = getattr(agent, "_connected_factors", [])
        for factor in factors:
            if factor.cost_table is not None:
                indices = [None] * len(factor.connection_number)
                valid_lookup = True
                for var_name, dim in factor.connection_number.items():
                    if var_name == agent.name:
                        indices[dim] = value
                    elif var_name in neighbors_values:
                        indices[dim] = neighbors_values[var_name]
                    else:
                        valid_lookup = False
                        break
                if valid_lookup and None not in indices:
                    try:
                        total_cost += factor.cost_table[tuple(indices)]
                    except (IndexError, TypeError):
                        pass
        return total_cost

    def reset_phase(self) -> None:
        """Resets the computator to the gain calculation phase."""
        self.phase = "gain_calculation"
        self.agent_gains.clear()

    def move_to_decision_phase(self) -> None:
        """Moves the computator to the decision-making phase."""
        self.phase = "decision"


class KOptMGMComputator(SearchComputator):
    """A computator for the K-Optimal Maximum Gain Message (k-opt MGM) algorithm.

    This is an extension of MGM that allows for coalitions of up to `k` agents
    to coordinate and make simultaneous moves, which can help escape local optima.

    The algorithm cycles through three phases:
    1.  **Exploration**: Agents calculate their best individual moves and gains.
    2.  **Coordination**: Agents (via the engine) form coalitions to maximize joint gain.
    3.  **Execution**: Agents in a winning coalition execute their planned moves.
    """

    def __init__(self, k: int = 2, coalition_timeout: int = 10):
        """Initializes the K-Opt MGM computator.

        Args:
            k: The maximum size of a coalition (e.g., k=2 allows pairs of
                agents to move together). Defaults to 2.
            coalition_timeout: The maximum number of iterations to attempt
                forming coalitions. Defaults to 10.
        """
        super().__init__()
        self.k = k
        self.coalition_timeout = coalition_timeout
        self.phase = "exploration"
        self.current_gains = {}
        self.coalition_attempts = 0
        self.coalitions = []
        self.best_values = {}

    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """Computes the k-opt MGM decision based on the current phase.

        Args:
            agent: The agent making the decision.
            neighbors_values: A dictionary of neighbor assignments.

        Returns:
            The new value if a move is executed, otherwise the current value or None.
        """
        if self.phase == "exploration":
            curr_value = getattr(agent, "curr_assignment", 0)
            curr_cost = self.evaluate_cost(agent, curr_value, neighbors_values)
            best_value, best_gain = curr_value, 0.0

            domain_size = getattr(agent, "domain", 2)
            for value in range(domain_size):
                if value != curr_value:
                    new_cost = self.evaluate_cost(agent, value, neighbors_values)
                    gain = curr_cost - new_cost
                    if gain > best_gain:
                        best_gain, best_value = gain, value

            self.current_gains[agent.name] = best_gain
            self.best_values[agent.name] = best_value
            return None
        elif self.phase == "coordination":
            return None
        elif self.phase == "execution":
            for coalition in self.coalitions:
                if agent.name in coalition:
                    return self.best_values.get(agent.name)
            return getattr(agent, "curr_assignment", 0)

    def evaluate_cost(
        self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]
    ) -> float:
        """Evaluates the cost of a potential value assignment.

        Note:
            This is a placeholder and should be implemented by a subclass that
            is specific to the problem being solved.

        Args:
            agent: The agent being evaluated.
            value: The potential value for the agent.
            neighbors_values: A dictionary of neighbor assignments.

        Returns:
            The local cost (defaults to 0.0).
        """
        return 0.0

    def form_coalitions(
        self,
        agents: List[Agent],
        constraints: Dict[Tuple[str, str], Dict[Tuple[Any, Any], float]],
    ) -> List[Set[str]]:
        """Forms coalitions of up to k agents to maximize overall gain.

        This is a greedy implementation where coalitions are built around the
        agents with the highest individual gains.

        Args:
            agents: A list of all agents to consider for coalitions.
            constraints: A dictionary defining the costs between pairs of agents.

        Returns:
            A list of coalitions, where each coalition is a set of agent names.
        """
        coalitions = []
        agent_names = [agent.name for agent in agents]
        available_agents = set(agent_names)
        sorted_agents = sorted(
            agent_names, key=lambda a: self.current_gains.get(a, 0.0), reverse=True
        )

        for seed_agent in sorted_agents:
            if seed_agent not in available_agents:
                continue

            coalition = {seed_agent}
            coalition_gain = self.current_gains.get(seed_agent, 0.0)
            available_agents.remove(seed_agent)

            while len(coalition) < self.k and available_agents:
                best_addition, best_addition_gain = None, 0.0
                for candidate in available_agents:
                    marginal_gain = self.current_gains.get(candidate, 0.0)
                    if marginal_gain > best_addition_gain:
                        best_addition, best_addition_gain = candidate, marginal_gain

                if best_addition and best_addition_gain > 0:
                    coalition.add(best_addition)
                    available_agents.remove(best_addition)
                    coalition_gain += best_addition_gain
                else:
                    break
            if coalition_gain > 0:
                coalitions.append(coalition)
        return coalitions

    def next_iteration(self) -> None:
        """Advances the algorithm to the next phase in its cycle."""
        super().next_iteration()
        if self.phase == "exploration":
            self.phase = "coordination"
        elif self.phase == "coordination":
            self.coalition_attempts += 1
            if self.coalition_attempts >= self.coalition_timeout:
                self.phase = "execution"
                self.coalition_attempts = 0
        elif self.phase == "execution":
            self.phase = "exploration"
            self.current_gains.clear()
            self.coalitions.clear()
            self.best_values.clear()
