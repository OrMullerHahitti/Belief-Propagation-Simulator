"""
Computator classes for search-based algorithms like DSA and MGM.
These extend the base Computator interface to fit search problems.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Set
import numpy as np

from base_all.DCOP_base import Computator, Agent
from base_all.components import Message


class SearchComputator(Computator, ABC):
    """
    Abstract base class for search-based algorithm computators.
    This class adapts the BP Computator interface for search algorithms.
    """

    def __init__(self):
        """Initialize the search computator."""
        super().__init__()
        self.iteration = 0
        self.is_decision_phase = True  # Toggles between decision and value phases

    @abstractmethod
    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """
        Compute a decision based on neighbors' current values.

        Args:
            agent: The agent making the decision
            neighbors_values: Dictionary mapping neighbor agent names to their current values

        Returns:
            The computed decision value
        """
        pass

    @abstractmethod
    def evaluate_cost(
        self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]
    ) -> float:
        """
        Evaluate the cost of a potential value assignment.

        Args:
            agent: The agent being evaluated
            value: The potential value for this agent
            neighbors_values: Dictionary mapping neighbor agent names to their current values

        Returns:
            The cost (lower is better)
        """
        pass

    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Adapt BP's compute_Q for search algorithms.
        In search algorithms, Q messages usually contain value assignments.

        Args:
            messages: List of incoming messages

        Returns:
            List of outgoing messages
        """
        if not messages:
            return []

        # Default implementation for compatibility
        variable = messages[0].recipient
        outgoing_messages = []

        for msg in messages:
            factor = msg.sender
            # Create a simple reply message with the variable's current assignment
            outgoing_messages.append(
                Message(
                    data=np.array([getattr(variable, "curr_assignment", 0)]),
                    sender=variable,
                    recipient=factor,
                )
            )

        return outgoing_messages

    def compute_R(self, cost_table, incoming_messages: List[Message]) -> List[Message]:
        """
        Adapt BP's compute_R for search algorithms.
        In search algorithms, R messages may contain cost information.

        Args:
            cost_table: The cost table used for computation
            incoming_messages: List of incoming messages

        Returns:
            List of outgoing messages
        """
        if not incoming_messages:
            return []

        # Default implementation for compatibility
        factor = incoming_messages[0].recipient
        outgoing_messages = []

        for msg in incoming_messages:
            variable = msg.sender
            # Create a simple reply message with cost information
            outgoing_messages.append(
                Message(
                    data=np.array([0.0]),  # Placeholder for cost
                    sender=factor,
                    recipient=variable,
                )
            )

        return outgoing_messages

    def next_iteration(self):
        """Move to the next iteration and toggle phase if needed."""
        self.iteration += 1
        # Some search algorithms alternate between decision and value phases
        self.is_decision_phase = not self.is_decision_phase


class DSAComputator(SearchComputator):
    """
    Base class for Distributed Stochastic Algorithm (DSA) computator.
    DSA is a local search algorithm where agents probabilistically decide to change their values.
    """

    def __init__(self, probability: float = 0.7):
        """
        Initialize the DSA computator.

        Args:
            probability: Probability of changing value when an improvement is found
        """
        super().__init__()
        self.probability = probability


class MGMComputator(SearchComputator):
    """
    Base class for Maximum Gain Message (MGM) computator.
    MGM is a local search algorithm where agents coordinate to make the move with maximum gain.
    """

    def __init__(self):
        """Initialize the MGM computator."""
        super().__init__()


class KOptMGMComputator(SearchComputator):
    """
    K-Opt MGM (Maximum Gain Message) computator.

    This is an extension of MGM that allows for k agents to coordinate and make
    simultaneous moves, potentially escaping local optima that standard MGM
    would get stuck in.

    The algorithm has two main phases:
    1. Exploration: Agents explore potential value changes and calculate gains
    2. Coordination: Agents coordinate to form coalitions of size <= k to maximize overall gain
    """

    def __init__(self, k: int = 2, coalition_timeout: int = 10):
        """
        Initialize the K-Opt MGM computator.

        Args:
            k: Maximum coalition size (k=1 is standard MGM, k=2 is MGM-2, etc.)
            coalition_timeout: Maximum iterations to attempt forming coalitions
        """
        super().__init__()
        self.k = k
        self.coalition_timeout = coalition_timeout
        self.phase = "exploration"  # exploration, coordination, or execution
        self.current_gains = {}  # Agent name -> potential gain
        self.coalition_attempts = 0
        self.coalitions = []  # List of formed coalitions
        self.best_values = {}  # Agent name -> best value to switch to

    def compute_decision(self, agent: Agent, neighbors_values: Dict[str, Any]) -> Any:
        """
        Compute decision based on the current phase of k-opt MGM.

        Args:
            agent: The agent making the decision
            neighbors_values: Dictionary of neighbor values

        Returns:
            The computed decision (may be None if no decision yet)
        """
        if self.phase == "exploration":
            # Calculate best local move and gain
            curr_value = getattr(agent, "curr_assignment", 0)
            curr_cost = self.evaluate_cost(agent, curr_value, neighbors_values)

            best_value = curr_value
            best_gain = 0.0

            # Check all possible values
            domain_size = getattr(agent, "domain", 2)
            for value in range(domain_size):
                if value != curr_value:
                    new_cost = self.evaluate_cost(agent, value, neighbors_values)
                    gain = curr_cost - new_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_value = value

            # Store best value and gain
            self.current_gains[agent.name] = best_gain
            self.best_values[agent.name] = best_value

            return None  # No immediate decision in exploration phase

        elif self.phase == "coordination":
            # Determine if this agent should be part of a coalition
            # This requires complex message exchange logic handled by the engine
            return None

        elif self.phase == "execution":
            # Execute the planned move if part of a coalition
            for coalition in self.coalitions:
                if agent.name in coalition:
                    return self.best_values[agent.name]

            # If not in a coalition, keep current value
            return getattr(agent, "curr_assignment", 0)

    def evaluate_cost(
        self, agent: Agent, value: Any, neighbors_values: Dict[str, Any]
    ) -> float:
        """
        Evaluate the cost of a potential value assignment.
        This is problem-specific and should be overridden in subclasses.

        Args:
            agent: The agent being evaluated
            value: The potential value for this agent
            neighbors_values: Dictionary of neighbor values

        Returns:
            The cost (lower is better)
        """
        # Default implementation - should be overridden in specific problem instances
        return 0.0

    def form_coalitions(
        self,
        agents: List[Agent],
        constraints: Dict[Tuple[str, str], Dict[Tuple[Any, Any], float]],
    ) -> List[Set[str]]:
        """
        Form coalitions of up to k agents to maximize overall gain.

        Args:
            agents: List of agents
            constraints: Dictionary mapping agent pairs to their constraint costs
                constraints[(a1, a2)][(v1, v2)] = cost of a1=v1, a2=v2

        Returns:
            List of coalitions (sets of agent names)
        """
        coalitions = []
        agent_names = [agent.name for agent in agents]
        available_agents = set(agent_names)

        # Start with highest gain agents
        sorted_agents = sorted(
            agent_names, key=lambda a: self.current_gains.get(a, 0.0), reverse=True
        )

        # Greedy coalition formation
        for seed_agent in sorted_agents:
            if seed_agent not in available_agents:
                continue

            # Start a new coalition
            coalition = {seed_agent}
            coalition_gain = self.current_gains.get(seed_agent, 0.0)
            available_agents.remove(seed_agent)

            # Try to add more agents to this coalition
            while len(coalition) < self.k and available_agents:
                best_addition = None
                best_addition_gain = 0.0

                for candidate in available_agents:
                    # Calculate marginal gain of adding this agent
                    # This is a simplified estimate - full calculation would need to
                    # evaluate all constraint violations in the coalition
                    marginal_gain = self.current_gains.get(candidate, 0.0)

                    if marginal_gain > best_addition_gain:
                        best_addition = candidate
                        best_addition_gain = marginal_gain

                if best_addition and best_addition_gain > 0:
                    coalition.add(best_addition)
                    available_agents.remove(best_addition)
                    coalition_gain += best_addition_gain
                else:
                    break  # No more beneficial additions

            if coalition_gain > 0:
                coalitions.append(coalition)

        return coalitions

    def next_iteration(self):
        """
        Advance to the next phase in the k-opt MGM algorithm cycle.
        """
        super().next_iteration()

        # Cycle through phases
        if self.phase == "exploration":
            self.phase = "coordination"
        elif self.phase == "coordination":
            self.coalition_attempts += 1
            if self.coalition_attempts >= self.coalition_timeout:
                self.phase = "execution"
                self.coalition_attempts = 0
        elif self.phase == "execution":
            # Reset for next cycle
            self.phase = "exploration"
            self.current_gains.clear()
            self.coalitions.clear()
            self.best_values.clear()
