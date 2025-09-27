"""Agent Classes for Search-Based DCOP Algorithms.

This module provides extensions to the core agent classes, adapting them for
use in local search algorithms like DSA and MGM, rather than belief propagation.
"""

from typing import Dict, Any, List, Optional
import logging

from ..core.agents import VariableAgent as BaseVariableAgent
from .search_computator import SearchComputator

logger = logging.getLogger(__name__)


class SearchVariableAgent(BaseVariableAgent):
    """An extension of `VariableAgent` with capabilities for search algorithms.

    This class overrides and adds methods required by local search algorithms,
    such as directly computing a new assignment based on neighbor states
    rather than using belief propagation messages.

    Attributes:
        neighbor_gains (dict): A dictionary to store gains for coordinating
            with neighbors, used in algorithms like MGM.
    """

    def __init__(self, name: str, domain: int):
        """Initializes the SearchVariableAgent.

        Args:
            name: The name of the agent.
            domain: The size of the agent's domain.
        """
        super().__init__(name, domain)
        self._connected_factors = []
        self.neighbor_gains = {}
        self._pending_assignment = None

    def set_connected_factors(self, factors: List[Any]) -> None:
        """Sets the list of connected factors for local cost evaluation.

        Args:
            factors: A list of factor objects connected to this agent.
        """
        self._connected_factors = factors

    def get_neighbor_values(self, graph: Any) -> Dict[str, Any]:
        """Gets the current assignments of neighboring variable agents.

        Args:
            graph: The factor graph containing this agent.

        Returns:
            A dictionary mapping neighbor names to their current assignments.
        """
        neighbors_values = {}
        if hasattr(graph, "G"):
            for neighbor in graph.G.neighbors(self):
                if getattr(neighbor, "type", "") == "variable":
                    neighbors_values[neighbor.name] = neighbor.curr_assignment
        return neighbors_values

    def compute_search_step(
        self, neighbors_values: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Computes the best next value for this agent using its search computator.

        This method defers the decision-making to the attached `SearchComputator`
        and stores the result as a pending assignment.

        Args:
            neighbors_values: An optional dictionary of neighbor values. If not
                provided, it will be computed.

        Returns:
            The computed next value, or `None` if no change or an error occurs.
        """
        if not isinstance(self.computator, SearchComputator):
            logger.warning(f"Agent {self.name} does not have a SearchComputator")
            return None

        neighbors_values = neighbors_values or {}

        try:
            decision = self.computator.compute_decision(self, neighbors_values)
            self._pending_assignment = decision
            return decision
        except Exception as e:
            logger.error(f"Error computing search step for {self.name}: {e}")
            return None

    def update_assignment(self) -> None:
        """Updates the agent's assignment based on the pending decision.

        This is intended to be called after all agents have computed their
        search steps in a synchronous update scheme.
        """
        if self._pending_assignment is not None:
            if self._pending_assignment != self.curr_assignment:
                old_value = self.curr_assignment
                self._assignment = self._pending_assignment
                logger.debug(
                    f"Agent {self.name} changed from {old_value} to {self._pending_assignment}"
                )
            self._pending_assignment = None

    @property
    def curr_assignment(self) -> int:
        """Overrides the base property to allow for direct assignment storage.

        In search algorithms, the assignment is set directly, not derived from
        beliefs. This property falls back to the belief-based assignment if no

        direct assignment has been set.
        """
        return getattr(self, '_assignment', super().curr_assignment)

    @curr_assignment.setter
    def curr_assignment(self, value: int) -> None:
        """Allows for direct setting of the agent's assignment."""
        self._assignment = value


def extend_variable_agent_for_search(agent: BaseVariableAgent) -> SearchVariableAgent:
    """Converts a `BaseVariableAgent` to a `SearchVariableAgent`.

    This utility function facilitates the process of adapting a factor graph
    initially created for belief propagation to be used with a search-based
    algorithm.

    Args:
        agent: The base variable agent to convert.

    Returns:
        A new `SearchVariableAgent` instance with the properties of the original.
    """
    search_agent = SearchVariableAgent(agent.name, agent.domain)

    # Copy over important attributes to maintain state.
    if hasattr(agent, "computator"):
        search_agent.computator = agent.computator
    if hasattr(agent, "mailer"):
        search_agent.mailer = agent.mailer
    if hasattr(agent, "_history"):
        search_agent._history = agent._history
    if hasattr(agent, "_assignment"):
        search_agent._assignment = agent._assignment

    return search_agent
