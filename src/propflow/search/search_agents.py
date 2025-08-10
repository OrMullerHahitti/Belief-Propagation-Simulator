"""
Search-specific agent extensions.
This module extends base agents with search-specific functionality.
"""

from typing import Dict, Any, List, Optional
import logging

from ..core.agents import VariableAgent as BaseVariableAgent
from .search_computator import SearchComputator

logger = logging.getLogger(__name__)


class SearchVariableAgent(BaseVariableAgent):
    """
    Extension of VariableAgent with search-specific capabilities.

    This class adds methods needed by search algorithms like DSA and MGM.
    """

    def __init__(self, name: str, domain: int):
        super().__init__(name, domain)
        self._connected_factors = []  # List of connected factors
        self.neighbor_gains = {}  # For MGM coordination
        self._pending_assignment = None  # For delayed assignment updates

    def set_connected_factors(self, factors: List):
        """Set the list of connected factors for cost evaluation."""
        self._connected_factors = factors

    def get_neighbor_values(self, graph) -> Dict[str, Any]:
        """
        Get current assignments of neighboring variable agents.

        Args:
            graph: The factor graph containing this agent

        Returns:
            Dictionary mapping neighbor names to their current assignments
        """
        neighbors_values = {}

        # Get neighbors through the graph
        if hasattr(graph, "G"):
            for neighbor in graph.G.neighbors(self):
                if hasattr(neighbor, "curr_assignment") and hasattr(neighbor, "name"):
                    # Only include variable agents, not factor agents
                    if getattr(neighbor, "type", "") == "variable":
                        neighbors_values[neighbor.name] = neighbor.curr_assignment

        return neighbors_values

    def compute_search_step(
        self, neighbors_values: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Compute the next value for this agent using the search computator.

        Args:
            neighbors_values: Optional dict of neighbor values (will be computed if not provided)

        Returns:
            The computed next value, or None if no change
        """
        if not isinstance(self.computator, SearchComputator):
            logger.warning(f"Agent {self.name} does not have a SearchComputator")
            return None

        if neighbors_values is None:
            # This will need to be provided by the engine in most cases
            neighbors_values = {}

        try:
            decision = self.computator.compute_decision(self, neighbors_values)
            self._pending_assignment = decision
            return decision
        except Exception as e:
            logger.error(f"Error computing search step for {self.name}: {e}")
            return None

    def update_assignment(self):
        """
        Update the agent's assignment based on the pending decision.
        This is called after all agents have computed their search steps.
        """
        if self._pending_assignment is not None:
            # Only update if the value actually changed
            if self._pending_assignment != self.curr_assignment:
                old_value = self.curr_assignment
                # Note: curr_assignment is a property that uses argmin(belief)
                # For search algorithms, we need to directly set the assignment
                # We'll need to override the property or use a different approach
                self._assignment = self._pending_assignment
                logger.debug(
                    f"Agent {self.name} changed from {old_value} to {self._pending_assignment}"
                )

            self._pending_assignment = None

    @property
    def curr_assignment(self) -> int:
        """Override to use direct assignment in search mode."""
        if hasattr(self, "_assignment"):
            return self._assignment
        else:
            # Fall back to belief-based assignment
            return super().curr_assignment

    @curr_assignment.setter
    def curr_assignment(self, value: int):
        """Allow direct assignment setting."""
        self._assignment = value


def extend_variable_agent_for_search(agent: BaseVariableAgent) -> SearchVariableAgent:
    """
    Convert a base VariableAgent to a SearchVariableAgent.

    This function can be used to extend existing agents with search capabilities.

    Args:
        agent: The base variable agent to extend

    Returns:
        Extended search variable agent
    """
    # Create new search agent with same properties
    search_agent = SearchVariableAgent(agent.name, agent.domain)

    # Copy over important attributes
    if hasattr(agent, "computator"):
        search_agent.computator = agent.computator
    if hasattr(agent, "mailer"):
        search_agent.mailer = agent.mailer
    if hasattr(agent, "_history"):
        search_agent._history = agent._history
    if hasattr(agent, "_assignment"):
        search_agent._assignment = agent._assignment

    return search_agent
