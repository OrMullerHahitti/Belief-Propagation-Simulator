from __future__ import annotations

from abc import ABC


class Agent(ABC):
    """The top-level abstract base class for any node in the DCOP.

    Attributes:
        name (str): A human-readable name for the node.
        type (str): The type of the node (e.g., 'variable', 'factor').
        mailer: A mailer instance for handling message passing.
    """

    def __init__(self, name: str, node_type: str = "general"):
        """Initializes an Agent.

        Args:
            name (str): The name of the agent.
            node_type (str): The type of the agent. Defaults to "general".
        """
        self.name = name
        self.type = node_type
        self._computator = None
        self.mailer = None

    @property
    def computator(self):
        return self._computator

    @computator.setter
    def computator(self, computator) -> None:
        self._computator = computator

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on name and type."""
        if not isinstance(other, Agent):
            return NotImplemented
        return self.name == other.name and self.type == other.type

    def __hash__(self) -> int:
        """Computes the hash based on name and type."""
        return hash((self.name, self.type))

    def __repr__(self) -> str:
        return f"Agent({self.name}, {self.type})"
