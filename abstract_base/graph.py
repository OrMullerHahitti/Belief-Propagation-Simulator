from abc import ABC, abstractmethod
from typing import Any


class AbstractGraphSystem(ABC):
    """
    Abstract base class for any graph-like system.
    Defines common operations such as adding agents and connecting them.
    """
    @abstractmethod
    def add_agent(self, agent: Any) -> None:
        """
        Add an agent (node) to the graph system.
        """
        pass

    @abstractmethod
    def connect(self, agent1_name: str, agent2_name: str) -> None:
        """
        Connect two agents (nodes) in the graph system.
        """
        pass

    @abstractmethod
    def get_agents(self) -> Any:
        """
        Retrieve all agents (nodes) in the system.
        """
        pass
