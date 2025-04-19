from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Set, TypeAlias
import numpy as np

from utils.randomes import create_random_table
from networkx import Graph
Computator :TypeAlias =  Any
class AbstractGraphSystem(ABC):
    """
    Abstract base class for any graph-like system.
    Defines common operations such as adding agents and connecting them.
    """
    @abstractmethod
    def add_agent(self, agent: Agent) -> None:
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
class Agent(ABC):
    """
    The top-level abstract base class for any node in the DCOP problem.
    """
    def __init__(self, name: str, node_type: str):
        self.name = name  # Human-readable name for the node
        self.type = node_type  # Type of the node (e.g., 'variable', 'factor')
        self._computator = None
    @property
    def computator(self):
        return self._computator

    @computator.setter
    def computator(self, computator: Computator) -> None:
        """
        Set the computator for this agent.
        :param computator: The computator to be set.
        """
        self._computator = computator

    def __eq__(self, other):
        return self.name == other.name and self.type != other.type

    def __hash__(self):
        return hash((self.name,self.type))
#mailer class that will handle recieveing and sending the messages to the right nodes
class Mailer(Agent):
    def __init__(self):
        super().__init__("mailer", "mailer")
        self.mailbox = {}


    def send_message(self, recipient: Agent, message: Any) -> None:
        if recipient.name in self.mailbox:
            self.mailbox[recipient.name].append(message)
        else:
            self.mailbox[recipient.name] = [message]

    def retrieve_messages(self, recipient: Agent) -> List[Any]:
        if recipient.name in self.mailbox:
            return self.mailbox[recipient.name]
        else:
            return []

    def clear_mailbox(self) -> None:
        self.mailbox = {}

Policy : TypeAlias = Any


