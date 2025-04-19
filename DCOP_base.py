from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Set, TypeAlias
import numpy as np

from bp_base.components import Message
from utils.randomes import create_random_table
from networkx import Graph
class Computator(ABC):
    """
    Abstract base class for a Computator.
    This class defines the interface for computing messages in a DCOP system.
    """
    def __init__(self):
        """
        Initialize the Computator.
        This method can be overridden in subclasses to perform additional initialization.
        """
        pass
    def __init_subclass__(cls, **kwargs):
        """
        This method is called when a subclass is created.
        It can be used to register the subclass in a registry or perform other initialization tasks.
        """
        super().__init_subclass__(**kwargs)


    @abstractmethod
    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Compute Q messages based on the incoming messages.
        :param messages: List of incoming messages.
        :return: List of computed Q messages.
        """
        pass

    @abstractmethod
    def compute_R(self, cost_table: np.ndarray, message: List[Message]) -> List[Message]:
        """
        Compute R messages based on the cost table and incoming message.
        :param cost_table: The cost table to be used for computation.
        :param message: The incoming message.
        :return: The computed R message.
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


