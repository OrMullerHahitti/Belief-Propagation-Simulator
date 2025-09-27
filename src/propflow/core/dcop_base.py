from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np

from .protocols import Message


class Computator(ABC):
    """Abstract base class for a Computator.

    This class defines the interface for computing messages in a Distributed
    Constraint Optimization Problem (DCOP) system. Subclasses must implement
    the `compute_Q` and `compute_R` methods.
    """

    def __init_subclass__(cls, **kwargs):
        """Initializes a subclass, can be used for registration or setup."""
        super().__init_subclass__(**kwargs)

    @abstractmethod
    async def compute_Q(self, messages: List[Message]) -> List[Message]:
        """Computes outgoing messages from a variable node.

        Args:
            messages: A list of incoming messages from factor nodes.

        Returns:
            A list of computed messages to be sent to factor nodes.
        """
        pass

    @abstractmethod
    async def compute_R(
        self, cost_table: np.ndarray, incoming_messages: List[Message]
    ) -> List[Message]:
        """Computes outgoing messages from a factor node.

        Args:
            cost_table: The cost table associated with the factor.
            incoming_messages: A list of incoming messages from variable nodes.

        Returns:
            A list of computed messages to be sent to variable nodes.
        """
        pass


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
        self._computator: Computator | None = None
        self.mailer = None

    @property
    def computator(self) -> Computator | None:
        """Computator | None: The computator used by this agent."""
        return self._computator

    @computator.setter
    def computator(self, computator: Computator) -> None:
        """Sets the computator for this agent.

        Args:
            computator: The computator instance to be set.
        """
        self._computator = computator

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on name and type."""
        if not isinstance(other, Agent):
            return NotImplemented
        return self.name == other.name and self.type == other.type

    def __hash__(self) -> int:
        """Computes the hash based on name and type."""
        try:
            name_val = self.name
        except AttributeError:
            name_val = str(id(self))

        try:
            type_val = self.type
        except AttributeError:
            type_val = "unknown"

        return hash((name_val, type_val))

    def __repr__(self) -> str:
        """Returns a string representation of the Agent."""
        return f"Agent({self.name}, {self.type})"


class Mailer(Agent):
    """A simple mailer class for message passing between agents.

    This class provides basic functionality for sending, retrieving, and
    clearing messages.

    Note:
        This class appears to be a simpler, possibly legacy, alternative to
        `MailHandler`.

    Attributes:
        mailbox (dict): A dictionary to store messages, keyed by recipient name.
    """

    def __init__(self):
        """Initializes the Mailer."""
        super().__init__("mailer", "mailer")
        self.mailbox = {}

    def send_message(self, recipient: Agent, message: Any) -> None:
        """Sends a message to a specific recipient.

        Args:
            recipient: The agent to receive the message.
            message: The message content.
        """
        if recipient.name in self.mailbox:
            self.mailbox[recipient.name].append(message)
        else:
            self.mailbox[recipient.name] = [message]

    def retrieve_messages(self, recipient: Agent) -> List[Any]:
        """Retrieves all messages for a specific recipient.

        Args:
            recipient: The agent whose messages are to be retrieved.

        Returns:
            A list of messages for the recipient, or an empty list if none.
        """
        if recipient.name in self.mailbox:
            return self.mailbox[recipient.name]
        else:
            return []

    def clear_mailbox(self) -> None:
        """Clears all messages from the mailbox."""
        self.mailbox = {}
