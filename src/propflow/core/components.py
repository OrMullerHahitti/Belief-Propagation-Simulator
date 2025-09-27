from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from typing import List, TypeAlias, TYPE_CHECKING

from .dcop_base import Agent

if TYPE_CHECKING:
    from .agents import FGAgent

CostTable: TypeAlias = np.ndarray


class Message:
    """Represents a message passed between agents in the belief propagation algorithm.

    Attributes:
        data (np.ndarray): The content of the message, typically a numpy array
                           representing costs or beliefs.
        sender (Agent): The agent sending the message.
        recipient (Agent): The agent receiving the message.
    """

    def __init__(self, data: np.ndarray, sender: Agent, recipient: Agent):
        """Initializes a Message instance.

        Args:
            data (np.ndarray): The message content.
            sender (Agent): The sender of the message.
            recipient (Agent): The recipient of the message.
        """
        self.data = data
        self.sender = sender
        self.recipient = recipient

    def copy(self) -> Message:
        """Creates a deep copy of the message.

        Returns:
            Message: A new `Message` instance with copied data.
        """
        return Message(
            data=np.copy(self.data), sender=self.sender, recipient=self.recipient
        )

    def __hash__(self) -> int:
        """Computes the hash of the message based on sender and recipient names."""
        return hash((self.sender.name, self.recipient.name))

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on sender and recipient names."""
        if not isinstance(other, Message):
            return NotImplemented
        return (
            self.sender.name == other.sender.name
            and self.recipient.name == other.recipient.name
        )

    def __ne__(self, other: object) -> bool:
        """Checks for inequality."""
        return not self == other

    def __str__(self) -> str:
        """Returns a human-readable string representation of the message."""
        return f"Message from {self.sender.name} to {self.recipient.name}: {self.data}"

    def __repr__(self) -> str:
        """Returns a detailed string representation of the message."""
        return self.__str__()


class MailHandler:
    """Handles message passing with deduplication and synchronization.

    This class manages an agent's incoming and outgoing messages, ensuring that
    only the latest message from each sender is stored.

    Attributes:
        pruning_policy: An optional policy for selectively discarding messages.
    """

    def __init__(self, _domain_size: int):
        """Initializes the MailHandler.

        Args:
            _domain_size (int): The domain size for messages, used to initialize
                                empty messages.
        """
        self._message_domain_size = _domain_size
        self._incoming: Dict[str, Message] = {}  # Key: sender_key, Value: message
        self._outgoing: List[Message] = []
        self._clear_after_staging = True

    def set_pruning_policy(self, policy) -> None:
        """Sets a message pruning policy.

        Args:
            policy: An object with a `should_accept_message` method.
        """
        self.pruning_policy = getattr(self, "pruning_policy", None)
        self.pruning_policy = policy

    def _make_key(self, agent: Agent) -> str:
        """Creates a unique key for an agent to prevent collisions.

        Args:
            agent (Agent): The agent for which to create a key.

        Returns:
            str: A unique string identifier for the agent.
        """
        return f"{agent.name}_{agent.type}"

    def set_first_message(self, owner: FGAgent, neighbor: FGAgent) -> None:
        """Initializes the inbox with a zero-message from a neighbor.

        This is used to ensure that an agent has a message from each neighbor
        before computation begins.

        Args:
            owner (FGAgent): The agent who owns this mail handler.
            neighbor (FGAgent): The neighboring agent to initialize a message from.
        """
        key = self._make_key(neighbor)

        # Default initialization with zeros
        self._incoming[key] = Message(
            np.zeros(self._message_domain_size),
            neighbor,
            owner,
        )

    def receive_messages(self, messages: Message | list[Message]) -> None:
        """Receives and handles one or more messages.

        Applies a pruning policy if one is set and stores the message,
        overwriting any previous message from the same sender.

        Args:
            messages: A single `Message` or a list of `Message` objects.
        """
        if isinstance(messages, list):
            for message in messages:
                self.receive_messages(message)
            return

        message = messages

        # Check for pruning policy
        if hasattr(self, "pruning_policy") and self.pruning_policy is not None:
            owner = message.recipient

            if not self.pruning_policy.should_accept_message(owner, message):
                return  # Message pruned

        # Accept message
        key = self._make_key(message.sender)
        self._incoming[key] = message

    def send(self) -> None:
        """Sends all staged outgoing messages to their recipients."""
        for message in self._outgoing:
            message.recipient.mailer.receive_messages(message)

    def stage_sending(self, messages: List[Message]) -> None:
        """Stages a list of messages to be sent.

        Args:
            messages (List[Message]): The messages to be sent.
        """
        self._outgoing = messages.copy()

    def prepare(self) -> None:
        """Clears the outbox, typically after messages have been sent."""
        self._outgoing.clear()

    def clear_inbox(self) -> None:
        """Clears all messages from the inbox."""
        self._incoming.clear()

    def clear_outgoing(self) -> None:
        """Clears all messages from the outbox."""
        self._outgoing.clear()

    @property
    def inbox(self) -> List[Message]:
        """list[Message]: A list of incoming messages."""
        return list(self._incoming.values())

    @inbox.setter
    def inbox(self, li: List[Message]) -> None:
        """Sets the inbox from a list of messages.

        Args:
            li (List[Message]): A list of messages to populate the inbox with.
        """
        self._incoming.clear()
        for msg in li:
            key = self._make_key(msg.sender)
            self._incoming[key] = msg

    @property
    def outbox(self) -> List[Message]:
        """list[Message]: A list of outgoing messages."""
        return self._outgoing

    @outbox.setter
    def outbox(self, li: List[Message]) -> None:
        """Sets the outbox from a list of messages.

        Args:
            li (List[Message]): A list of messages to populate the outbox with.
        """
        self._outgoing = li

    def __getitem__(self, sender_name: str) -> Optional[Message]:
        """Retrieves a message from the inbox by the sender's name.

        Args:
            sender_name (str): The name of the sender.

        Returns:
            The `Message` object if found, otherwise `None`.
        """
        for key, msg in self._incoming.items():
            if msg.sender.name == sender_name:
                return msg
        return None

    def __len__(self) -> int:
        """Returns the number of messages in the inbox."""
        return len(self._incoming)

    def __iter__(self):
        """Returns an iterator over the messages in the inbox."""
        return iter(self.inbox)
