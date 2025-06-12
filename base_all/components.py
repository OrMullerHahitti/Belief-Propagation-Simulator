from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from functools import singledispatchmethod
from typing import List, TypeAlias, TYPE_CHECKING

from base_all.DCOP_base import Agent

if TYPE_CHECKING:
    from base_all.agents import FGAgent

CostTable: TypeAlias = np.ndarray


class Message:
    """
    Represents a message in the BP algorithm.
    """

    def __init__(self, data: np.ndarray, sender: Agent, recipient: Agent):
        self.data = data
        self.sender = sender
        self.recipient = recipient

    def copy(self) -> Message:
        """
        Create a copy of this message with a new data array.
        """
        return Message(
            data=np.copy(self.data), sender=self.sender, recipient=self.recipient
        )

    def __hash__(self):
        return hash((self.sender.name, self.recipient.name))

    def __eq__(self, other):
        return (
            self.sender.name == other.sender.name
            and self.recipient.name == other.recipient.name
        )

    def __ne__(self, other):
        return self != other

    def __str__(self):
        return f"Message from {self.sender.name} to {self.recipient.name}: {self.data}"

    def __repr__(self):
        return self.__str__()


class MailHandler:
    """
    Handles message passing with proper deduplication and synchronization.
    """

    def __init__(self, _domain_size: int):
        self._message_domain_size = _domain_size
        self._incoming: Dict[str, Message] = {}  # Key: sender_key, Value: message
        self._outgoing: List[Message] = []
        self._clear_after_staging = True

    def set_pruning_policy(self, policy):
        """Set message pruning policy."""
        self.pruning_policy = getattr(self, "pruning_policy", None)
        self.pruning_policy = policy

    def _make_key(self, agent: Agent) -> str:
        """Create unique key for agent to handle identity issues."""
        return f"{agent.name}_{agent.type}"

    def set_first_message(self, owner: FGAgent, neighbor: FGAgent) -> None:
        """
        Initialize with zero message from neighbor.
        Supports both original neighbors and split factor neighbors (with ' or '' suffix).
        """
        key = self._make_key(neighbor)

        # Default initialization with zeros
        self._incoming[key] = Message(
            np.zeros(self._message_domain_size),
            neighbor,
            owner,
        )

    @singledispatchmethod
    def receive_messages(self, message: Message):
        """Handle a single Message with optional pruning."""
        # Check for pruning policy
        if hasattr(self, "pruning_policy") and self.pruning_policy is not None:
            owner = message.recipient

            if not self.pruning_policy.should_accept_message(owner, message):
                return  # Message pruned

        # Accept message
        key = self._make_key(message.sender)
        self._incoming[key] = message

    @receive_messages.register(list)
    def _(self, messages: list[Message]):
        """Handle a list of Messages."""
        for message in messages:
            self.receive_messages(message)

    def send(self):
        """Send all outgoing messages to their recipients."""
        for message in self._outgoing:
            message.recipient.mailer.receive_messages(message)

    def stage_sending(self, messages: List[Message]):
        """Stage messages for sending."""
        self._outgoing = messages.copy()

    def prepare(self):
        """Clear outgoing messages after sending."""
        self._outgoing.clear()

    def clear_inbox(self):
        """Clear the inbox."""
        self._incoming.clear()

    def clear_outgoing(self):
        """Clear outgoing messages."""
        self._outgoing.clear()

    @property
    def inbox(self) -> List[Message]:
        """Return messages as list, sorted by sender name for consistency."""
        return sorted(self._incoming.values(), key=lambda m: m.sender.name)

    @inbox.setter
    def inbox(self, li: List[Message]):
        """Set inbox from list of messages."""
        self._incoming.clear()
        for msg in li:
            key = self._make_key(msg.sender)
            self._incoming[key] = msg

    @property
    def outbox(self) -> List[Message]:
        return self._outgoing

    @outbox.setter
    def outbox(self, li: List[Message]):
        self._outgoing = li

    def __getitem__(self, sender_name: str) -> Optional[Message]:
        """Get message by sender name."""
        for key, msg in self._incoming.items():
            if msg.sender.name == sender_name:
                return msg
        return None

    def __len__(self):
        return len(self._incoming)

    def __iter__(self):
        return iter(self.inbox)
