from __future__ import annotations
from typing import Dict, List, Optional, Protocol, Any, Union, TypeVar, Generic
import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, TypeAlias, TYPE_CHECKING

from DCOP_base import Agent

if TYPE_CHECKING:
    from bp_base.agents import BPAgent

CostTable: TypeAlias = np.ndarray

class Message:
    '''
    Represents a message in the BP algorithm.
    '''
    def __init__(self, data: np.ndarray, sender: Agent, recipient: Agent):
        self.data = data
        self.sender = sender
        self.recipient = recipient
    
    def copy(self) -> Message:
        '''
        Create a copy of this message with a new data array.
        :return: A new Message object with same sender, recipient and a copy of data
        '''
        return Message(
            data=np.copy(self.data),
            sender=self.sender,
            recipient=self.recipient
        )
        
    def __hash__(self):
        return hash((self.sender, self.recipient))
    def __eq__(self, other):
        return self.sender == other.sender and self.recipient == other.recipient
    def __ne__(self, other):
        return not self == other
    def __str__(self):
        return f"Message from {self.sender.name} to {self.recipient.name}: {self.data}"
    def __repr__(self):
        return self.__str__()


class MessageBox:
    def __init__(self):
        self._messages: List[Message] = []

    def add(self, message: Message):
        """Add a message, replacing any from the same sender."""
        for idx, existing in enumerate(self._messages):
            if existing.sender == message.sender:
                self._messages[idx] = message
                return
        self._messages.append(message)

    def get_all(self) -> List[Message]:
        return list(self._messages)

    def clear(self):
        self._messages.clear()

    def get_from(self, sender) -> Optional[Message]:
        for msg in self._messages:
            if msg.sender == sender:
                return msg
        return None

    def __len__(self):
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)


class BPMessage(Message):
    pass

