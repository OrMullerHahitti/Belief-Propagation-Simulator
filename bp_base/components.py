import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, TypeAlias, Generic, TypeVar, TYPE_CHECKING

from DCOP_base import Agent

if TYPE_CHECKING:
    from bp_base.agents import BPAgent

CostTable: TypeAlias = np.ndarray

A = TypeVar('A', bound=Agent)
S = TypeVar('S', bound=Agent)  # Sender type
R = TypeVar('R', bound=Agent)  # Recipient type


class Message(Generic[S, R]):
    '''
    Represents a message in the BP algorithm.
    Generic over sender and recipient agent types.
    '''
    def __init__(self, data: np.ndarray, sender: S, recipient: R):
        self.data = data
        self.sender = sender
        self.recipient = recipient
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

class BPComputator(ABC):
    @abstractmethod
    def compute_Q(self, messages: List[Message]) -> List[Message]:
        pass
    @abstractmethod
    def compute_R(self, cost_table: np.ndarray, messages: Message) -> Message:
        '''input: cost_table: np.ndarray, messages: List[Message]
        output: List of messages computed from the cost table and the incoming messages for each variable node'''
        pass

class BPMessage(Message["BPAgent", "BPAgent"]):
    pass
