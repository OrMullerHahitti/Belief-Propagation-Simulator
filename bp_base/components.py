from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, TypeAlias, Generic,TypeVar

import numpy as np

from DCOP_base import Agent


CostTable: TypeAlias = np.ndarray

A = TypeVar('A', bound=Agent)


class Message(Generic[A]):
    '''
    Represents a message in the BP algorithm.

    '''
    def __init__(self,data:np.ndarray,sender:A,recipient:A):
        self.data = data
        self.sender = sender
        self.recipient = recipient
    def __hash__(self):
        return hash((self.sender,self.recipient))
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
    def compute_Q(self,messages:List[Message]) -> List[Message]:
        pass
    @abstractmethod
    def compute_R(self,cost_table:np.ndarray,messages:Message)->Message:
        '''input: cost_table: np.ndarray, messages: List[Message]
        output: List of messages computed from the cost table and the incoming messages for each variable node'''
        pass
@dataclass
class History():
    '''A class to hold the history of the messages sent and received by each agent'''
    messages_sent:List[Message]
    messages_received:List[Message]
    beliefs:List[np.ndarray]
    map_estimates:List[np.ndarray]
    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.beliefs = []
        self.map_estimates = []
