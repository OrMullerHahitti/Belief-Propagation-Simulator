from abc import abstractmethod, ABC
from typing import List

import numpy as np

from DCOP_base import Agent


class Message():
    def __init__(self,message:np.ndarray,sender:Agent,recipient:Agent):
        self.message = message
        self.sender = sender
        self.recipient = recipient
    def __hash__(self):
        return hash((self.sender,self.recipient))
    def __eq__(self, other):
        return self.sender == other.sender and self.recipient == other.recipient
    def __ne__(self, other):
        return not self == other
    def __str__(self):
        return f"Message from {self.sender.name} to {self.recipient.name}: {self.message}"
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
