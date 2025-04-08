from abc import abstractmethod, ABC
from typing import List, TypeAlias, Generic,TypeVar

import numpy as np
from narwhals import Object

from DCOP_base import Agent
from bp_base import BPAgent

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

class Computator():
    ''' an onject that implements the logic of the agent its in'''


class BPMessage(Message[BPAgent]):
    pass
Computator.__doc__ = ''' an oasadasdawrw453535nject that implements the logic of the agent its in'''


