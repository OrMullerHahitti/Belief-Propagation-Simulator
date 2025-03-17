from __future__ import annotations

from abc import ABC,abstractmethod
from typing import Dict, List, List, TypeAlias, Any
import numpy as np
from networkx import Graph
from numpy import ndarray

from DCOP_base import Agent, Computator
from utils.decorators import validate_message_direction
from utils.randomes import create_random_table
Iteration:TypeAlias = Dict[int,List['Message']]
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

class BPComputator(ABC,Computator):
    @abstractmethod
    def compute_Q(self,messages:List[Message]) -> Message:
        pass
    @abstractmethod
    def compute_R(self,cost_table:np.ndarray,messages:Message)->[Message]:
        '''input: cost_table: np.ndarray, messages: List[Message]
        output: List of messages computed from the cost table and the incoming messages for each variable node'''
        pass



class BPAgent(Agent):


    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to message passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self,  name: str, node_type: str, computator:BPComputator):
        super().__init__( name, node_type) # List of connected node IDs
        self.computator = computator
        self.messages: List[Message]
        curr_message:np.ndarray|None = None# Stores incoming messages

    def add_message(self, message:Message) -> None:
        '''mailer uses this function to add a message to the agent'''
        self.messages.add(message)

    @abstractmethod
    def compute_messages(self) -> List[Message]:
        pass






class VariableNode(BPAgent):

    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """


    def __init__(self, node_id: str, name: str, computator: BPComputator, domain_size: int = 3,
                 ):
        """
        :param node_id: Unique identifier
        :param name: Human-readable name
         :param computator: A policy object that implements the logic
                           for computing messages & belief (MinSum, MaxSum, etc.)
        :param domain_size: e.g., length of the domain array

        """
        super().__init__(node_id, name, node_type="variable")
        self.domain_size = domain_size
        # A policy object controlling how messages & belief are computed:
        self.computator = computator

    def compute_messages(self) -> List[Message]:
        """
        Called by the BPAgent framework to compute outgoing messages.
        """
        return self.computator.compute_Q(self.messages)

    def _update_local_variables(self) -> None:
        """
        For example, increment iteration count or do some local update logic.
        """
        self.iteration += 1
    @property
    def belief(self) -> np.ndarray:
        pass

class FactorNode(BPAgent):
    """
    Purpouse: recieve and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, name: str,computator:BPComputator, cost_table :np.ndarray|None=None):
        super().__init__(name, "factor")
      # List of variable node IDs this factor depends on
        self.computator = computator
        if cost_table is not None:
            self.cost_table = cost_table
        else:
            self.cost_table = create_random_table(3)
        self.messages:List[Message]|None=None
        self.to_send :List[Message]|None = None

    @validate_message_direction
    def compute_message(self, message:Message) -> Message:
        return self.computator.compute_R(self.cost_table,message)


    @property
    def mean_cost(self) -> float:
        return np.mean(self.cost_table)


