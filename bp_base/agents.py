from __future__ import annotations

from abc import ABC,abstractmethod
from typing import Dict, List, TypeAlias
import numpy as np

from bp_base.components import Message, BPComputator
from DCOP_base import Agent
from saved_for_later.decorators import validate_message_direction
from utils.randomes import create_random_table
Iteration:TypeAlias = Dict[int,List['Message']]




class BPAgent(Agent):


    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to data passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self,  name: str, node_type: str, computator:BPComputator|None=None):
        super().__init__( name, node_type) # List of connected node IDs
        self.computator = computator
        self.messages: List[Message] =[]
        self.domains:Dict[BPAgent,int] ={}
        curr_message:np.ndarray|None = None# Stores incoming messages

    def add_message(self, message:Message) -> None:
        '''mailer uses this function to add a data to the agent'''
        self.messages.add(message)
    def update_computatpr(self,computator:BPComputator) -> None:
        self.computator = computator

    def add_domain(self,other:BPAgent,domain:int) -> None:
        ''' uses this function to add a data to the agent'''
        other.domains[self] = domain
        self.domains[other] == domain








class VariableAgent(BPAgent):

    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """


    def __init__(self, name: str, domain_size: int = 3,):
        """
        :param node_id: Unique identifier
        :param name: Human-readable nam
        :param domain_size: e.g., length of the domain array

        """

        super().__init__(name, node_type="variable")
        self.domain_size = domain_size

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
    __repr__ = lambda self: f"VariableAgent: {self.name}"

class FactorAgent(BPAgent):
    """
    Purpose: receive and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, name: str, cost_table :np.ndarray|None=None):
        super().__init__(name, "factor")
      # List of variable node IDs this factor depends on

        if cost_table is not None:
            self.cost_table = cost_table
        else:
            self.cost_table = create_random_table(3)

    def compute_message(self, message:Message) -> Message:
        return self.computator.compute_R(self.cost_table,message)


    @property
    def mean_cost(self) -> float:
        return np.mean(self.cost_table)
    def compute_messages(self) -> List[Message]:
        return []
    def __repr__(self):
        return f"FactorAgent: {self.name}"


