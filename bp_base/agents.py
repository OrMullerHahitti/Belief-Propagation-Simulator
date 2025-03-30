from __future__ import annotations

from abc import ABC,abstractmethod
from typing import Dict, List, TypeAlias
import numpy as np

from bp_base.components import Message, BPComputator, CostTable
from DCOP_base import Agent
from saved_for_later.decorators import validate_message_direction
from utils.randomes import create_random_table

from config.hyper_parameters_config import MESSAGE_DOMAIN_SIZE, CT_CREATION_FUNCTION, CT_CREATION_PARAMS,COMPUTATOR

class BPAgent(Agent):


    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to data passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self,  name: str, node_type: str):
        super().__init__( name, node_type)
        self.domain = MESSAGE_DOMAIN_SIZE
        self.computator = COMPUTATOR()
        self.messages_before_compute: List[Message] =[]
        self.messages_after_compute: List[Message] =[]
        self.domains:Dict[BPAgent,int] ={}


    def receive_message(self, message:Message["BPAgent"]) -> None:
        '''mailer uses this function to add a data to the agent'''
        self.messages_before_compute.append(message)
    def send_message(self, message:Message["BPAgent"]) -> None:
        message.recipient.receive_message(message)





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

    def compute_messages(self) -> None:
        """
        Called by the BPAgent framework to compute outgoing messages.
        """
        self.messages_after_compute= self.computator.compute_Q(self.messages_before_compute)

    def _update_local_variables(self) -> None:
        """
        For example, increment iteration count or do some local update logic.
        """
        pass
    #TODO create the self belief function

    # @property
    # def belief(self) -> np.ndarray:
    #     pass
    # __repr__ = lambda self: f"VariableAgent: {self.name}"

class FactorAgent(BPAgent):
    """
    Purpose: receive and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, name: str):
        super().__init__(name, "factor")
        self.cost_table :CostTable|None = None
        self.connection_number : Dict[VariableAgent,int] = {}


    def compute_message(self, messages:List[Message["BPAgent"]]) -> List[Message["BPAgent"]]:
        """
        Compute the message to be sent to the variable node.
        :param messages: List of incoming messages from variable nodes.
        :return:
        """
        return self.computator.compute_R(self.cost_table,messages)


    def _update_cost_table(self, domain:int, *args, **kwargs) -> None:
        """
        Create a cost table based on the specified distribution and parameters given the domain after connections
        """
        if self.cost_table is not None:
            raise ValueError("Cost table already exists. Cannot create a new one.")
        self.cost_table = CT_CREATION_FUNCTION(len(self.connection_number),MESSAGE_DOMAIN_SIZE,**CT_CREATION_PARAMS)

    #TODO :fix the self naming after creating agents

    # @property
    # def name(self) -> str:
    #     if self.domains is None:
    #         return self.name
    #     return f'f{''.join(str(i) for i in self.domains.keys()[1:])}_'


    @property
    def mean_cost(self) -> float:
        return np.mean(self.cost_table)
    def compute_messages(self) -> None:
        self.messages_after_compute = self.computator.compute_Q(self.messages_before_compute)
    def __repr__(self):
        return f"FactorAgent: {self.name}"


