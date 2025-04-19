from __future__ import annotations

from abc import ABC,abstractmethod
from typing import Dict, List, TypeAlias, Any,Callable
import numpy as np


from bp_base.components import Message, CostTable
from bp_base.computators import BPComputator
from DCOP_base import Agent
from saved_for_later.decorators import validate_message_direction
from utils.randomes import create_random_table

from configs.hyper_parameters_config import MESSAGE_DOMAIN_SIZE, CT_CREATION_FUNCTION, CT_CREATION_PARAMS,COMPUTATOR

class BPAgent(Agent,ABC):

    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to data passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self,  name: str, node_type: str,domain:int ,computator:BPComputator):
        ### --- attributes --- ###
        super().__init__( name, node_type)
        self.domain = domain
        self.computator = computator
        ### --- message handling --- ###
        self.mailbox: List[Message] =[]
        self.messages_to_send: List[Message] =[]


    def receive_message(self, message:Message) -> None:
        '''mailer uses this function to add a data to the agent'''
        # Check if we already have a message from this sender
        if self._check_existing_message(message):
            return
        # If no matching message found, append the new one
        self.mailbox.append(message)

    def send_message(self, message:Message) -> None:
        message.recipient.receive_message(message)
    @abstractmethod
    def compute_messages(self, messages:List[Message]) -> List[Message]:
        """
        Abstract method to compute messages.
        This should be implemented by subclasses.
        """
        pass
    #TODO: keep it but make irrelevent by automatically moving old messages to history/ empty mailbox
    def _check_existing_message(self,message:Message) -> bool:
        for i, existing_msg in enumerate(self.mailbox):
            if existing_msg.sender == message.sender:
                # Replace the existing message
                self.mailbox[i] = message
                return True
        return False


class VariableAgent(BPAgent):

    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """


    def __init__(self, name: str,domain:int,computator: BPComputator):
        """
        :param name: in our case most of the times will be x1,x2,x3

        """
        node_type = "variable"
        super().__init__(name, node_type,domain,computator)

    def compute_messages(self) -> None:
        """
        Called by the BPAgent framework to compute outgoing messages.
        """
        self.messages_to_send= self.computator.compute_Q(self.mailbox)
    #TODO : make this more modular right now its only for maxsum
    @property
    def curr_belief(self) -> np.ndarray:
        """
        Compute the current belief based on incoming messages.
        :return: Current belief as a numpy array.
        """
        return np.add([message.data for message in self.mailbox],axis=0)
    @property
    def curr_assignment(self) -> int|float:
        """
        Compute the current assignment based on incoming messages.
        :return: Current assignment as a numpy array.
        """
        return np.argmax(self.curr_belief, axis=0)


    #TODO create the self belief function

    # @property
    # def belief(self) -> np.ndarray:
    #     pass
    # __repr__ = lambda self: f"VariableAgent: {self.name}"
#TODO : add the option to just add a cost table and not to create automaticall
class FactorAgent(BPAgent):
    """
    Purpose: receive and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, name: str,domain:int,computator:BPComputator,ct_creation_func:Callable,param:Dict[str,Any] ):
        node_type = "factor"
        super().__init__(name, node_type,domain,computator)
        self.cost_table :CostTable|None = None
        #TODO add the connection number on the edgeds of the graph it self
        self.connection_number : Dict[VariableAgent,int] = {}
        self.ct_creation_func = ct_creation_func
        self.ct_creation_params = param


    def compute_messages(self, messages:List[Message]) -> List[Message]:
        """
        Compute the message to be sent to the variable node.
        :param messages: List of incoming messages from variable nodes.
        :return:
        """
        return self.computator.compute_R(cost_table=self.cost_table,messages=messages)


    def initiate_cost_table(self) -> None:
        """
        Create a cost table based on the specified distribution and parameters given the domain after connections
        """
        if self.cost_table is not None:
            raise ValueError("Cost table already exists. Cannot create a new one.")
        self.cost_table = self.ct_creation_func(len(self.connection_number),self.domain,**self.ct_creation_params)
    def set_dim_for_variable(self, variable:VariableAgent, dim:int) -> None:
        """
        Add a an index to repressent a variable nodes dimension in the CT.
        :param variable: Variable node
        :param dim: dimension index
        """
        self.connection_number[variable] = dim
    def set_name_for_factor(self) -> None:
        """
        Set the name of the factor agent based on the connected variable agents.
        """
        if self.connection_number is None:
            raise ValueError("Domains not set. Cannot set name.")
        self.name = f"f{''.join(str(variable.name[1:]) for variable in self.connection_number.keys())}_"


    #TODO :fix the self naming after creating agents

    # @property
    # def name(self) -> str:
    #     if self.domains is None:
    #         return self.name
    #     return f'f{''.join(str(i) for i in self.domains.keys()[1:])}_'


    @property
    def mean_cost(self,axis = None) -> float:
        return np.mean(self.cost_table,axis=axis)
    def compute_messages(self) -> None:
        self.messages_after_compute = self.computator.compute_Q(self.messages_before_compute)
    def __repr__(self):
        return f"FactorAgent: {self.name}"
    def __str__(self):
        return f"FactorAgent: {self.name}"

