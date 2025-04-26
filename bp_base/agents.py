from __future__ import annotations

from abc import ABC,abstractmethod
from dataclasses import field
from typing import Dict, List, TypeAlias, Any,Callable
import numpy as np
from pyexpat.errors import messages

from bp_base.components import Message, CostTable, MailHandler
from bp_base.computators import BPComputator
from DCOP_base import Agent
from utils.randomes import create_random_table

from configs.hyper_parameters_config import MESSAGE_DOMAIN_SIZE, CT_CREATION_FUNCTION, CT_CREATION_PARAMS,COMPUTATOR

class BPAgent(Agent,ABC):

    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to data passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self,  name: str, node_type: str,domain:int ):
        ### --- attributes --- ###
        super().__init__( name, node_type)
        self.domain = domain
        ### --- message handling --- ###
        self.mailer= MailHandler(domain)
    def receive_message(self, message: Message) -> None:
        """
        Receive a message and add it to the mailbox.
        :param message: Message to be received.
        """
        self.mailer.receive_messages(message)
    def empty_mailbox(self) -> None:
        """
        Clear the mailbox.
        """
        self.mailer.clear_inbox()
    def empty_outgoing(self):
        """
        Clear the outbox.
        """
        self.mailer.clear_outgoing()
    @property
    def inbox(self):
        return self.mailer.inbox
    @abstractmethod
    def compute_messages(self) -> List[Message]:
        """
        Abstract method to compute messages.
        This should be implemented by subclasses.
        """
        pass



##### ----- Variable Agent ----- #####


class VariableAgent(BPAgent):

    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """


    def __init__(self, name: str,domain:int):
        """
        :param name: in our case most of the times will be x1,x2,x3

        """
        node_type = "variable"
        super().__init__(name, node_type,domain)

    def compute_messages(self) -> None:
        """
        Called by the BPAgent framework to compute outgoing messages.
        """
        self.mailbox.stage = self.computator.compute_Q(self.mailbox.inbox)

    #TODO : make this more modular right now its only for maxsum
    @property
    def curr_belief(self) -> np.ndarray:
        """
        Compute the current belief based on incoming messages.
        :return: Current belief as a numpy array.
        """
        return np.sum([message.data for message in self.mailbox], axis=0)
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



### ---- Factor Agent --- ###


#TODO : add the option to just add a cost table and not to create automaticall
class FactorAgent(BPAgent):
    """
    Purpose: receive and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, name: str,domain:int,ct_creation_func:Callable,param:Dict[str,Any] ):
        node_type = "factor"
        super().__init__(name, node_type,domain)
        self.cost_table :CostTable|None = None
        #TODO add the connection number on the edgeds of the graph it self
        self.connection_number : Dict[VariableAgent,int] = {}
        self.ct_creation_func = ct_creation_func
        self.ct_creation_params = param


    def compute_messages(self) -> List[Message]:
        """
        Compute the message to be sent to the variable node.
        :param messages: List of incoming messages from variable nodes.
        :return:
        """
        return self.computator.compute_R(cost_table=self.cost_table,incoming_messages=self.mailbox)


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
    def __repr__(self):
        return f"FactorAgent: {self.name}"
    def __str__(self):
        return f"FactorAgent: {self.name}"

