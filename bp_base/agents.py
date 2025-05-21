from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Any, Callable
import numpy as np

from bp_base.components import Message, CostTable, MailHandler
from bp_base.DCOP_base import Agent


class BPAgent(Agent, ABC):
    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to data passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self, name: str, node_type: str, domain: int):
        ### --- attributes --- ###
        super().__init__(name, node_type)
        self.domain = domain
        self._history = []
        ### --- message handling --- ###
        self.mailer = MailHandler(domain)

    def receive_message(self, message: Message) -> None:
        """
        Receive a message and add it to the mailbox.
        :param message: Message to be received.
        """
        self.mailer.receive_messages(message)

    def send_message(self, message: Message) -> None:
        """
        Send a message to the recipient.
        :param message: Message to be sent.
        """
        self.mailer.send()

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

    def __init__(self, name: str, domain: int):
        """
        :param name: in our case most of the times will be x1,x2,x3

        """
        node_type = "variable"

        super().__init__(name, node_type, domain)

    def compute_messages(self) -> None:
        """
        Called by the BPAgent framework to compute outgoing messages.
        """
        self.mailer.stage_sending(self.computator.compute_Q(self.mailer.inbox))

    # TODO : make this more modular right now its only for sum- kind
    @property
    def belief(self) -> np.ndarray:
        """
        Compute the current belief based on incoming messages.
        :return: Current belief as a numpy array.
        """
        return np.sum([message.data for message in self.inbox], axis=0)

    @property
    def curr_assignment(self) -> int | float:
        """
        Compute the current assignment based on incoming messages.
        :return: Current assignment as a numpy array.
        """
        return np.argmax(self.belief, axis=0)

    __str__ = lambda self: self.name.upper()

    @property
    def last_iteration(self) -> List[Message]:
        """
        Get the last iteration messages.
        :return: List of last iteration messages.
        """
        if not self._history:
            return []
        return self._history[-1]

    def append_last_iteration(self):
        self._history.append([msg.copy() for msg in self.mailer.outbox])


### ---- Factor Agent --- ###


# TODO : add the option to just add a cost table and not to create automaticall
class FactorAgent(BPAgent):
    """
    Purpose: receive and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(
        self,
        name: str,
        domain: int,
        ct_creation_func: Callable,
        param: Dict[str, Any] | None = None,
        cost_table: CostTable | None = None,
    ):
        node_type = "factor"
        super().__init__(name, node_type, domain)

        self.cost_table: None = None if cost_table is None else cost_table.copy()
        self.connection_number: Dict[str, int] = (
            {}
        )  # Store variable names instead of objects to save memory
        self.ct_creation_func = ct_creation_func
        self.ct_creation_params = param

        self._original: np.ndarray | None = (
            None  # in case of a policy changes original cost table this is meant to save it
        )

    @classmethod
    def create_from_cost_table(cls, name: str, cost_table: CostTable):
        return cls(
            name=name,
            domain=cost_table.shape[0],
            ct_creation_func=lambda *args, **kwargs: cost_table,
            param=None,
            cost_table=cost_table,
        )

    def compute_messages(self) -> List[Message]:
        """
        Compute the message to be sent to the variable node.
        :param messages: List of incoming messages from variable nodes.
        :return:
        """
        return self.mailer.stage_sending(
            self.computator.compute_R(
                cost_table=self.cost_table, incoming_messages=self.inbox
            )
        )

    def initiate_cost_table(self) -> None:
        """
        Create a cost table based on the specified distribution and parameters given the domain after connections
        """
        if self.cost_table is not None:
            raise ValueError("Cost table already exists. Cannot create a new one.")
        self.cost_table = self.ct_creation_func(
            len(self.connection_number), self.domain, **self.ct_creation_params
        )

    def set_dim_for_variable(self, variable: VariableAgent, dim: int) -> None:
        """
        Add an index to represent a variable node's dimension in the CT.
        :param variable: Variable node
        :param dim: dimension index
        """
        self.connection_number[variable.name] = dim

    def set_name_for_factor(self) -> None:
        """
        Set the name of the factor agent based on the connected variable names.
        """
        if self.connection_number is None:
            raise ValueError("Domains not set. Cannot set name.")
        self.name = f"f{''.join(str(var_name[1:]) for var_name in self.connection_number.keys())}_"

    def save_original(self):
        """
        Save the original cost table.
        """
        if self._original is None:
            self._original = np.copy(self.cost_table)
        else:
            return

    @property
    def mean_cost(self, axis=None) -> float:
        return np.mean(self.cost_table, axis=axis)

    @property
    def total_cost(self, axis=None) -> float:
        return np.sum(self.cost_table, axis=axis)

    @property
    def original_cost_table(self) -> np.ndarray | None:
        return self._original

    def __repr__(self):
        return f"FactorAgent: {self.name}"

    __str__ = lambda self: self.name.upper()
