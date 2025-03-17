from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Set
import numpy as np

from utils.randomes import create_random_table


class Agent(ABC):
    """
    The top-level abstract base class for any node in the DCOP problem.
    """
    def __init__(self, name: str, node_type: str):
        self.name = name  # Human-readable name for the node
        self.type = node_type  # Type of the node (e.g., 'variable', 'factor')
    def __eq__(self, other):
        return self.name == other.name and self.type != other.type
    def __hash__(self):
        return hash((self.name,self.type))
#mailer class that will handle recieveing and sending the messages to the right nodes
class Mailer(Agent):
    def __init__(self):
        super().__init__("mailer", "mailer")
        self.mailbox = {}
    def send_message(self, recipient: Agent, message: Any) -> None:
        if recipient.name in self.mailbox:
            self.mailbox[recipient.name].append(message)
        else:
            self.mailbox[recipient.name] = [message]
    def retrieve_messages(self, recipient: Agent) -> List[Any]:
        if recipient.name in self.mailbox:
            return self.mailbox[recipient.name]
        else:
            return []
    def clear_mailbox(self) -> None:
        self.mailbox = {}

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


class BPAgent(Agent):


    """
    Abstract base class for belief propagation (BP) nodes.
    Extends the Node class with methods relevant to message passing,
    updating local belief, and retrieving that belief.
    """

    def __init__(self,  name: str, node_type: str):
        super().__init__( name, node_type) # List of connected node IDs
        self.messages: Set[Message] = set()  # Stores incoming messages
        self.history:Dict[Any:Set[Message]]={}# Stores message history
        self.iteration=0
        self.state = 0



    def receive_message(self, message:Message) -> None:
        self.state =1
        self.messages.add(message)

    def update_local_beliefs(self) -> None:
        self.history[self.iteration] = self.messages
        self.messages=self._compute_messages()
        self._update_local_variables()

    @property
    @abstractmethod
    def belief(self):
        '''Return the belief of the node.
        could be either a value, a distribution, or an assignment'''
        pass

    @abstractmethod
    def _compute_messages(self) -> Set[Message]:
        '''Compute the messages to be sent to the neighbors'''
        pass
    @abstractmethod
    def _update_local_variables(self) -> None:
        '''Update the local variables of the node'''
        pass




class VariableNode(BPAgent):

    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """


    def __init__(self, node_id: str, name: str, domain_size: int = 3,
                 computator: VariableComputator = computator.MinSumVariableComputator()):
        """
        :param node_id: Unique identifier
        :param name: Human-readable name
        :param domain_size: e.g., length of the domain array
        :param computator: A policy object that implements the logic
                           for computing messages & belief (MinSum, MaxSum, etc.)
        """
        super().__init__(node_id, name, node_type="variable")
        self.domain_size = domain_size
        # A policy object controlling how messages & belief are computed:
        self.computator = computator

    def _compute_messages(self) -> Set[Message]:
        """
        Called by the BPAgent framework to compute outgoing messages.
        """
        return self.computator.compute_outgoing_messages(self, self.messages)

    @property
    def belief(self) -> np.ndarray:
        """
        Uses the policy to compute and return this node's local belief.
        """
        return self.computator.compute_belief(self, self.messages)

    def _update_local_variables(self) -> None:
        """
        For example, increment iteration count or do some local update logic.
        """
        self.iteration += 1

class FactorNode(BPAgent):
    def __init__(self, node_id: str, name: str, cost_table: np.ndarray | None = None,
                 computator: FactorComputator = computators.MinSumFactorComputator()):
        super().__init__(name, "factor")
        self.cost_table = cost_table if cost_table is not None else create_random_table(3)
        self.messages = set()
        self.computator = computator

    def _compute_messages(self) -> Set[Message]:
        # Defer to the policy class:
        return self.computator.compute_outgoing_messages(self, self.cost_table, self.messages)






class FactorNode(BPAgent):
    """
    Purpouse: recieve and send messages to the right nodes to the others, computing the beliefs to be sent
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, node_id: str, name: str, cost_table :np.ndarray|None=None):
        super().__init__(name, "factor")
      # List of variable node IDs this factor depends on
        if cost_table is not None:
            self.cost_table = cost_table
        else:
            self.cost_table = create_random_table(3)
        self.messages=Set[Message]
    def _compute_message(self, message:Message,compute:Computator) -> Message:
        return compute.compute_message(self.cost_table,message)


    @property
    def mean_cost(self) -> float:
        return np.mean(self.cost_table)



