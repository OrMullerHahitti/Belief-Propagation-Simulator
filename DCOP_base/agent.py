from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Set
import numpy as np

from policies.Computator import Computator


class Agent(ABC):
    """
    The top-level abstract base class for any node in the DCOP problem.
    """
    def __init__(self, name: str, node_type: str):
        self.name = name  # Human-readable name for the node
        self.type = node_type  # Type of the node (e.g., 'variable', 'factor')
    def __eq__(self, other):
        return self.name == other.name
    def __hash__(self):
        return hash(self.name)
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
        from policies.Computator import Computator
        super().__init__( name, node_type) # List of connected node IDs
        self.messages: Set[Message] = set()  # Stores incoming messages
        self.history:Dict[Any:Set[Message]]={}# Stores message history
        self.iteration=0
        self.state = 0

    @property
    @abstractmethod
    def belief(self):
        '''Return the belief of the node.
        could be either a value, a distribution, or an assignment'''
        pass

    def receive_message(self, message:Message) -> None:
        self.state =1
        self.messages.add(message)

    @abstractmethod
    def update_local_beliefs(self) -> None:
        self.history[self.iteration] = self.messages
        self.messages=self._compute_messages()
        self.iteration=self.iteration+1

    @abstractmethod
    def _compute_messages(self) -> Set[Message]:
        '''Compute the messages to be sent to the neighbors'''
        pass
    def __eq__ (self, other):
        return self.name == other.name and self.type != other.type
    def __hash__(self):
        return hash((self.name,self.type))




class VariableNode(BPAgent):
    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """
    def __init__(self, node_id: str, name: str):
        super().__init__(node_id, name, node_type="variable")


    def compute_messages(self) -> Set[Message]:
        message_set = set()
        sender_to_msg = {m.sender: m.message for m in self.messages}
        total = np.sum(list(sender_to_msg.values()), axis=0)
        for recipient in sender_to_msg:
            new_msg_value = total - sender_to_msg[recipient]
            message_set.add(Message(new_msg_value, self, recipient))
        return message_set

    def receive_message(self, message:Message) -> None:
        self.messages.add(message)
    def update_local_belief(self) -> None:
        pass

    def get_belief(self) -> Dict[str, float]:
        pass





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
        self.messages=Set[Message]
    def compute_message(self, message:Message,compute:Computer=MinSumComputer()) -> Message:
        return compute.compute_message(self.cost_table,message)
    def receive_message(self,message: Message) -> None:
        self.messages.add(message)

            # A callable function that evaluates assignments
    @property
    def mean_cost(self) -> float:
        return np.mean(self.cost_table)



