from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Set
import numpy as np
from policies.compute_message_policy import Computer


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

    def __init__(self, node_id: str, name: str, node_type: str):
        super().__init__( name, node_type)
        self.neighbors: List[str] = []  # List of connected node IDs
        self.messages: Dict[str, Any] = {}  # Stores incoming messages
        self.belief: Dict[str, float] = {}  # The belief state of the node

    @abstractmethod
    def compute_message(self, recipient_id: str) -> Any:
        pass

    @abstractmethod
    def receive_message(self, message:Message) -> None:
        pass

    @abstractmethod
    def update_local_belief(self) -> None:
        pass

    @abstractmethod
    def get_belief(self) -> Dict[str, float]:
        pass
    def __eq__ (self, other):
        return self.name == other.name and self.type != other.type
    def __hash__(self):
        return hash((self.name,self.type))


class VariableNode(BPAgent):
    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """



    def __init__(self, node_id: str, name: str, domain: List[Any]):
        super().__init__(node_id, name, "variable")
        self.domain = domain  # The domain of possible values
        self.current_value = None  # The selected value (default is unassigned)
        self.messages = Set[Message] # Stores incoming messages
        self.message_history = {}  # Stores incoming messages for debugging

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



