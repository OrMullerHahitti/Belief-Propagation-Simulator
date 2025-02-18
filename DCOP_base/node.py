from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable
import numpy as np
class Node(ABC):
    """
    The top-level abstract base class for any node in the DCOP problem.
    """
    def __init__(self, name: str, node_type: str):
        self.name = name  # Human-readable name for the node
        self.type = node_type  # Type of the node (e.g., 'variable', 'factor')
    def __eq__(self, other):
        return self.name == other.name


class BPNode(Node):
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
    def receive_message(self, sender_id: str, message: Any) -> None:
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


class VariableNode(BPNode):
    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """



    def __init__(self, node_id: str, name: str, domain: List[Any]):
        super().__init__(node_id, name, "variable")
        self.domain = domain  # The domain of possible values
        self.current_value = None  # The selected value (default is unassigned)
        self.messages = {}  # Stores incoming messages
        self.message_history = {}  # Stores incoming messages for debugging
    def compute_message(self, recipient_id: str) -> Any:
        pass

    def receive_message(self, sender_id: str, message: Any) -> None:
        pass

    def update_local_belief(self) -> None:
        pass

    def get_belief(self) -> Dict[str, float]:
        pass




class FactorNode(BPNode):
    """
    Represents a factor node, storing a function that links multiple variables.
    """

    def __init__(self, node_id: str, name: str, scope: List['Node'], cost_table :np.ndarray|None=None):
        super().__init__(name, "factor")
        self.scope = scope  # List of variable node IDs this factor depends on
        if cost_table is not None:
            self.cost_table = cost_table
        else:
            self.cost_table = np.random.rand(2 ** len(scope)).reshape((2,) * len(scope))

            # A callable function that evaluates assignments
    @property
    def mean_cost(self) -> float:
        return np.mean(self.cost_table)



