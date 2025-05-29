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
        super().__init__(name, node_type)
        self.domain = domain
        self._history = []
        self._max_history = 10  # Limit history size to prevent memory issues
        self.mailer = MailHandler(domain)

    def receive_message(self, message: Message) -> None:
        """Receive a message and add it to the mailbox."""
        self.mailer.receive_messages(message)

    def send_message(self, message: Message) -> None:
        """Send a message to the recipient."""
        self.mailer.send()

    def empty_mailbox(self) -> None:
        """Clear the mailbox."""
        self.mailer.clear_inbox()

    def empty_outgoing(self):
        """Clear the outbox."""
        self.mailer.clear_outgoing()

    @property
    def inbox(self):
        return self.mailer.inbox

    @property
    def outbox(self):
        return self.mailer.outbox

    @abstractmethod
    def compute_messages(self) -> List[Message]:
        """Abstract method to compute messages."""
        pass

    @property
    def last_iteration(self) -> List[Message]:
        """Get the last iteration messages."""
        if not self._history:
            return []
        return self._history[-1]

    def last_cycle(self, diameter: int = 1) -> List[Message]:
        """Get the last cycle messages."""
        if not self._history:
            return []
        return self._history[-diameter]

    def append_last_iteration(self):
        """Append current messages to history with size limit."""
        self._history.append([msg.copy() for msg in self.mailer.outbox])
        if len(self._history) > self._max_history:
            self._history.pop(0)  # Remove oldest to maintain size limit


class VariableAgent(BPAgent):
    """
    Represents a variable node in DCOP, holding a variable and its domain.
    """

    def __init__(self, name: str, domain: int):
        node_type = "variable"
        super().__init__(name, node_type, domain)

    def compute_messages(self) -> None:
        """Called by the BPAgent framework to compute outgoing messages."""
        if self.computator and self.mailer.inbox:
            messages = self.computator.compute_Q(self.mailer.inbox)
            self.mailer.stage_sending(messages)

    @property
    def belief(self) -> np.ndarray:
        """Compute the current belief based on incoming messages."""
        if not self.inbox:
            return np.ones(self.domain) / self.domain  # Uniform belief

        # Sum all incoming messages
        belief = np.zeros(self.domain)
        for message in self.inbox:
            belief += message.data

        # Normalize to avoid numerical issues
        belief_min = np.min(belief)
        if belief_min < 0:
            belief -= belief_min

        return belief

    # TODO: make it argmin or argmax based on the problem type
    @property
    def curr_assignment(self) -> int | float:
        """Compute the current assignment based on beliefs."""
        return int(np.argmin(self.belief))

    def __str__(self):
        return self.name.upper()

    def __repr__(self):
        return f"VariableAgent({self.name}, domain={self.domain})"


class FactorAgent(BPAgent):
    """
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

        self.cost_table = None if cost_table is None else cost_table.copy()
        self.connection_number: Dict[str, int] = {}  # var_name -> dimension
        self.ct_creation_func = ct_creation_func
        self.ct_creation_params = param if param is not None else {}
        self._original: np.ndarray | None = None

    @classmethod
    def create_from_cost_table(cls, name: str, cost_table: CostTable):
        """Create a factor agent from an existing cost table."""
        return cls(
            name=name,
            domain=cost_table.shape[0],
            ct_creation_func=lambda *args, **kwargs: cost_table,
            param=None,
            cost_table=cost_table,
        )

    def compute_messages(self) -> List[Message]:
        """Compute messages to be sent to variable nodes."""
        if self.computator and self.cost_table is not None and self.inbox:
            messages = self.computator.compute_R(
                cost_table=self.cost_table, incoming_messages=self.inbox
            )
            self.mailer.stage_sending(messages)

    def initiate_cost_table(self) -> None:
        """Create a cost table based on the specified distribution."""
        if self.cost_table is not None:
            raise ValueError("Cost table already exists. Cannot create a new one.")

        if not self.connection_number:
            raise ValueError("No connections set. Cannot create cost table.")

        # Create cost table with correct dimensions
        num_vars = len(self.connection_number)
        self.cost_table = self.ct_creation_func(
            num_vars, self.domain, **self.ct_creation_params
        )

    def set_dim_for_variable(self, variable: VariableAgent, dim: int) -> None:
        """Add an index to represent a variable node's dimension in the CT."""
        self.connection_number[variable.name] = dim

    def set_name_for_factor(self) -> None:
        """Set the name of the factor agent based on connected variables."""
        if not self.connection_number:
            raise ValueError("No connections set. Cannot set name.")

        var_indices = []
        for var_name in sorted(self.connection_number.keys()):
            if var_name.startswith("x"):
                var_indices.append(var_name[1:])

        self.name = f"f{''.join(var_indices)}_"

    def save_original(self, ct: CostTable | None = None) -> None:
        """Save the original cost table before modifications."""
        if self._original is None and self.cost_table is not None and ct is None:
            self._original = np.copy(self.cost_table)
        elif ct is not None and self._original is None and self.cost_table is not None:
            self._original = np.copy(ct)

    @property
    def mean_cost(self) -> float:
        """Calculate mean cost of the cost table."""
        if self.cost_table is None:
            return 0.0
        return float(np.mean(self.cost_table))

    @property
    def total_cost(self) -> float:
        """Calculate total cost of the cost table."""
        if self.cost_table is None:
            return 0.0
        return float(np.sum(self.cost_table))

    @property
    def original_cost_table(self) -> np.ndarray | None:
        """Get the original cost table if saved."""
        return self._original

    def __repr__(self):
        return f"FactorAgent({self.name}, connections={list(self.connection_number.keys())})"

    def __str__(self):
        return self.name.upper()
