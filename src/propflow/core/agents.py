from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Callable, Any, Dict

import numpy as np

from .components import Message, CostTable, MailHandler
from .dcop_base import Agent


class FGAgent(Agent, ABC):
    """Abstract base class for belief propagation (BP) nodes.

    Extends the `Agent` class with methods relevant to message passing,
    updating local belief, and retrieving that belief. It serves as a
    foundation for both `VariableAgent` and `FactorAgent` classes.

    Attributes:
        domain (int): The size of the variable domain.
        mailer (MailHandler): Handles incoming and outgoing messages.
    """

    def __init__(self, name: str, node_type: str, domain: int):
        """Initializes an FGAgent.

        Args:
            name (str): The name of the agent.
            node_type (str): The type of the node (e.g., 'variable', 'factor').
            domain (int): The size of the variable domain.
        """
        super().__init__(name, node_type)
        self.domain = domain
        self._history = []
        self._max_history = 10  # Limit history size to prevent memory issues
        self.mailer = MailHandler(domain)

    def receive_message(self, message: Message) -> None:
        """Receives a message and adds it to the mailer's inbox.

        Args:
            message (Message): The message to be received.
        """
        self.mailer.receive_messages(message)

    def send_message(self, message: Message) -> None:
        """Sends a message to its recipient via the mailer.

        Args:
            message (Message): The message to be sent.
        """
        self.mailer.send()

    def empty_mailbox(self) -> None:
        """Clears all messages from the mailer's inbox."""
        self.mailer.clear_inbox()

    def empty_outgoing(self):
        """Clears all messages from the mailer's outbox."""
        self.mailer.clear_outgoing()

    @property
    def inbox(self) -> List[Message]:
        """list[Message]: A list of incoming messages."""
        return self.mailer.inbox

    @property
    def outbox(self) -> List[Message]:
        """list[Message]: A list of outgoing messages."""
        return self.mailer.outbox

    @abstractmethod
    def compute_messages(self) -> List[Message]:
        """Abstract method to compute outgoing messages.

        This method must be implemented by subclasses to define how
        messages are calculated based on the agent's current state
        and incoming messages.

        Returns:
            A list of messages to be sent.
        """
        pass

    @property
    def last_iteration(self) -> List[Message]:
        """list[Message]: The last list of messages sent."""
        if not self._history:
            return []
        return self._history[-1]

    def last_cycle(self, diameter: int = 1) -> List[Message]:
        """Retrieves messages from a previous cycle.

        Args:
            diameter (int): The number of iterations in a cycle. Defaults to 1.

        Returns:
            A list of messages from the specified previous cycle.
        """
        if not self._history:
            return []
        return self._history[-diameter]

    def append_last_iteration(self):
        """Appends the current outbox to the history.

        Maintains a history of sent messages, limited by `_max_history`.
        """
        self._history.append([msg.copy() for msg in self.mailer.outbox])
        if len(self._history) > self._max_history:
            self._history.pop(0)  # Remove oldest to maintain size limit


class VariableAgent(FGAgent):
    """Represents a variable node in a factor graph.

    This agent is responsible for aggregating messages from neighboring
    factor nodes to compute its belief over its domain.

    Attributes:
        computator: An object that handles the computation of messages and beliefs.
    """

    def __init__(self, name: str, domain: int):
        """Initializes a VariableAgent.

        Args:
            name (str): The name of the variable (e.g., 'x1').
            domain (int): The size of the variable's domain.
        """
        node_type = "variable"
        super().__init__(name, node_type, domain)

    def compute_messages(self) -> None:
        """Computes outgoing messages to factor nodes.

        Uses the assigned `computator` to calculate messages based on
        the contents of the inbox.
        """
        if self.computator and self.mailer.inbox:
            messages = self.computator.compute_Q(self.mailer.inbox)
            self.mailer.stage_sending(messages)

    @property
    def belief(self) -> np.ndarray:
        """np.ndarray: The current belief distribution over the variable's domain."""
        if self.computator:
            return self.computator.compute_belief(self.inbox, self.domain)
        else:
            # Fallback to sum-product behavior if no computator
            if not self.inbox:
                return np.ones(self.domain) / self.domain  # Uniform belief

            # Sum all incoming messages
            belief = np.zeros(self.domain)
            for message in self.inbox:
                belief += message.data

            return belief

    @property
    def curr_assignment(self) -> int | float:
        """int | float: The current assignment for the variable."""
        if self.computator:
            return self.computator.get_assignment(self.belief)
        else:
            # Fallback to default MinSum behavior if no computator
            return int(np.argmin(self.belief))

    def __str__(self) -> str:
        """Returns the uppercase name of the agent."""
        return self.name.upper()

    def __repr__(self) -> str:
        """Returns a string representation of the VariableAgent."""
        return f"VariableAgent({self.name}, domain={self.domain})"


class FactorAgent(FGAgent):
    """Represents a factor node in a factor graph.

    This agent stores a cost function (or utility function) that defines
    the relationship between a set of connected variable nodes.

    Attributes:
        cost_table (CostTable): The table of costs for each combination of assignments.
        connection_number (dict): A mapping from variable names to their dimension index.
        ct_creation_func (Callable): A function to create the cost table.
        ct_creation_params (dict): Parameters for the cost table creation function.
    """

    def __init__(
        self,
        name: str,
        domain: int,
        ct_creation_func: Callable,
        param: Dict[str, Any] | None = None,
        cost_table: CostTable | None = None,
    ):
        """Initializes a FactorAgent.

        Args:
            name (str): The name of the factor (e.g., 'f12').
            domain (int): The size of the variable domain.
            ct_creation_func (Callable): A function to generate the cost table.
            param (dict, optional): Parameters for `ct_creation_func`. Defaults to None.
            cost_table (CostTable, optional): An existing cost table. Defaults to None.
        """
        node_type = "factor"
        super().__init__(name, node_type, domain)

        self.cost_table = None if cost_table is None else cost_table.copy()
        self.connection_number: Dict[str, int] = {}  # var_name -> dimension
        self.ct_creation_func = ct_creation_func
        self.ct_creation_params = param if param is not None else {}
        self._original: np.ndarray | None = None

    @classmethod
    def create_from_cost_table(cls, name: str, cost_table: CostTable) -> FactorAgent:
        """Creates a FactorAgent from an existing cost table.

        Args:
            name (str): The name of the factor.
            cost_table (CostTable): The cost table to use.

        Returns:
            A new `FactorAgent` instance.
        """
        return cls(
            name=name,
            domain=cost_table.shape[0],
            ct_creation_func=lambda *args, **kwargs: cost_table,
            param=None,
            cost_table=cost_table,
        )

    def compute_messages(self) -> None:
        """Computes messages to be sent to variable nodes.

        Uses the assigned `computator` to calculate messages based on the
        cost table and incoming messages from variable nodes.
        """
        if self.computator and self.cost_table is not None and self.inbox:
            messages = self.computator.compute_R(
                cost_table=self.cost_table, incoming_messages=self.inbox
            )
            self.mailer.stage_sending(messages)

    def initiate_cost_table(self) -> None:
        """Creates the cost table using the provided creation function.

        Raises:
            ValueError: If the cost table already exists or if no connections are set.
        """
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
        """Maps a variable to a dimension in the cost table.

        Args:
            variable (VariableAgent): The variable agent to map.
            dim (int): The dimension index in the cost table.
        """
        self.connection_number[variable.name] = dim

    def set_name_for_factor(self) -> None:
        """Sets the factor's name based on its connected variables.

        Raises:
            ValueError: If no connections are set.
        """
        if not self.connection_number:
            raise ValueError("No connections set. Cannot set name.")

        var_indices = []
        for var_name in sorted(self.connection_number.keys()):
            if var_name.startswith("x"):
                var_indices.append(var_name[1:])

        self.name = f"f{''.join(var_indices)}_"

    def save_original(self, ct: CostTable | None = None) -> None:
        """Saves a copy of the original cost table.

        Args:
            ct (CostTable, optional): An external cost table to save. Defaults to None.
        """
        if self._original is None and self.cost_table is not None and ct is None:
            self._original = np.copy(self.cost_table)
        elif ct is not None and self._original is None and self.cost_table is not None:
            self._original = np.copy(ct)

    @property
    def mean_cost(self) -> float:
        """float: The mean value of the costs in the cost table."""
        if self.cost_table is None:
            return 0.0
        return float(np.mean(self.cost_table))

    @property
    def total_cost(self) -> float:
        """float: The sum of all costs in the cost table."""
        if self.cost_table is None:
            return 0.0
        return float(np.sum(self.cost_table))

    @property
    def original_cost_table(self) -> np.ndarray | None:
        """np.ndarray | None: The original, unmodified cost table, if saved."""
        return self._original

    def __repr__(self) -> str:
        """Returns a string representation of the FactorAgent."""
        return f"FactorAgent({self.name}, connections={list(self.connection_number.keys())})"

    def __str__(self) -> str:
        """Returns the uppercase name of the agent."""
        return self.name.upper()
