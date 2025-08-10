from typing import Dict, List
import numpy as np
from ..core.agents import VariableAgent, FactorAgent
from ..core.components import Message
from ..core.protocols import PolicyType, CostTable
from .bp_policies import Policy


def init_normalization(li: List[FactorAgent]):
    x = len(li)
    for factor in li:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table / x


def normalize_soft_max(cost_table: np.ndarray) -> CostTable:
    """
    Normalize the cost table using softmax to ensure all values are positive and sum to 1.

    Args:
        cost_table: n-dimensional cost table as numpy array

    Returns:
        Normalized cost table
    """
    exp_cost_table = np.exp(cost_table - np.max(cost_table))  # Stability improvement
    return exp_cost_table / np.sum(exp_cost_table)


def normalize_cost_table_sum(cost_table: np.ndarray) -> CostTable:
    """
    Normalize the cost table so that the sum of all dimensions is equal.

    Args:
        cost_table: n-dimensional cost table as numpy array

    Returns:
        Normalized cost table
    """
    total_sum = np.sum(cost_table)
    shape = cost_table.shape
    for dim in range(len(shape)):
        curr_sum = np.sum(cost_table, axis=dim)
        cost_table = cost_table / (curr_sum * total_sum)
    return cost_table


def normalize_inbox(variables: List[VariableAgent]):
    """
    Normalize the message data of all variables in the factor graph after each cycle.
    """
    for var in variables:
        if var.last_iteration is not None:
            # Normalize the last iteration messages
            for message in var.last_iteration:
                if message.data is not None:
                    message.data = message.data - message.data.min()
        for message in var.mailer.inbox:
            if message.data is not None:
                # Normalize the message data
                message.data = message.data - message.data.min()


class MessagePruningPolicy(Policy):
    """
    Policy that prunes redundant messages based on L2-norm threshold.
    Prevents message explosion by filtering out similar consecutive messages.
    """

    def __init__(
        self,
        prune_threshold: float = 1e-4,
        min_iterations: int = 5,
        adaptive_threshold: bool = True,
    ):
        super().__init__(PolicyType.MESSAGE)
        self.prune_threshold = prune_threshold
        self.min_iterations = min_iterations
        self.adaptive_threshold = adaptive_threshold
        self.iteration_count = 0
        self.pruned_count = 0
        self.total_count = 0

    def __call__(self, agent, new_message: Message) -> bool:
        """
        Decide whether to accept or prune an incoming message.

        Args:
            agent: The receiving agent (Variable or Factor)
            new_message: The incoming message to evaluate

        Returns:
            bool: True if message should be kept, False if pruned
        """
        self.total_count += 1

        # Always accept messages in first few iterations
        if self.iteration_count < self.min_iterations:
            return True

        # Get previous message from same sender
        prev_message = agent.mailer[new_message.sender.name]
        if prev_message is None:
            return True

        # Calculate L2 norm of difference
        diff_norm = np.linalg.norm(new_message.data - prev_message.data)

        # Adaptive threshold based on message magnitude
        threshold = self.prune_threshold
        if self.adaptive_threshold:
            msg_magnitude = np.linalg.norm(new_message.data)
            threshold = self.prune_threshold * max(1.0, msg_magnitude * 0.1)

        # Prune if difference is below threshold
        if diff_norm < threshold:
            self.pruned_count += 1
            return False

        return True

    def step_completed(self):
        """Called after each BP step to update internal state."""
        self.iteration_count += 1

    def get_pruning_stats(self) -> Dict[str, float]:
        """Get statistics about message pruning performance."""
        if self.total_count == 0:
            return {"pruning_rate": 0.0, "total_messages": 0, "pruned_messages": 0}

        return {
            "pruning_rate": self.pruned_count / self.total_count,
            "total_messages": self.total_count,
            "pruned_messages": self.pruned_count,
            "iterations": self.iteration_count,
        }

    def reset(self):
        """Reset statistics for new run."""
        self.iteration_count = 0
        self.pruned_count = 0
        self.total_count = 0


# Integration with MailHandler
class PruningMailHandler:
    """Extended MailHandler with message pruning capability."""

    def __init__(self, domain_size: int, pruning_policy: MessagePruningPolicy = None):
        # Copy existing MailHandler initialization
        self._message_domain_size = domain_size
        self._incoming: Dict[str, Message] = {}
        self._outgoing: List[Message] = []
        self.pruning_policy = pruning_policy

    def receive_messages_with_pruning(self, owner, message: Message):
        """Receive message with optional pruning."""
        if self.pruning_policy is None:
            # Fallback to original behavior
            key = self._make_key(message.sender)
            self._incoming[key] = message
            return

        # Apply pruning policy
        should_keep = self.pruning_policy(owner, message)
        if should_keep:
            key = self._make_key(message.sender)
            self._incoming[key] = message

    def _make_key(self, agent) -> str:
        """Create unique key for agent to handle identity issues."""
        return f"{agent.name}_{agent.type}"
