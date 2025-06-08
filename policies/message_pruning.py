from typing import Dict
import numpy as np
from base_all.agents import BPAgent
from base_all.components import Message
from base_all.protocols import PolicyType
from policies.bp_policies import Policy
import logging

logger = logging.getLogger(__name__)


class MessagePruningPolicy(Policy):
    """Policy that prunes redundant messages to prevent memory explosion."""

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

    def should_accept_message(self, agent: BPAgent, new_message: Message) -> bool:
        """Decide whether to accept or prune an incoming message."""
        self.total_count += 1

        # Always accept in early iterations
        if self.iteration_count < self.min_iterations:
            return True

        # Get previous message from same sender
        prev_message = agent.mailer[new_message.sender.name]
        if prev_message is None:
            return True

        # Calculate difference
        diff_norm = np.linalg.norm(new_message.data - prev_message.data)

        # Adaptive threshold
        threshold = self.prune_threshold
        if self.adaptive_threshold:
            msg_magnitude = np.linalg.norm(new_message.data)
            threshold = self.prune_threshold * max(1.0, msg_magnitude * 0.1)

        # Prune if below threshold
        if diff_norm < threshold:
            self.pruned_count += 1
            logger.debug(
                f"Pruned message {new_message.sender.name} -> "
                f"{new_message.recipient.name}, diff: {diff_norm:.6f}"
            )
            return False

        return True

    def step_completed(self):
        """Called after each step."""
        self.iteration_count += 1

    def get_stats(self) -> Dict[str, float]:
        """Get pruning statistics."""
        return {
            "pruning_rate": self.pruned_count / max(self.total_count, 1),
            "total_messages": self.total_count,
            "pruned_messages": self.pruned_count,
            "iterations": self.iteration_count,
        }

    def reset(self):
        """Reset for new run."""
        self.iteration_count = 0
        self.pruned_count = 0
        self.total_count = 0
