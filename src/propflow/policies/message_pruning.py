"""A Policy for Pruning Redundant Messages in Belief Propagation.

This module provides a policy that can be used to reduce the number of
messages that are processed and stored in a belief propagation simulation.
It works by comparing a new incoming message to the previous message from the
same sender and discarding the new one if the change is below a certain
threshold. This can significantly reduce memory usage and computation time in
simulations where messages quickly stabilize.
"""
from typing import Dict
import numpy as np
from ..core.agents import FGAgent
from ..core.components import Message
from ..core.protocols import PolicyType
from ..configs.global_config_mapping import POLICY_DEFAULTS
from .bp_policies import Policy
import logging

logger = logging.getLogger(__name__)


class MessagePruningPolicy(Policy):
    """A policy that prunes messages that have not changed significantly.

    This policy helps to optimize belief propagation by avoiding the processing
    of messages that are redundant. It compares the norm of the difference
    between a new message and the previous message from the same sender.

    Attributes:
        prune_threshold (float): The threshold for pruning.
        min_iterations (int): The number of initial iterations during which
            no pruning will occur.
        adaptive_threshold (bool): If True, the threshold is scaled by the
            magnitude of the message.
        iteration_count (int): The current iteration number.
        pruned_count (int): The total number of pruned messages.
        total_count (int): The total number of messages considered for pruning.
    """

    def __init__(
        self,
        prune_threshold: float = None,
        min_iterations: int = 5,
        adaptive_threshold: bool = True,
    ):
        """Initializes the MessagePruningPolicy.

        Args:
            prune_threshold: The base threshold for pruning. If the change
                is less than this, the message is pruned. Defaults to the
                value in `POLICY_DEFAULTS`.
            min_iterations: The number of iterations to run before pruning
                begins. Defaults to 5.
            adaptive_threshold: Whether to use an adaptive threshold that
                scales with the message magnitude. Defaults to True.
        """
        super().__init__(PolicyType.MESSAGE)
        self.prune_threshold = (
            prune_threshold
            if prune_threshold is not None
            else POLICY_DEFAULTS["pruning_threshold"]
        )
        self.min_iterations = min_iterations
        self.adaptive_threshold = adaptive_threshold
        self.iteration_count = 0
        self.pruned_count = 0
        self.total_count = 0

    def should_accept_message(self, agent: FGAgent, new_message: Message) -> bool:
        """Determines whether to accept or prune an incoming message.

        Args:
            agent: The agent receiving the message.
            new_message: The new `Message` object.

        Returns:
            True if the message should be accepted, False if it should be pruned.
        """
        self.total_count += 1

        if self.iteration_count < self.min_iterations:
            return True

        prev_message = agent.mailer[new_message.sender.name]
        if prev_message is None:
            return True

        diff_norm = np.linalg.norm(new_message.data - prev_message.data)

        threshold = self.prune_threshold
        if self.adaptive_threshold:
            msg_magnitude = np.linalg.norm(new_message.data)
            threshold *= max(1.0, msg_magnitude * POLICY_DEFAULTS["pruning_magnitude_factor"])

        if diff_norm < threshold:
            self.pruned_count += 1
            logger.debug(
                f"Pruned message {new_message.sender.name} -> "
                f"{new_message.recipient.name}, diff: {diff_norm:.6f}"
            )
            return False

        return True

    def step_completed(self) -> None:
        """Signals that a simulation step has completed, incrementing the iteration count."""
        self.iteration_count += 1

    def get_stats(self) -> Dict[str, float]:
        """Returns statistics about the pruning process.

        Returns:
            A dictionary containing the pruning rate, total messages considered,
            number of pruned messages, and total iterations.
        """
        return {
            "pruning_rate": self.pruned_count / max(self.total_count, 1),
            "total_messages": self.total_count,
            "pruned_messages": self.pruned_count,
            "iterations": self.iteration_count,
        }

    def reset(self) -> None:
        """Resets the policy's internal state for a new simulation run."""
        self.iteration_count = 0
        self.pruned_count = 0
        self.total_count = 0
