import numpy as np
import logging
from typing import List

from bp_base.DCOP_base import Computator, Agent
from bp_base.typing import CostTable
from bp_base.components import Message

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BPComputator(Computator):
    """
    Generic class for message computation in belief propagation.
    Can be configured for different BP variants (min-sum, max-sum, etc.)
    """

    def __init__(self, reduce_func, combine_func):
        """
        Initialize the computator with the appropriate operations.
        """
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        logger.info(
            f"Initialized Computator with reduce_func={reduce_func.__name__}, "
            f"combine_func={combine_func.__name__}"
        )

    def _get_node_dimension(self, factor, node: Agent) -> int:
        """
        Safely get dimension index for a node in factor's connection_number.
        Handles both string keys and object keys for backward compatibility.
        """
        # Try direct name lookup first (preferred)
        if node.name in factor.connection_number:
            return factor.connection_number[node.name]

        # Handle legacy case where objects are used as keys
        for key, dim in factor.connection_number.items():
            if isinstance(key, Agent) and key.name == node.name:
                return dim
            elif isinstance(key, str) and key == node.name:
                return dim

        # If not found, provide helpful error
        available_keys = list(factor.connection_number.keys())
        raise KeyError(
            f"Node '{node.name}' not found in factor '{factor.name}' connections. "
            f"Available connections: {available_keys}"
        )

    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Compute variable->factor messages (Q messages).

        For each outgoing message to a factor f, the variable combines all
        incoming messages from factors EXCEPT f.
        """
        logger.debug(f"Computing Q messages with {len(messages)} incoming messages")

        if not messages:
            logger.warning("No incoming messages, returning empty list")
            return []

        # All messages have the same recipient (the variable node)
        variable = messages[0].recipient
        outgoing_messages = []

        # For each incoming message, create an outgoing message back to sender
        for i, msg_i in enumerate(messages):
            factor = msg_i.sender

            # Combine all messages except the one from this factor
            other_messages = [msg_j.data for j, msg_j in enumerate(messages) if j != i]

            if other_messages:
                # Start with copy to preserve dimensions
                combined_data = other_messages[0].copy()
                for msg_data in other_messages[1:]:
                    combined_data = self.combine_func(combined_data, msg_data)
            else:
                # If no other messages, send uniform/zero message
                combined_data = np.zeros_like(msg_i.data)

            # Create outgoing message
            outgoing_message = Message(
                data=combined_data,
                sender=variable,
                recipient=factor
            )
            outgoing_messages.append(outgoing_message)

        logger.debug(f"Computed {len(outgoing_messages)} outgoing Q messages")
        return outgoing_messages

    def compute_R(self, cost_table: CostTable, incoming_messages: List[Message]) -> List[Message]:
        """
        Compute factor->variable messages (R messages).

        The factor combines its cost table with messages from all variables
        except the recipient, then marginalizes.
        """
        logger.debug(
            f"Computing R messages with {len(incoming_messages)} incoming messages "
            f"and cost table shape {cost_table.shape if hasattr(cost_table, 'shape') else 'unknown'}"
        )

        if not incoming_messages:
            logger.warning("No incoming messages, returning empty list")
            return []

        factor = incoming_messages[0].recipient

        # Handle test cases that might not have connection_number
        if not hasattr(factor, 'connection_number') or not factor.connection_number:
            # For tests, create connection_number based on message order
            factor.connection_number = {}
            for i, msg in enumerate(incoming_messages):
                factor.connection_number[msg.sender.name] = i

        outgoing_messages = []

        # For each variable, compute outgoing message
        for i, msg_i in enumerate(incoming_messages):
            variable_node = msg_i.sender

            try:
                dim = self._get_node_dimension(factor, variable_node)
            except KeyError as e:
                logger.error(f"Failed to find dimension: {e}")
                raise

            # Start with copy of cost table
            augmented_costs = cost_table.copy()

            # Add messages from all OTHER variables
            for j, msg_j in enumerate(incoming_messages):
                if j != i:  # Skip the recipient variable
                    sender = msg_j.sender
                    sender_dim = self._get_node_dimension(factor, sender)

                    # Create broadcasting shape
                    broadcast_shape = [1] * len(cost_table.shape)
                    broadcast_shape[sender_dim] = len(msg_j.data)

                    # Reshape message for broadcasting
                    reshaped_msg = msg_j.data.reshape(broadcast_shape)

                    # Combine with cost table
                    augmented_costs = self.combine_func(augmented_costs, reshaped_msg)

            # Marginalize over all dimensions except the recipient's
            axes_to_reduce = tuple(j for j in range(len(cost_table.shape)) if j != dim)
            reduced_msg = self.reduce_func(augmented_costs, axis=axes_to_reduce)

            # Create outgoing message
            outgoing_message = Message(
                data=reduced_msg,
                sender=factor,
                recipient=variable_node
            )
            outgoing_messages.append(outgoing_message)

        logger.debug(f"Computed {len(outgoing_messages)} outgoing R messages")
        return outgoing_messages


class MinSumComputator(BPComputator):
    """
    Min-sum algorithm for belief propagation.
    Used to find the assignment that minimizes the sum of costs.
    """

    def __init__(self):
        super().__init__(reduce_func=np.min, combine_func=np.add)
        logger.info("Initialized MinSumComputator")


class MaxSumComputator(BPComputator):
    """
    Max-sum algorithm for belief propagation.
    Used to find the assignment that maximizes the sum of utilities.
    """

    def __init__(self):
        super().__init__(reduce_func=np.max, combine_func=np.add)
        logger.info("Initialized MaxSumComputator")