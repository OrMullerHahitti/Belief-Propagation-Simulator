import numpy as np
from typing import List, Dict, Tuple
import logging
from functools import lru_cache

from bp_base.DCOP_base import Agent
from bp_base.components import Message

# Minimal logging for computators
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class BPComputator:
    """
    Vectorized, cache-friendly version of the original BPComputator.
    """

    def __init__(self, reduce_func, combine_func):
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        # Cache frequently used operations
        self._broadcast_cache = {}

    def _get_node_dimension(self, factor, node) -> int:
        """
        Optimized dimension lookup with caching.
        Same interface as original but with performance improvements.
        """
        # Use cached connection lookup if available
        if hasattr(factor, '_connection_cache'):
            return factor._connection_cache.get(node.name,
                                                factor.connection_number.get(node.name, 0))

        # Original logic with caching
        if not hasattr(factor, '_connection_cache'):
            factor._connection_cache = {}


        if node.name in factor.connection_number:
            factor._connection_cache[node.name] = factor.connection_number[node.name]
            return factor.connection_number[node.name]

        # Handle legacy case where objects are used as keys
        for key, dim in factor.connection_number.items():
            if isinstance(key, Agent) and key.name == node.name:
                factor._connection_cache[node.name] = dim
                return dim
            elif isinstance(key, str) and key == node.name:
                factor._connection_cache[node.name] = dim
                return dim

        # Error handling (same as original)
        available_keys = list(factor.connection_number.keys())
        raise KeyError(
            f"Node '{node.name}' not found in factor '{factor.name}' connections. "
            f"Available connections: {available_keys}"
        )

    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Optimized Q message computation - same interface as original.
        Uses vectorized operations for better performance.
        """
        if not messages:
            return []

        variable = messages[0].recipient
        n_messages = len(messages)

        # Fast path for single message
        if n_messages == 1:
            return [Message(
                data=np.zeros_like(messages[0].data),
                sender=variable,
                recipient=messages[0].sender
            )]

        # Vectorized computation when possible
        try:
            # Stack all message data for vectorized operations
            msg_data = np.stack([msg.data for msg in messages])
            outgoing_messages = []

            # Vectorized computation: for each output message i, sum all except i
            for i in range(n_messages):
                # Create boolean mask to exclude message i
                mask = np.ones(n_messages, dtype=bool)
                mask[i] = False

                # Sum remaining messages using vectorized operation
                if np.any(mask):
                    combined_data = np.sum(msg_data[mask], axis=0)
                else:
                    combined_data = np.zeros_like(messages[i].data)

                outgoing_messages.append(Message(
                    data=combined_data,
                    sender=variable,
                    recipient=messages[i].sender
                ))

            return outgoing_messages

        except (ValueError, TypeError):
            # Fallback to original algorithm if vectorization fails
            outgoing_messages = []
            for i, msg_i in enumerate(messages):
                factor = msg_i.sender
                other_messages = [msg_j.data for j, msg_j in enumerate(messages) if j != i]

                if other_messages:
                    combined_data = other_messages[0].copy()
                    for msg_data in other_messages[1:]:
                        combined_data = self.combine_func(combined_data, msg_data)
                else:
                    combined_data = np.zeros_like(msg_i.data)

                outgoing_messages.append(Message(
                    data=combined_data, sender=variable, recipient=factor
                ))

            return outgoing_messages

    def compute_R(self, cost_table, incoming_messages: List[Message]) -> List[Message]:
        """
        Optimized R message computation - same interface as original.
        Uses caching and vectorized operations for better performance.
        """
        if not incoming_messages:
            return []

        factor = incoming_messages[0].recipient

        # Initialize connection cache if needed (same logic as original)
        if not hasattr(factor, "connection_number") or not factor.connection_number:
            factor.connection_number = {}
            for i, msg in enumerate(incoming_messages):
                factor.connection_number[msg.sender.name] = i

        outgoing_messages = []
        cost_table_shape = cost_table.shape
        ndim = len(cost_table_shape)

        # Optimized computation for each message
        for i, msg_i in enumerate(incoming_messages):
            variable_node = msg_i.sender

            try:
                dim = self._get_node_dimension(factor, variable_node)
            except KeyError as e:
                # Same error handling as original
                raise

            # Optimized cost augmentation
            augmented_costs = cost_table.copy()

            # Vectorized addition of messages from other variables
            for j, msg_j in enumerate(incoming_messages):
                if j != i:
                    sender = msg_j.sender
                    sender_dim = self._get_node_dimension(factor, sender)

                    # Cached broadcast shape computation
                    cache_key = (ndim, sender_dim, len(msg_j.data))
                    if cache_key not in self._broadcast_cache:
                        broadcast_shape = [1] * ndim
                        broadcast_shape[sender_dim] = len(msg_j.data)
                        self._broadcast_cache[cache_key] = tuple(broadcast_shape)

                    broadcast_shape = self._broadcast_cache[cache_key]
                    reshaped_msg = msg_j.data.reshape(broadcast_shape)
                    augmented_costs = self.combine_func(augmented_costs, reshaped_msg)

            # Marginalize over all dimensions except the recipient's
            axes_to_reduce = tuple(j for j in range(ndim) if j != dim)
            if axes_to_reduce:
                reduced_msg = self.reduce_func(augmented_costs, axis=axes_to_reduce)
            else:
                reduced_msg = augmented_costs

            # Ensure proper shape
            if reduced_msg.ndim > 1:
                reduced_msg = reduced_msg.ravel()

            outgoing_messages.append(Message(
                data=reduced_msg, sender=factor, recipient=variable_node
            ))

        return outgoing_messages


class MinSumComputator(BPComputator):
    """
    Optimized Min-sum algorithm - drop-in replacement.
    Same interface as original but with performance improvements.
    """

    def __init__(self):
        super().__init__(reduce_func=np.min, combine_func=np.add)


class MaxSumComputator(BPComputator):
    """
    Optimized Max-sum algorithm - drop-in replacement.
    Same interface as original but with performance improvements.
    """

    def __init__(self):
        super().__init__(reduce_func=np.max, combine_func=np.add)