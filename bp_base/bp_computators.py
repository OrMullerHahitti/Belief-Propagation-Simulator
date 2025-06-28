import numpy as np
from typing import List
import logging
from functools import lru_cache

from base_all.agents import FactorAgent
from base_all.protocols import Computator

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from base_all.DCOP_base import Agent
from base_all.components import Message

# Minimal logging for computators
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class BPComputator(Computator):
    """
    Vectorized, cache-friendly version of the original BPComputator.
    Same interface as original but optimized for performance.
    """

    def __init__(self, reduce_func, combine_func):
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        # Cache frequently used operations
        self._broadcast_cache = {}
        self._connection_cache = {}
        # Initialize attributes used by optimized version
        # but make them backward compatible
        self._use_jit = False
        self._parallel = False
        self._operation_type = 0  # Default to addition
        self._current_factor = None



    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Optimized Q message computation - same interface as original.
        Uses vectorized operations for better performance.
        """
        early = self._validate(messages=messages)
        if early is not None:
            return early
        # the recipient is the same for all messages
        variable = messages[0].recipient
        n_messages = len(messages)

        # Vectorized computation when possible
        # try:
        # Stack all message data for vectorized operations
        msg_data = np.stack([msg.data for msg in messages])
        total_sum = np.sum(msg_data, axis=0)
        outgoing_messages = []

        # Vectorized computation: subtract own message from total
        for i in range(n_messages):
            combined_data = total_sum - msg_data[i]
            outgoing_messages.append(
                Message(
                    data=combined_data,
                    sender=variable,
                    recipient=messages[i].sender,
                )
            )

        return outgoing_messages

        # except (ValueError, TypeError):
        #     # Fallback to the original algorithm if vectorization fails
        #     outgoing_messages = []
        #     for i, message_i in enumerate(messages):
        #         factor = message_i.sender
        #         other_messages = [
        #             message_j.data for j, message_j in enumerate(messages) if j != i
        #         ]
        #
        #         if other_messages:
        #             combined_data = other_messages[0].copy()
        #             for msg_data in other_messages[1:]:
        #                 combined_data = self.combine_func(combined_data, msg_data)
        #         else:
        #             combined_data = np.zeros_like(message_i.data)
        #
        #         outgoing_messages.append(
        #             Message(data=combined_data, sender=variable, recipient=factor)
        #         )
        #
        #     return outgoing_messages

    def compute_R(self, cost_table, incoming_messages: List[Message]) -> List[Message]:
        """
        Optimized R message computation - same interface as original.
        Uses caching and vectorized operations for better performance.
        """
        early = self._validate(
            cost_table=cost_table, incoming_messages=incoming_messages
        )
        if early is not None:
            return early
        factor: FactorAgent = incoming_messages[0].recipient

        outgoing_messages = []
        cost_table_shape = cost_table.shape
        ct_dim = len(cost_table_shape)

        # Optimized computation for each message
        for i, message_i in enumerate(incoming_messages):
            variable_node = message_i.sender
            dim = self._get_node_dimension(factor, variable_node)

            # Optimized cost augmentation
            augmented_costs = cost_table + 0

            # Vectorized addition of messages from other variables
            for j, message_j in enumerate(incoming_messages):
                if j != i:
                    sender = message_j.sender
                    sender_dim = self._get_node_dimension(factor, sender)

                    # Cached broadcast shape computation
                    broadcast_shape = self._get_broadcast_shape(
                        ct_dim, sender_dim, len(message_j.data)
                    )
                    shaped_message = message_j.data.reshape(broadcast_shape)
                    augmented_costs = self.combine_func(augmented_costs, shaped_message)

            # Marginalize over all dimensions except the recipient's
            axes_to_reduce = tuple(j for j in range(ct_dim) if j != dim)
            if axes_to_reduce:
                reduced_msg = self.reduce_func(augmented_costs, axis=axes_to_reduce)
            else:
                reduced_msg = augmented_costs

            # if reduced_msg.ndim > 1:
            #     reduced_msg = reduced_msg.ravel()

            outgoing_messages.append(
                Message(data=reduced_msg, sender=factor, recipient=variable_node)
            )

        return outgoing_messages


    def _validate(self, messages=None, cost_table=None, incoming_messages=None):
        """
        Validate and handle early return cases for compute_Q and compute_R.
        """
        if messages is not None:
            if not messages:
                return []
            if len(messages) == 1:
                variable = messages[0].recipient
                return [
                    Message(
                        data=np.zeros_like(messages[0].data),
                        sender=variable,
                        recipient=messages[0].sender,
                    )
                ]
        if incoming_messages is not None:
            if not incoming_messages:
                return []
            factor = incoming_messages[0].recipient
            if not hasattr(factor, "connection_number") or not factor.connection_number:
                factor.connection_number = {}
                for i, msg in enumerate(incoming_messages):
                    factor.connection_number[msg.sender.name] = i
        return None

    def _get_node_dimension(self, factor, node) -> int:
        """
        Optimized dimension lookup with caching.
        Same interface as original but with performance improvements.
        """
        # Use cached connection lookup if available
        cache_key = (id(factor), node.name)

        if cache_key in self._connection_cache:
            return self._connection_cache[cache_key]

        # Original logic with caching
        if hasattr(factor, "connection_number") and factor.connection_number:
            if node.name in factor.connection_number:
                dim = factor.connection_number[node.name]
                self._connection_cache[cache_key] = dim
                return dim

        # Handle case where objects are used as keys #TODO; never use objects as keys
        for key, dim in getattr(factor, "connection_number", {}).items():
            if isinstance(key, Agent) and key.name == node.name:
                self._connection_cache[cache_key] = dim
                return dim
            elif isinstance(key, str) and key == node.name:
                self._connection_cache[cache_key] = dim
                return dim

        # Error handling
        available_keys = list(getattr(factor, "connection_number", {}).keys())
        raise KeyError(
            f"Node '{node.name}' not found in factor '{factor.name}' connections. "
            f"Available connections: {available_keys}"
        )

    @lru_cache(maxsize=1024)
    def _get_broadcast_shape(self, ct_dim: int, sender_dim: int, msg_len: int) -> tuple:
        """Cached broadcast shape computation."""
        shape = [1] * ct_dim
        shape[sender_dim] = msg_len
        return tuple(shape)


class MinSumComputator(BPComputator):
    """
    Min-sum algorithm.
    """

    def __init__(self):
        super().__init__(reduce_func=np.min, combine_func=np.add)


class MaxSumComputator(BPComputator):
    """
    Max-sum algorithm .
    """

    def __init__(self):
        super().__init__(reduce_func=np.max, combine_func=np.add)
