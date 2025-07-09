import numpy as np
from typing import List
import logging
from functools import lru_cache

from src.propflow.base_models.protocols import Computator

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.propflow.base_models.components import Message

# Minimal logging for computators
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class BPComputator(Computator):
    """
    Vectorized, cache-friendly version of the original BPComputator.
    Same interface as original but optimized for performance.
    """

    def __init__(self, reduce_func=np.min, combine_func=np.add):
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        # Cache frequently used operations
        self._connection_cache = {}

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

    def compute_R(self, cost_table: np.ndarray, incoming_messages: List[Message]):
        k = cost_table.ndim
        shape = cost_table.shape
        dtype = cost_table.dtype
        add = np.add
        reduce_msg = np.ndarray.min if self.reduce_func is np.min else np.ndarray.max

        # 1) broadcast each Q once
        b_msgs = []
        axes_cache = []
        for axis, msg in enumerate(incoming_messages):
            q = np.asarray(msg.data, dtype=dtype)
            br = q.reshape([shape[axis] if i == axis else 1 for i in range(k)])
            b_msgs.append(br)
            axes_cache.append(tuple(j for j in range(k) if j != axis))

        # 2) aggregate once  (F + sum of Q)
        agg = cost_table.astype(dtype, copy=True)
        for q in b_msgs:
            add(agg, q, out=agg)  # => no new array, no wrappers

        # 3) build each R_i
        out = []
        for axis, broadcasted_q in enumerate(b_msgs):
            r_vec = reduce_msg(agg - broadcasted_q, axis=axes_cache[axis])  # ndarray.min/max
            out.append(
                Message(
                    data=r_vec,
                    sender=incoming_messages[axis].recipient,
                    recipient=incoming_messages[axis].sender,
                )
            )
        return out

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
