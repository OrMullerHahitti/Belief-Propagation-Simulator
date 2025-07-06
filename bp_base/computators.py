import numpy as np
from typing import List
import logging
from functools import lru_cache

from base_models.protocols import Computator

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from base_models.dcop_base import Agent
from base_models.components import Message

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
        # Enhanced caching for better performance
        self._broadcast_cache = {}
        self._connection_cache = {}
        self._shape_cache = {}
        # Initialize attributes used by optimized version
        # but make them backward compatible
        self._use_jit = False
        self._parallel = False
        self._operation_type = 0  # Default to addition
        self._current_factor = None

    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Highly optimized Q message computation with vectorization.
        Uses pre-allocated arrays and eliminates unnecessary operations.
        """
        early = self._validate(messages=messages)
        if early is not None:
            return early
            
        # Use pre-allocated arrays for vectorized operations
        variable = messages[0].recipient
        n_messages = len(messages)
        
        # Stack all message data once
        msg_data = np.stack([msg.data for msg in messages])
        total_sum = np.sum(msg_data, axis=0)
        
        # Vectorized computation: subtract each message from total
        outgoing_data = total_sum[np.newaxis, :] - msg_data
        
        # Create messages with pre-computed data
        return [
            Message(
                data=outgoing_data[i],
                sender=variable,
                recipient=messages[i].sender,
            )
            for i in range(n_messages)
        ]

        # except (ValueError, TypeError):
        #     # Fallback to the original algorithm if vectorization fails
        #     outgoing_messages = []
        #     for i, msg_i in enumerate(messages):
        #         factor = msg_i.sender
        #         other_messages = [
        #             msg_j.data for j, msg_j in enumerate(messages) if j != i
        #         ]
        #
        #         if other_messages:
        #             combined_data = other_messages[0].copy()
        #             for msg_data in other_messages[1:]:
        #                 combined_data = self.combine_func(combined_data, msg_data)
        #         else:
        #             combined_data = np.zeros_like(msg_i.data)
        #
        #         outgoing_messages.append(
        #             Message(data=combined_data, sender=variable, recipient=factor)
        #         )
        #
        #     return outgoing_messages

    @lru_cache(maxsize=512)
    def _get_broadcast_shapes(self, cost_shape: tuple, n_dims: int) -> tuple:
        """Cache broadcast shapes for reuse."""
        shapes = []
        for axis in range(n_dims):
            shape = [1] * n_dims
            shape[axis] = cost_shape[axis]
            shapes.append(tuple(shape))
        return tuple(shapes)

    def compute_R(self, cost_table: np.ndarray, incoming_messages: List[Message]):
        """Optimized R message computation with pre-allocated arrays and caching."""
        if not incoming_messages:
            return []
            
        k = cost_table.ndim
        shape = cost_table.shape
        dtype = cost_table.dtype
        reduce_msg = np.ndarray.min if self.reduce_func is np.min else np.ndarray.max

        # Cache broadcast shapes
        broadcast_shapes = self._get_broadcast_shapes(shape, k)
        
        # Pre-compute all broadcasted messages and reduce axes
        b_msgs = []
        axes_cache = []
        
        for axis, msg in enumerate(incoming_messages):
            q = np.asarray(msg.data, dtype=dtype)
            br = q.reshape(broadcast_shapes[axis])
            b_msgs.append(br)
            axes_cache.append(tuple(j for j in range(k) if j != axis))

        # Single aggregation step - avoid copy when possible
        agg = cost_table.astype(dtype, copy=False) if cost_table.dtype == dtype else cost_table.astype(dtype)
        for q in b_msgs:
            agg = agg + q  # Use broadcasting efficiently

        # Compute all R messages with optimized memory access
        out = []
        for axis, (broadcasted_q, msg) in enumerate(zip(b_msgs, incoming_messages)):
            r_vec = reduce_msg(agg - broadcasted_q, axis=axes_cache[axis])
            out.append(
                Message(
                    data=r_vec,
                    sender=msg.recipient,
                    recipient=msg.sender,
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

    @lru_cache(maxsize=2048)
    def _get_broadcast_shape(self, ct_dim: int, sender_dim: int, msg_len: int) -> tuple:
        """Cached broadcast shape computation with enhanced cache size."""
        shape = [1] * ct_dim
        shape[sender_dim] = msg_len
        return tuple(shape)
    
    def _create_message_key(self, messages):
        """Create a cache key for message computations."""
        # Use a lightweight signature based on message shapes and senders
        key_parts = []
        for msg in messages:
            sender_id = id(msg.sender) if hasattr(msg, 'sender') else 0
            data_shape = msg.data.shape if hasattr(msg.data, 'shape') else len(msg.data)
            key_parts.append((sender_id, data_shape))
        return tuple(key_parts)


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
