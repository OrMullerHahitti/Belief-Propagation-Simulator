import numpy as np
from typing import List
import logging
import functools
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
        
        # Pre-compile optimized functions for zero overhead
        if reduce_func is np.min:
            self._reduce_msg = np.ndarray.min
            self._argreduce_func = np.ndarray.argmin
        elif reduce_func is np.max:
            self._reduce_msg = np.ndarray.max
            self._argreduce_func = np.ndarray.argmax
        elif reduce_func is np.sum:
            self._reduce_msg = np.ndarray.sum
            self._argreduce_func = np.ndarray.argmax  # Sum-product typically uses argmax for assignment
        else:
            # Generic fallback for custom reduce functions
            self._reduce_msg = lambda x, axis: reduce_func(x, axis=axis)
            self._argreduce_func = np.ndarray.argmax  # Default to argmax
        
        # Pre-compile combine functions for compute_Q and beliefs
        if combine_func is np.add:
            self._combine_axis = np.sum
            self._combine_inverse = np.subtract
            self._belief_identity = np.zeros
        elif combine_func is np.multiply:
            self._combine_axis = np.prod
            self._combine_inverse = np.divide
            self._belief_identity = np.ones
        else:
            # Generic fallback (will have function call overhead)
            self._combine_axis = lambda x, axis: np.apply_along_axis(
                lambda arr: np.functools.reduce(combine_func, arr), axis, x
            )
            self._combine_inverse = None  # Will need special handling
            self._belief_identity = np.ones  # Safe default

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
        total_combined = self._combine_axis(msg_data, axis=0)
        outgoing_messages = []

        # Vectorized computation: combine all except own message
        for i in range(n_messages):
            if self._combine_inverse is not None:
                # Fast path for add/multiply operations
                combined_data = self._combine_inverse(total_combined, msg_data[i])
            else:
                # Generic fallback: recompute without message i
                other_msgs = np.concatenate([msg_data[:i], msg_data[i+1:]], axis=0)
                combined_data = self._combine_axis(other_msgs, axis=0) if len(other_msgs) > 0 else self._belief_identity(msg_data[i].shape)
            
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
        combine_func = self.combine_func
        reduce_msg = self._reduce_msg

        # 1) broadcast each Q once
        b_msgs = []
        axes_cache = []
        for axis, msg in enumerate(incoming_messages):
            q = np.asarray(msg.data, dtype=dtype)
            br = q.reshape([shape[axis] if i == axis else 1 for i in range(k)])
            b_msgs.append(br)
            axes_cache.append(tuple(j for j in range(k) if j != axis))

        # 2) aggregate once  (F combined with all Q messages)
        agg = cost_table.astype(dtype, copy=True)
        for q in b_msgs:
            combine_func(agg, q, out=agg)  # Use modular combine function

        # 3) build each R_i
        out = []
        for axis, broadcasted_q in enumerate(b_msgs):
            # Use inverse operation to "remove" the message from aggregate
            if self._combine_inverse is not None:
                # Fast path for add/multiply operations
                temp = self._combine_inverse(agg, broadcasted_q) #remove the message that is being sent to (the one sent before)
                r_vec = reduce_msg(temp, axis=axes_cache[axis])
            else:
                # Generic fallback: recompute aggregate without this message
                temp_agg = cost_table.astype(dtype, copy=True)
                for i, q in enumerate(b_msgs):
                    if i != axis:
                        combine_func(temp_agg, q, out=temp_agg)
                r_vec = reduce_msg(temp_agg, axis=axes_cache[axis])
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

    def get_assignment(self, belief: np.ndarray) -> int:
        """Get optimal assignment from belief vector with zero overhead."""
        return int(self._argreduce_func(belief))

    def compute_belief(self, messages: List[Message], domain: int) -> np.ndarray:
        """Compute belief from incoming messages using modular combine function."""
        if not messages:
            return np.ones(domain) / domain  # Uniform belief
        
        # Initialize belief with identity element
        belief = self._belief_identity(domain)
        
        # Combine all incoming messages
        for message in messages:
            belief = self.combine_func(belief, message.data)
        
        return belief


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


class MaxProductComputator(BPComputator):
    """
    Max-product algorithm using multiplication for combining messages.
    """

    def __init__(self):
        super().__init__(reduce_func=np.max, combine_func=np.multiply)


class SumProductComputator(BPComputator):
    """
    Sum-product algorithm using multiplication for combining messages and sum for reduction.
    """

    def __init__(self):
        super().__init__(reduce_func=np.sum, combine_func=np.multiply)
