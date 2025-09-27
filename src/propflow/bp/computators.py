import numpy as np
from typing import List
import logging
import functools
from functools import lru_cache

from ..core.protocols import Computator

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from ..core.components import Message

# Minimal logging for computators
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class BPComputator(Computator):
    """A vectorized, cache-friendly Belief Propagation computator.

    This class provides a highly optimized implementation for the core
    computations in belief propagation algorithms. It uses function dispatch
    tables for common operations (e.g., `min`, `max`, `sum`, `add`, `multiply`)
    to minimize overhead and leverages vectorized numpy operations for
    performance.

    The behavior of the computator (e.g., Min-Sum, Max-Product) is determined
    by the `reduce_func` and `combine_func` arguments passed during
    initialization.

    Attributes:
        reduce_func (Callable): The function used for message reduction (e.g., `np.min`).
        combine_func (Callable): The function used for message combination (e.g., `np.add`).
    """
    # Function dispatch tables for zero-overhead lookups
    _REDUCE_DISPATCH = {
        np.min: (np.ndarray.min, np.ndarray.argmin),
        np.max: (np.ndarray.max, np.ndarray.argmax),
        np.sum: (np.ndarray.sum, np.ndarray.argmax),
    }

    _COMBINE_DISPATCH = {
        np.add: (np.sum, np.subtract, np.zeros),
        np.multiply: (np.prod, np.divide, np.ones),
    }

    def __init__(self, reduce_func=np.min, combine_func=np.add):
        """Initializes the BPComputator.

        Args:
            reduce_func (Callable): The function to use for reducing messages
                (e.g., `np.min` for Min-Sum). Defaults to `np.min`.
            combine_func (Callable): The function to use for combining messages
                (e.g., `np.add` for Min-Sum). Defaults to `np.add`.
        """
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        self._connection_cache = {}

        # Pre-select optimized functions using dispatch tables
        self._reduce_msg, self._argreduce_func = self._setup_reduce_functions(
            reduce_func
        )
        (
            self._combine_axis,
            self._combine_inverse,
            self._belief_identity,
        ) = self._setup_combine_functions(combine_func)

    def _setup_reduce_functions(self, reduce_func):
        """Sets up reduce functions from the dispatch table for performance."""
        if reduce_func in self._REDUCE_DISPATCH:
            return self._REDUCE_DISPATCH[reduce_func]
        else:
            # Generic fallback for custom reduce functions
            return (
                lambda x, axis: reduce_func(x, axis=axis),
                np.ndarray.argmax,  # Default to argmax
            )

    def _setup_combine_functions(self, combine_func):
        """Sets up combine functions from the dispatch table for performance."""
        if combine_func in self._COMBINE_DISPATCH:
            return self._COMBINE_DISPATCH[combine_func]
        else:
            # Generic fallback
            return (
                lambda x, axis: np.apply_along_axis(
                    lambda arr: functools.reduce(combine_func, arr), axis, x
                ),
                None,  # No inverse function available
                np.ones,  # Safe default for identity
            )

    def _remove_message_from_aggregate(
        self, agg, message_to_remove, all_messages, axis, cost_table=None
    ):
        """Efficiently removes a message from an aggregate.

        Uses a fast inverse operation if available (e.g., subtraction for
        addition), otherwise falls back to re-computing the aggregate.

        Args:
            agg (np.ndarray): The current aggregate array.
            message_to_remove (np.ndarray): The message data to remove.
            all_messages (list): A list of all message data for fallback.
            axis (int): The index of the message to remove.
            cost_table (np.ndarray, optional): The cost table for fallback.

        Returns:
            np.ndarray: The aggregate with the message removed.
        """
        if self._combine_inverse is not None:
            return self._combine_inverse(agg, message_to_remove)
        else:
            # Fallback: recompute aggregate without this message
            if cost_table is not None:
                temp_agg = cost_table.astype(agg.dtype, copy=True)
                for i, msg in enumerate(all_messages):
                    if i != axis:
                        self.combine_func(temp_agg, msg, out=temp_agg)
            else:
                temp_agg = self._belief_identity(agg.shape).astype(agg.dtype)
                for i, msg in enumerate(all_messages):
                    if i != axis:
                        temp_agg = self.combine_func(temp_agg, msg)
            return temp_agg

    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """Computes outgoing messages from a variable node to factor nodes (Q messages).

        This is an optimized, vectorized implementation.

        Args:
            messages: A list of incoming messages from factor nodes.

        Returns:
            A list of computed messages to be sent to factor nodes.
        """
        early = self._validate(messages=messages)
        if early is not None:
            return early

        variable = messages[0].recipient
        n_messages = len(messages)

        msg_data = np.stack([msg.data for msg in messages])
        total_combined = self._combine_axis(msg_data, axis=0)
        outgoing_messages = []

        for i in range(n_messages):
            combined_data = self._remove_message_from_aggregate(
                total_combined, msg_data[i], msg_data, i
            )
            outgoing_messages.append(
                Message(
                    data=combined_data,
                    sender=variable,
                    recipient=messages[i].sender,
                )
            )
        return outgoing_messages

    def compute_R(self, cost_table: np.ndarray, incoming_messages: List[Message]) -> List[Message]:
        """Computes outgoing messages from a factor node to variable nodes (R messages).

        This is an optimized, vectorized implementation that involves three main steps:
        1. Broadcast each incoming Q message to the dimensionality of the cost table.
        2. Combine the cost table with all broadcasted Q messages once.
        3. For each recipient, efficiently "remove" its Q message from the aggregate
           and reduce to produce the outgoing R message.

        Args:
            cost_table: The factor's cost table.
            incoming_messages: A list of incoming messages from variable nodes.

        Returns:
            A list of computed messages to be sent to variable nodes.
        """
        k = cost_table.ndim
        shape = cost_table.shape
        dtype = cost_table.dtype
        combine_func = self.combine_func
        reduce_msg = self._reduce_msg

        b_msgs = []
        axes_cache = []
        for axis, msg in enumerate(incoming_messages):
            q = np.asarray(msg.data, dtype=dtype)
            br = q.reshape([shape[axis] if i == axis else 1 for i in range(k)])
            b_msgs.append(br)
            axes_cache.append(tuple(j for j in range(k) if j != axis))

        agg = cost_table.astype(dtype, copy=True)
        for q in b_msgs:
            combine_func(agg, q, out=agg)

        out = []
        for axis, broadcasted_q in enumerate(b_msgs):
            temp = self._remove_message_from_aggregate(
                agg, broadcasted_q, b_msgs, axis, cost_table
            )
            r_vec = reduce_msg(temp, axis=axes_cache[axis])
            out.append(
                Message(
                    data=r_vec,
                    sender=incoming_messages[axis].recipient,
                    recipient=incoming_messages[axis].sender,
                )
            )
        return out

    def _validate(self, messages=None, cost_table=None, incoming_messages=None):
        """Validates inputs and handles edge cases for compute methods."""
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
        """Optimized dimension lookup with caching."""
        cache_key = (id(factor), node.name)
        if cache_key in self._connection_cache:
            return self._connection_cache[cache_key]

        if hasattr(factor, "connection_number") and factor.connection_number:
            if node.name in factor.connection_number:
                dim = factor.connection_number[node.name]
                self._connection_cache[cache_key] = dim
                return dim

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
        """Gets the optimal assignment from a belief vector.

        Uses the pre-selected `_argreduce_func` (e.g., `argmin`, `argmax`)
        for zero-overhead execution.

        Args:
            belief: The belief vector.

        Returns:
            The index of the optimal assignment.
        """
        return int(self._argreduce_func(belief))

    def compute_belief(self, messages: List[Message], domain: int) -> np.ndarray:
        """Computes the belief of a variable node from incoming messages.

        Args:
            messages: A list of incoming messages.
            domain: The domain size of the variable.

        Returns:
            A numpy array representing the belief distribution.
        """
        if not messages:
            return np.ones(domain) / domain

        belief = self._belief_identity(domain)
        for message in messages:
            belief = self.combine_func(belief, message.data)

        return belief


class MinSumComputator(BPComputator):
    """A computator for the Min-Sum belief propagation algorithm.

    This is equivalent to finding the Most Probable Explanation (MPE) in a
    graphical model represented in the log-domain. It combines messages
    using addition and reduces them using the min operator.
    """

    def __init__(self):
        """Initializes the MinSumComputator."""
        super().__init__(reduce_func=np.min, combine_func=np.add)


class MaxSumComputator(BPComputator):
    """A computator for the Max-Sum belief propagation algorithm.

    This is used for problems where the goal is to maximize a sum of utilities.
    It combines messages using addition and reduces them using the max operator.
    """

    def __init__(self):
        """Initializes the MaxSumComputator."""
        super().__init__(reduce_func=np.max, combine_func=np.add)


class MaxProductComputator(BPComputator):
    """A computator for the Max-Product belief propagation algorithm.

    This is equivalent to finding the Most Probable Explanation (MPE) in a
    graphical model. It combines messages using multiplication and reduces
    them using the max operator.
    """

    def __init__(self):
        """Initializes the MaxProductComputator."""
        super().__init__(reduce_func=np.max, combine_func=np.multiply)


class SumProductComputator(BPComputator):
    """A computator for the Sum-Product belief propagation algorithm.

    This is used to compute marginal probabilities in a graphical model.
    It combines messages using multiplication and reduces them (marginalizes)
    using summation.
    """

    def __init__(self):
        """Initializes the SumProductComputator."""
        super().__init__(reduce_func=np.sum, combine_func=np.multiply)
