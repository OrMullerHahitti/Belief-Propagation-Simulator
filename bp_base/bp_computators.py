import numpy as np
from typing import List, Optional
import logging

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


def _compute_Q_numpy(msg_data: np.ndarray) -> np.ndarray:
    """Compute Q messages using pure numpy operations."""
    total_sum = np.sum(msg_data, axis=0)
    result = np.empty_like(msg_data)
    for i in range(msg_data.shape[0]):
        result[i] = total_sum - msg_data[i]
    return result


def _compute_R_numpy(
    cost_table: np.ndarray,
    incoming: np.ndarray,
    dims: np.ndarray,
    reduce_type: int,
    op_type: int,
) -> np.ndarray:
    """Compute R messages using numpy for environments without numba."""
    n_messages = incoming.shape[0]
    domain = incoming.shape[1]
    ndim = cost_table.ndim
    out = np.empty((n_messages, domain))
    for i in range(n_messages):
        augmented = cost_table.copy()
        for j in range(n_messages):
            if j != i:
                broadcast_shape = [1] * ndim
                broadcast_shape[dims[j]] = domain
                reshaped = incoming[j].reshape(tuple(broadcast_shape))
                if op_type == 0:
                    augmented = augmented + reshaped
                else:
                    augmented = augmented * reshaped
        axes = tuple(k for k in range(ndim) if k != dims[i])
        if axes:
            if reduce_type == 0:
                reduced = augmented.min(axis=axes)
            else:
                reduced = augmented.max(axis=axes)
        else:
            reduced = augmented
        if reduced.ndim > 1:
            reduced = reduced.ravel()
        out[i] = reduced
    return out


if HAS_NUMBA:
    from numba import njit

    @njit
    def _compute_Q_jit(msg_data: np.ndarray) -> np.ndarray:
        """JIT-compiled helper mirroring :func:`_compute_Q_numpy`."""
        total_sum = np.sum(msg_data, axis=0)
        out = np.empty_like(msg_data)
        for i in range(msg_data.shape[0]):
            out[i] = total_sum - msg_data[i]
        return out

    @njit
    def _compute_R_jit(
        cost_table: np.ndarray,
        incoming: np.ndarray,
        dims: np.ndarray,
        reduce_type: int,
        op_type: int,
    ) -> np.ndarray:
        """JIT-compiled helper mirroring :func:`_compute_R_numpy`."""
        n_messages = incoming.shape[0]
        domain = incoming.shape[1]
        ndim = cost_table.ndim
        out = np.empty((n_messages, domain))
        for i in range(n_messages):
            augmented = cost_table.copy()
            for j in range(n_messages):
                if j != i:
                    broadcast_shape = [1] * ndim
                    broadcast_shape[dims[j]] = domain
                    reshaped = incoming[j].reshape(tuple(broadcast_shape))
                    if op_type == 0:
                        augmented = augmented + reshaped
                    else:
                        augmented = augmented * reshaped
            axes_list = [k for k in range(ndim) if k != dims[i]]
            if len(axes_list) > 0:
                axes = tuple(axes_list)
                if reduce_type == 0:
                    reduced = augmented.min(axis=axes)
                else:
                    reduced = augmented.max(axis=axes)
            else:
                reduced = augmented
            if reduced.ndim > 1:
                reduced = reduced.ravel()
            out[i] = reduced
        return out

else:
    _compute_Q_jit = _compute_Q_numpy
    _compute_R_jit = _compute_R_numpy


class BPComputator:
    """
    Vectorized, cache-friendly version of the original BPComputator.
    """

    def __init__(self, reduce_func, combine_func, use_jit: bool = True):
        self.reduce_func = reduce_func
        self.combine_func = combine_func

        # Cache frequently used operations
        self._broadcast_cache = {}

        # Determine if JIT should be used
        self._use_jit = bool(use_jit and HAS_NUMBA)

        # Operation/reduction types used by jitted functions
        if combine_func is np.add:
            self._operation_type = 0
        elif combine_func is np.multiply:
            self._operation_type = 1
        else:
            self._operation_type = -1

        if reduce_func is np.min:
            self._reduce_type = 0
        elif reduce_func is np.max:
            self._reduce_type = 1
        else:
            self._reduce_type = -1

        self._current_factor = None

        # Choose implementations based on JIT availability
        if self._use_jit:
            self._compute_Q_impl = _compute_Q_jit
            self._compute_R_impl = _compute_R_jit
        else:
            self._compute_Q_impl = _compute_Q_numpy
            self._compute_R_impl = _compute_R_numpy

    def _get_node_dimension(self, factor, node) -> int:
        """
        Optimized dimension lookup with caching.
        Same interface as original but with performance improvements.
        """
        # Use cached connection lookup if available
        if hasattr(factor, "_connection_cache"):
            return factor._connection_cache.get(
                node.name, factor.connection_number.get(node.name, 0)
            )

        # Original logic with caching
        if not hasattr(factor, "_connection_cache"):
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

    def compute_Q(
        self, messages: List[Message], use_jit: Optional[bool] = None
    ) -> List[Message]:
        """Compute Q messages with optional JIT acceleration."""
        if not messages:
            return []

        variable = messages[0].recipient
        n_messages = len(messages)

        if n_messages == 1:
            return [
                Message(
                    data=np.zeros_like(messages[0].data),
                    sender=variable,
                    recipient=messages[0].sender,
                )
            ]

        msg_data = np.stack([msg.data for msg in messages])

        use_jit = self._use_jit if use_jit is None else bool(use_jit and HAS_NUMBA)

        try:
            results = (self._compute_Q_impl if use_jit else _compute_Q_numpy)(msg_data)
        except Exception:
            results = _compute_Q_numpy(msg_data)

        return [
            Message(data=results[i], sender=variable, recipient=messages[i].sender)
            for i in range(n_messages)
        ]

    def compute_R(
        self,
        cost_table,
        incoming_messages: List[Message],
        use_jit: Optional[bool] = None,
    ) -> List[Message]:
        """Compute R messages with optional JIT acceleration."""
        if not incoming_messages:
            return []

        factor = incoming_messages[0].recipient

        if not hasattr(factor, "connection_number") or not factor.connection_number:
            factor.connection_number = {
                msg.sender.name: i for i, msg in enumerate(incoming_messages)
            }

        dims = np.array(
            [self._get_node_dimension(factor, msg.sender) for msg in incoming_messages]
        )
        msg_data = np.stack([msg.data for msg in incoming_messages])

        use_jit = self._use_jit if use_jit is None else bool(use_jit and HAS_NUMBA)

        try:
            results = (self._compute_R_impl if use_jit else _compute_R_numpy)(
                cost_table, msg_data, dims, self._reduce_type, self._operation_type
            )
        except Exception:
            results = _compute_R_numpy(
                cost_table, msg_data, dims, self._reduce_type, self._operation_type
            )

        return [
            Message(
                data=results[i], sender=factor, recipient=incoming_messages[i].sender
            )
            for i in range(len(incoming_messages))
        ]


class MinSumComputator(BPComputator):
    """
    Optimized Min-sum algorithm - drop-in replacement.
    Same interface as original but with performance improvements.
    """

    def __init__(self, use_jit: bool = True):
        super().__init__(reduce_func=np.min, combine_func=np.add, use_jit=use_jit)


class MaxSumComputator(BPComputator):
    """
    Optimized Max-sum algorithm - drop-in replacement.
    Same interface as original but with performance improvements.
    """

    def __init__(self, use_jit: bool = True):
        super().__init__(reduce_func=np.max, combine_func=np.add, use_jit=use_jit)
