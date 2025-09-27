"""Factory functions for creating cost tables for factor nodes.

This module provides a set of functions to generate numpy arrays that can be
used as cost tables in factor graphs. It includes a general-purpose creator
and several convenience wrappers for common random distributions.
"""
import numpy as np
from typing import Callable, Any
from ...core.protocols import CostTable
from scipy.special import logsumexp


def _create_cost_table(
    connections: int, domain: int, policy: Callable[..., np.ndarray], **policy_params: Any
) -> CostTable:
    """A generic factory for creating a cost table of a given shape and distribution.

    Args:
        connections: The number of variables connected to the factor, which
            determines the number of dimensions of the cost table.
        domain: The domain size for each variable, which determines the size
            of each dimension.
        policy: A numpy random function used to generate the values (e.g.,
            `np.random.randint`, `np.random.uniform`).
        **policy_params: Additional keyword arguments to pass to the `policy` function.

    Returns:
        An n-dimensional numpy array representing the cost table.
    """
    shape = tuple([domain] * connections)
    return policy(**policy_params, size=shape)


def create_random_int_table(n: int, domain: int, low: int = 0, high: int = 10) -> CostTable:
    """Creates a cost table with random integer values from a uniform distribution.

    Args:
        n: The number of dimensions for the cost table (number of connected variables).
        domain: The size of the domain for each variable.
        low: The lower bound of the random integer range (inclusive).
        high: The upper bound of the random integer range (exclusive).

    Returns:
        A numpy array representing the cost table.
    """
    return _create_cost_table(n, domain, np.random.randint, low=low, high=high)


def create_uniform_table(n: int, domain: int, low: float = 0.0, high: float = 1.0) -> CostTable:
    """Creates a cost table with random float values from a uniform distribution.

    Args:
        n: The number of dimensions for the cost table.
        domain: The size of the domain for each variable.
        low: The lower bound of the distribution.
        high: The upper bound of the distribution.

    Returns:
        A numpy array representing the cost table.
    """
    return _create_cost_table(n, domain, np.random.uniform, low=low, high=high)


def create_normal_table(n: int, domain: int, loc: float = 0.0, scale: float = 1.0) -> CostTable:
    """Creates a cost table with values from a normal (Gaussian) distribution.

    Args:
        n: The number of dimensions for the cost table.
        domain: The size of the domain for each variable.
        loc: The mean of the distribution.
        scale: The standard deviation of the distribution.

    Returns:
        A numpy array representing the cost table.
    """
    return _create_cost_table(n, domain, np.random.normal, loc=loc, scale=scale)


def create_exponential_table(n: int, domain: int, scale: float = 1.0) -> CostTable:
    """Creates a cost table with values from an exponential distribution.

    Args:
        n: The number of dimensions for the cost table.
        domain: The size of the domain for each variable.
        scale: The scale parameter (beta = 1/lambda) of the distribution.

    Returns:
        A numpy array representing the cost table.
    """
    return _create_cost_table(n, domain, np.random.exponential, scale=scale)


def create_symmetric_cost_table(n: int, m: int) -> CostTable:
    """Creates a 2D symmetric cost table.

    The resulting table will have the property `cost[i, j] == cost[j, i]`.

    Args:
        n: The size of the first dimension.
        m: The size of the second dimension.

    Returns:
        A symmetric 2D numpy array.
    """
    cost_table = np.random.rand(n, m)
    return (cost_table + cost_table.T) / 2


def normalize_cost_table(cost_table: np.ndarray, axis: int = None) -> np.ndarray:
    """Normalizes a cost table into a probability distribution using log-domain softmin.

    This is useful for converting raw costs into a probabilistic representation
    where lower costs correspond to higher probabilities.

    Args:
        cost_table: The raw cost table (e.g., integers or floats).
        axis: The axis along which to normalize. If None, normalization is
            performed over the entire table.

    Returns:
        A normalized numpy array where values sum to 1 (across the specified axis).
    """
    log_potentials = -cost_table.astype(float)
    logZ = logsumexp(log_potentials, axis=axis, keepdims=True)
    log_probs = log_potentials - logZ
    return np.exp(log_probs)
