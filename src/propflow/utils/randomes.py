"""Utilities for generating random numpy arrays based on different distributions.

This module provides several 'randomness policies' (functions that generate
random data for a given shape) and factory functions to create random messages
and cost tables using these policies. This approach allows for easy dependency
injection of different random generation strategies.
"""
import numpy as np
from typing import Callable, Tuple

RandomnessPolicy = Callable[[Tuple[int, ...]], np.ndarray]
"""A type alias for a function that generates a random numpy array.

A randomness policy function should accept a tuple specifying the array shape
and return a numpy array of that shape.
"""


def uniform_random(shape: Tuple[int, ...]) -> np.ndarray:
    """Generates a numpy array with values from a uniform distribution [0.0, 1.0).

    Args:
        shape: The shape of the output array.

    Returns:
        A numpy array with random values.
    """
    return np.random.rand(*shape)


def normal_random(shape: Tuple[int, ...], mean: float = 0, std: float = 1) -> np.ndarray:
    """Generates a numpy array with values from a normal (Gaussian) distribution.

    Args:
        shape: The shape of the output array.
        mean: The mean of the normal distribution. Defaults to 0.
        std: The standard deviation of the normal distribution. Defaults to 1.

    Returns:
        A numpy array with random values.
    """
    return np.random.normal(size=shape, loc=mean, scale=std)


def integer_random(shape: Tuple[int, ...], low: int = 0, high: int = 10) -> np.ndarray:
    """Generates a numpy array of random integers within a specified range.

    The generated integers will be in the interval [low, high).

    Args:
        shape: The shape of the output array.
        low: The lower bound of the random integer range (inclusive).
        high: The upper bound of the random integer range (exclusive).

    Returns:
        A numpy array with random integer values.
    """
    return np.random.randint(low, high, size=shape)


def create_random_message(
    domain_size: int, randomness_policy: RandomnessPolicy = integer_random
) -> np.ndarray:
    """Creates a 1D random numpy array to represent a message.

    This function uses a specified randomness policy to generate the data.

    Args:
        domain_size: The size (length) of the message array.
        randomness_policy: The function used to generate the random data.
            Defaults to `integer_random`.

    Returns:
        A 1D numpy array of shape (domain_size,).
    """
    return randomness_policy((domain_size,))


def create_random_table(
    shape: Tuple[int, ...], randomness_policy: RandomnessPolicy = integer_random
) -> np.ndarray:
    """Creates a multi-dimensional random numpy array to represent a cost table.

    This function uses a specified randomness policy to generate the data.

    Args:
        shape: A tuple specifying the shape of the cost table.
        randomness_policy: The function used to generate the random data.
            Defaults to `integer_random`.

    Returns:
        A numpy array with the specified shape.
    """
    return randomness_policy(shape)
