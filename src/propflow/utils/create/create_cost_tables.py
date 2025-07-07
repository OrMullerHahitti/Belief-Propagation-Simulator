import numpy as np
from typing import Callable
from src.propflow.base_models.protocols import CostTable
from scipy.special import logsumexp


def _create_cost_table(
    connections: int, domain: int, policy: Callable = np.random.randint, **policy_params
) -> CostTable:
    """
    Main funtion that will be used in variations below
    Create a random cost table with shape (d0, d1, ..., dn-1).

    Args:
        connections: Number of connected variables to factor
        domain: domain size
        policy: policy function , e.g., np.random.randint, np.random.uniform, etc.
        policy_params: Additional parameters for the policy

    Returns:
        n-dimensional cost table as numpy array
    """
    # Handle dimension sizes
    if isinstance(domain, int):
        shape = tuple([domain] * connections)  # All dimensions have the same size
    else:
        raise ValueError("d must be an int")

    if callable(policy):
        # Use custom function to generate values
        return policy(**policy_params, size=shape)
    else:
        raise ValueError(f"Unknown policy: {policy}")


#####-------create_cost_table implementations --------#######
# functions that implement create_cost_table with different policies:
def create_random_int_table(n: int, domain: int, low=0, high=10) -> CostTable:
    """
    Create a random cost table with shape (domain, domain).

    Args:
        domain: Domain size

    Returns:
        Random cost table as numpy array
    """
    return _create_cost_table(n, domain, np.random.randint, low=low, high=high)


def create_uniform_table(n: int, domain: int, low=0, high=1) -> CostTable:
    """
    Create a uniform cost table with shape (domain, domain).

    Args:
        domain: Domain size

    Returns:
        Uniform cost table as numpy array
        :param n: number of connections
        :param domain: domain size
        :param high:
        :param low:
    """
    return _create_cost_table(n, domain, np.random.uniform, low=low, high=high)


def create_normal_table(n: int, domain: int, loc=0, scale=1) -> CostTable:
    """
    Create a normal cost table with shape (domain, domain).

    Args:
        domain: Domain size

    Returns:
        Normal cost table as numpy array
        :param domain: dimenstion size (conections)
        :param loc: mean
        :param scale: SD
    """
    return _create_cost_table(n, domain, np.random.normal, loc=loc, scale=scale)


def create_exponential_table(n: int, domain: int, scale=1) -> CostTable:
    """
    Create an exponential cost table with shape (domain, domain).

    Args:
        domain: Domain size

    Returns:
        Exponential cost table as numpy array
        :param scale: exponential parameter
    """
    return _create_cost_table(n, domain, np.random.exponential, scale=scale)


def create_symmetric_cost_table(n: int, m: int) -> CostTable:
    """
    Create a symmetric cost table of size n x m.
    """
    cost_table = np.random.rand(n, m)
    return (cost_table + cost_table.T) / 2


# example for noramlizing cost table for 3*3 ndarray


def normalize_cost_table(cost_table: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Convert integer cost table into a normalized distribution via log-domain softmin.

    Args:
        cost_table (np.ndarray): Raw cost table (e.g., integers or floats).
        axis (int, optional): Axis along which to normalize (e.g., 1 to normalize rows).

    Returns:
        np.ndarray: Normalized distribution (softmin over cost values).
    """
    # Convert cost to log-domain potentials (lower cost → higher potential)
    log_potentials = -cost_table.astype(float)

    # Compute log-normalizer (logZ) over the desired axis
    logZ = logsumexp(log_potentials, axis=axis, keepdims=True)

    # Compute normalized log-probabilities
    log_probs = log_potentials - logZ

    # Convert back from log-domain to probabilities
    probs = np.exp(log_probs)

    return probs


"""
# CostTable Creator Module Documentation

This module provides various functions to create cost tables for belief propagation algorithms.

## Functions

### create_cost_table
    Creates a random cost table with specified dimensions.

    Parameters:
        connections (int): Number of connected variables to factor
        domain (int): Domain size for each dimension
        policy (Callable): Function for generating values (default: np.random.randint)
        **policy_params: Additional parameters for the policy function

    Returns:
        np.ndarray: n-dimensional cost table

### create_random_table
    Creates a random integer cost table with shape (domain, domain).
    Values are integers from 0 to 9.

    Parameters:
        domain (int): Domain size

    Returns:
        np.ndarray: Random cost table

### create_uniform_table
    Creates a uniform cost table with shape (domain, domain).
    Values are floats from 0 to 1.

    Parameters:
        domain (int): Domain size

    Returns:
        np.ndarray: Uniform cost table

### create_normal_table
    Creates a normal distribution cost table with shape (domain, domain).

    Parameters:
        domain (int): Domain size
        loc (float): Mean of the distribution (default: 0)
        scale (float): Standard deviation (default: 1)

    Returns:
        np.ndarray: Normal cost table

### create_exponential_table
    Creates an exponential cost table with shape (domain, domain).

    Parameters:
        domain (int): Domain size
        scale (float): Scale parameter for exponential distribution (default: 1)

    Returns:
        np.ndarray: Exponential cost table

### create_symmetric_cost_table
    Creates a symmetric cost table of size n x m where cost(i,j) = cost(j,i).

    Parameters:
        n (int): First dimension size
        m (int): Second dimension size

    Returns:
        np.ndarray: Symmetric cost table

### normalize_cost_table
    Normalizes the cost table so the sum of all dimensions is equal.
    Note: Current implementation needs review.

    Parameters:
        cost_table (np.ndarray): n-dimensional cost table

    Returns:
        np.ndarray: Normalized cost table

## Usage Examples

# Create a 3x3 random cost table with integers from 0 to 9
random_table = create_random_table(3)

# Create a 4x4 uniform cost table with values from 0 to 1
uniform_table = create_uniform_table(4)

# Create a 5x5 normal distribution cost table with mean 0 and SD 1
normal_table = create_normal_table(5)

# Create a 3x3 exponential cost table with scale parameter 2
exp_table = create_exponential_table(3, scale=2)

# Create a 4x4 symmetric cost table
sym_table = create_symmetric_cost_table(4, 4)
"""
