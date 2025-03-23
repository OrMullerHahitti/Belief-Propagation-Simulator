import numpy as np
from typing import Callable, Tuple

# Define a type alias for a randomness policy function.  This makes the code more readable.
RandomnessPolicy = Callable[[Tuple[int, ...]], np.ndarray]



# Define some example randomness policies

def uniform_random(shape: Tuple[int, ...]) -> np.ndarray:
    """Generates a uniform random ndarray with values between 0 and 1."""
    return np.random.rand(*shape)

def normal_random(shape: Tuple[int, ...],mean =0,std=1) -> np.ndarray:
    """Generates a ndarray with values from a standard normal distribution (mean 0, stddev 1)."""
    return np.random.normal(size=shape,loc=mean,scale=std)

def integer_random(shape: Tuple[int, ...], low: int = 0, high: int = 10) -> np.ndarray:
    """Generates an array of random integers within a specified range."""
    return np.random.randint(low, high, size=shape)


def create_random_message(domain_size:int,randomness_policy:RandomnessPolicy=integer_random)->np.ndarray:
    '''
    Creates a random ndarray of a given size using a dependency-injected randomness policy.
    :param domain_size:
    :param randomness_policy:
    :return:
    '''
    return randomness_policy((domain_size,))
def create_random_table(domain_size: Tuple[int, ...], randomness_policy: RandomnessPolicy=integer_random ) -> np.ndarray:
    """
    Creates a random ndarray of a given size using a dependency-injected randomness policy.

    Args:
        domain_size: A tuple specifying the shape of the ndarray.
        randomness_policy: A function that takes a tuple representing the shape
                           and returns a random ndarray.

    Returns:
        A random ndarray.
    """
    return randomness_policy(domain_size)

#example : create randome table 3x3 with integeres - 0 to 10
# create_random_table((3, 3), integer_random)

#example : create randome data 3 with uniform random

