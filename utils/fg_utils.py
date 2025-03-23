import numpy as np

def create_ndarray(domain: int, num_connected: int) -> np.ndarray:
    """
    Create an ndarray with 'num_connected' dimensions,
    each dimension of size 'domain'.

    :param domain: The size of each dimension in the ndarray.
    :param num_connected: The number of dimensions in the ndarray.
    :return: An ndarray of shape (domain, domain, ..., domain)
             (num_connected times).
    """
    shape = (domain,) * num_connected  # e.g., (3,3) for domain=3, num_connected=2
    # Populate the array with random integers (0..9) for demonstration:
    return np.random.randint(0, 10, size=shape)