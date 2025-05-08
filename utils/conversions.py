import numpy as np
import pandas as pd
from typing import Union, Dict, Hashable, Any


def to_ndarray(
    data: Union[np.ndarray, Dict[Hashable, Any], pd.DataFrame, list],
) -> np.ndarray:
    """
    Convert various data structures to a NumPy ndarray.

    Args:
        data: Input data in form of numpy array, dictionary, pandas DataFrame, or nested list

    Returns:
        np.ndarray: Converted data as NumPy array

    Raises:
        ValueError: If input data cannot be converted to ndarray
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, dict):
        return np.array(list(data.values()))
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise ValueError(f"Cannot convert type {type(data)} to ndarray")
