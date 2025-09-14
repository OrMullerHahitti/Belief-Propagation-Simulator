"""
Utility functions for the analysis framework.

Provides common helper functions used across multiple modules.
"""

import numpy as np
from typing import Dict, List, Tuple, Iterator, Any, Optional
from itertools import product
from .snapshot import Snapshot


def product_labels(snap: Snapshot, var_names: List[str]) -> Iterator[Dict[str, str]]:
    """
    Generate all possible label assignments for given variables.

    Args:
        snap: Snapshot containing domain information
        var_names: List of variable names to generate assignments for

    Yields:
        Dict mapping variable name to label
    """
    if not var_names:
        yield {}
        return

    domains = [snap.dom[var_name] for var_name in var_names]
    for assignment_tuple in product(*domains):
        yield dict(zip(var_names, assignment_tuple))


def unique_edges_from_idxQ(idxQ: Dict[tuple, int]) -> Iterator[Tuple[str, str]]:
    """
    Extract unique (u,f) edges from Q slot indices.

    Args:
        idxQ: Q slot indices mapping (u,f,a) -> index

    Yields:
        (u,f) tuples for unique edges
    """
    seen_edges = set()
    for u, f, a in idxQ.keys():
        edge = (u, f)
        if edge not in seen_edges:
            seen_edges.add(edge)
            yield edge


def dim_of_var(snap: Snapshot, var_name: str) -> int:
    """
    Get domain size of a variable.

    Args:
        snap: Snapshot containing domain information
        var_name: Variable name

    Returns:
        Domain size
    """
    return len(snap.dom[var_name])


def is_close(a: float, b: float, abs_tol: float = 1e-10, rel_tol: float = 1e-9) -> bool:
    """
    Check if two floats are close within tolerances.

    Args:
        a, b: Values to compare
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance

    Returns:
        True if values are close
    """
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def safe_min(arr: np.ndarray, default: float = 0.0) -> float:
    """
    Safely compute minimum of array, returning default if empty.

    Args:
        arr: Input array
        default: Default value for empty arrays

    Returns:
        Minimum value or default
    """
    if len(arr) == 0:
        return default
    return float(np.min(arr))


def safe_max(arr: np.ndarray, default: float = 0.0) -> float:
    """
    Safely compute maximum of array, returning default if empty.

    Args:
        arr: Input array
        default: Default value for empty arrays

    Returns:
        Maximum value or default
    """
    if len(arr) == 0:
        return default
    return float(np.max(arr))


def safe_argmin(arr: np.ndarray, default: int = 0) -> int:
    """
    Safely compute argmin of array, returning default if empty.

    Args:
        arr: Input array
        default: Default index for empty arrays

    Returns:
        Index of minimum value or default
    """
    if len(arr) == 0:
        return default
    return int(np.argmin(arr))


def safe_argmax(arr: np.ndarray, default: int = 0) -> int:
    """
    Safely compute argmax of array, returning default if empty.

    Args:
        arr: Input array
        default: Default index for empty arrays

    Returns:
        Index of maximum value or default
    """
    if len(arr) == 0:
        return default
    return int(np.argmax(arr))


def normalize_to_min_zero(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array so minimum value is zero.

    Args:
        arr: Input array

    Returns:
        Normalized array
    """
    if len(arr) == 0:
        return arr.copy()
    return arr - np.min(arr)


def check_min_normalized(arr: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if array is min-normalized (minimum value is zero).

    Args:
        arr: Array to check
        tol: Tolerance for zero check

    Returns:
        True if min-normalized
    """
    if len(arr) == 0:
        return True
    return abs(np.min(arr)) <= tol


def format_key(key: Any) -> str:
    """
    Format a key (typically tuple) for display.

    Args:
        key: Key to format

    Returns:
        Formatted string
    """
    if isinstance(key, tuple):
        return f"({','.join(str(k) for k in key)})"
    return str(key)


def validate_probability_distribution(arr: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if array represents a valid probability distribution.

    Args:
        arr: Array to check
        tol: Tolerance for sum check

    Returns:
        True if valid probability distribution
    """
    if len(arr) == 0:
        return False

    # Check non-negative
    if np.any(arr < 0):
        return False

    # Check sums to 1
    total = np.sum(arr)
    return abs(total - 1.0) <= tol


def create_uniform_distribution(size: int) -> np.ndarray:
    """
    Create uniform probability distribution.

    Args:
        size: Distribution size

    Returns:
        Uniform distribution array
    """
    if size <= 0:
        return np.array([])
    return np.ones(size) / size


def entropy(prob_dist: np.ndarray, base: float = 2.0) -> float:
    """
    Compute entropy of probability distribution.

    Args:
        prob_dist: Probability distribution
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy value
    """
    if len(prob_dist) == 0:
        return 0.0

    # Filter out zero probabilities
    nonzero_probs = prob_dist[prob_dist > 0]
    if len(nonzero_probs) == 0:
        return 0.0

    if base == 2.0:
        return -np.sum(nonzero_probs * np.log2(nonzero_probs))
    elif base == np.e:
        return -np.sum(nonzero_probs * np.log(nonzero_probs))
    else:
        return -np.sum(nonzero_probs * np.log(nonzero_probs) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence D(P||Q).

    Args:
        p, q: Probability distributions

    Returns:
        KL divergence value (infinity if q has zeros where p doesn't)
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    if len(p) == 0:
        return 0.0

    # Handle edge cases
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0

    # Check if q has zeros where p doesn't (infinite KL divergence)
    if np.any((p > 0) & (q == 0)):
        return float("inf")

    # Compute KL divergence only for non-zero entries
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute total variation distance between distributions.

    Args:
        p, q: Probability distributions

    Returns:
        Total variation distance in [0,1]
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    if len(p) == 0:
        return 0.0

    return 0.5 * np.sum(np.abs(p - q))


def hamming_distance(a: List[Any], b: List[Any]) -> int:
    """
    Compute Hamming distance between two sequences.

    Args:
        a, b: Sequences to compare

    Returns:
        Number of positions where sequences differ
    """
    if len(a) != len(b):
        raise ValueError("Sequences must have same length")

    return sum(x != y for x, y in zip(a, b))


def moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Compute moving average of values.

    Args:
        values: Input values
        window: Window size

    Returns:
        Moving averages
    """
    if window <= 0 or len(values) == 0:
        return values.copy()

    if window >= len(values):
        return [np.mean(values)] * len(values)

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        end = i + 1
        result.append(np.mean(values[start:end]))

    return result


def detect_convergence(values: List[float], window: int = 3, tol: float = 1e-6) -> bool:
    """
    Detect if a sequence of values has converged.

    Args:
        values: Sequence of values
        window: Window size for checking stability
        tol: Tolerance for considering values equal

    Returns:
        True if converged
    """
    if len(values) < window:
        return False

    recent_values = values[-window:]
    return max(recent_values) - min(recent_values) <= tol


def format_analysis_dict(analysis: Dict[str, Any], indent: int = 2) -> str:
    """
    Format analysis dictionary for pretty printing.

    Args:
        analysis: Analysis dictionary
        indent: Indentation level

    Returns:
        Formatted string
    """

    def format_value(val: Any, level: int = 0) -> str:
        spaces = " " * (indent * level)

        if isinstance(val, dict):
            if not val:
                return "{}"
            lines = ["{"]
            for k, v in val.items():
                formatted_val = format_value(v, level + 1)
                lines.append(f"{spaces}  {k}: {formatted_val}")
            lines.append(f"{spaces}}}")
            return "\n".join(lines)

        elif isinstance(val, list):
            if not val:
                return "[]"
            if len(val) <= 3:  # Short lists on one line
                return str(val)
            return f"[...{len(val)} items...]"

        elif isinstance(val, float):
            return f"{val:.6g}"

        else:
            return str(val)

    return format_value(analysis)


# Example usage and testing
if __name__ == "__main__":
    print("=== Utility Functions Examples ===")

    # Test probability functions
    uniform = create_uniform_distribution(4)
    print(f"Uniform distribution: {uniform}")
    print(f"Is valid probability: {validate_probability_distribution(uniform)}")
    print(f"Entropy (bits): {entropy(uniform):.3f}")

    # Test skewed distribution
    skewed = np.array([0.8, 0.1, 0.05, 0.05])
    print(f"Skewed distribution: {skewed}")
    print(f"Entropy (bits): {entropy(skewed):.3f}")
    print(f"KL(uniform||skewed): {kl_divergence(uniform, skewed):.3f}")
    print(f"TV distance: {total_variation_distance(uniform, skewed):.3f}")

    # Test convergence detection
    converging = [1.0, 0.5, 0.25, 0.125, 0.1, 0.09, 0.089]
    oscillating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    print(f"Converging sequence converged: {detect_convergence(converging)}")
    print(f"Oscillating sequence converged: {detect_convergence(oscillating)}")

    # Test utility functions
    arr = np.array([3.5, 1.2, 0.8, 2.1])
    print(f"Array: {arr}")
    print(f"Min-normalized: {normalize_to_min_zero(arr)}")
    print(f"Is min-normalized: {check_min_normalized(normalize_to_min_zero(arr))}")

    print("\n=== Ready for use in analysis framework ===")
