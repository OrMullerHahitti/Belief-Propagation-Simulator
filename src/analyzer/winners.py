"""
Winner and min_idx computation for belief propagation analysis.

Computes exact argmin per (fcopy→v,a) using brute force for small factor domains,
and min indices for Q messages.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .snapshot import Snapshot
from .utils import product_labels, safe_argmin
from .config import AnalysisConfig


def compute_winners(
    snap: Snapshot, config: Optional[AnalysisConfig] = None
) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """
    Compute exact argmin per (fcopy→v,a) using brute force on small ∂f.

    For each factor f, variable v in ∂f, and label a ∈ X_v, find the assignment
    to other variables that minimizes:
        cost_f(x_1,...,x_k) + Σ_{u≠v} Q_{u→f}(x_u)

    Args:
        snap: Snapshot containing factor graph state
        config: Analysis configuration (optional)

    Returns:
        Dictionary mapping (fcopy, v, a) -> {u: best_label} for u ≠ v

    Raises:
        ValueError: If factor domains are too large for brute force
    """
    if config is None:
        config = AnalysisConfig()

    winners = {}

    for factor_name in snap.N_fac:
        connected_vars = snap.N_fac[factor_name]

        # Check domain size for brute force feasibility
        total_combinations = 1
        for var_name in connected_vars:
            total_combinations *= len(snap.dom[var_name])

        if total_combinations > config.max_brute_force_domain_size ** len(
            connected_vars
        ):
            # Skip factors that are too large for brute force
            continue

        # Get cost function for this factor
        cost_func = snap.cost.get(factor_name)
        if cost_func is None:
            # Use zero cost if no function provided
            cost_func = lambda assignment: 0.0

        # For each variable connected to this factor
        for target_var in connected_vars:
            other_vars = [v for v in connected_vars if v != target_var]

            # For each label of the target variable
            for target_label in snap.dom[target_var]:
                best_value = float("+inf")
                best_assignment = {}

                # Try all possible assignments to other variables
                for assignment in product_labels(snap, other_vars):
                    # Compute total cost for this assignment
                    full_assignment = {**assignment, target_var: target_label}

                    try:
                        # Factor cost
                        factor_cost = cost_func(full_assignment)

                        # Q message costs from other variables
                        message_cost = 0.0
                        for other_var in other_vars:
                            q_key = (other_var, factor_name)
                            if q_key in snap.Q:
                                q_msg = snap.Q[q_key]
                                # Convert label to index
                                label_idx = snap.dom[other_var].index(
                                    assignment[other_var]
                                )
                                if 0 <= label_idx < len(q_msg):
                                    message_cost += q_msg[label_idx]

                        total_cost = factor_cost + message_cost

                        # Update best if this is better
                        if total_cost < best_value:
                            best_value = total_cost
                            best_assignment = assignment.copy()

                    except (KeyError, ValueError, IndexError):
                        # Skip invalid assignments
                        continue

                # Store the winning assignment
                winners_key = (factor_name, target_var, target_label)
                winners[winners_key] = best_assignment

    return winners


def compute_min_idx(snap: Snapshot) -> Dict[Tuple[str, str], int]:
    """
    Compute argmin indices for Q messages.

    For each Q message Q_{u→f}, find the label index that minimizes the message value.
    Assumes deterministic tie-breaking (uses first minimum).

    Args:
        snap: Snapshot containing Q messages

    Returns:
        Dictionary mapping (u, f) -> argmin_label_index
    """
    min_idx = {}

    for q_key, q_msg in snap.Q.items():
        if len(q_key) == 2:  # (u, f) format
            var_name, factor_name = q_key
            min_index = safe_argmin(q_msg)
            min_idx[(var_name, factor_name)] = min_index

    return min_idx


def validate_winners(
    snap: Snapshot, winners: Dict[Tuple[str, str, str], Dict[str, str]]
) -> List[str]:
    """
    Validate computed winners against snapshot structure.

    Args:
        snap: Snapshot
        winners: Computed winners

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for (factor_name, target_var, target_label), assignment in winners.items():
        # Check factor exists
        if factor_name not in snap.N_fac:
            errors.append(f"Winner factor {factor_name} not in snapshot")
            continue

        # Check target variable is connected to factor
        connected_vars = snap.N_fac[factor_name]
        if target_var not in connected_vars:
            errors.append(
                f"Winner target variable {target_var} not connected to factor {factor_name}"
            )
            continue

        # Check target label is valid
        if target_label not in snap.dom.get(target_var, []):
            errors.append(
                f"Winner target label {target_label} invalid for variable {target_var}"
            )
            continue

        # Check assignment variables are correct (other variables connected to factor)
        other_vars = [v for v in connected_vars if v != target_var]
        assignment_vars = set(assignment.keys())
        expected_vars = set(other_vars)

        if assignment_vars != expected_vars:
            errors.append(
                f"Winner assignment variables {assignment_vars} don't match expected {expected_vars}"
            )
            continue

        # Check assignment labels are valid
        for var_name, label in assignment.items():
            if label not in snap.dom.get(var_name, []):
                errors.append(
                    f"Winner assignment label {label} invalid for variable {var_name}"
                )

    return errors


def validate_min_idx(snap: Snapshot, min_idx: Dict[Tuple[str, str], int]) -> List[str]:
    """
    Validate computed min indices against snapshot structure.

    Args:
        snap: Snapshot
        min_idx: Computed min indices

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for (var_name, factor_name), idx in min_idx.items():
        # Check Q message exists
        q_key = (var_name, factor_name)
        if q_key not in snap.Q:
            errors.append(f"Min idx Q message {q_key} not in snapshot")
            continue

        q_msg = snap.Q[q_key]

        # Check index is valid
        if not (0 <= idx < len(q_msg)):
            errors.append(
                f"Min idx {idx} out of range for Q message {q_key} (length {len(q_msg)})"
            )
            continue

        # Check index actually corresponds to minimum (within tolerance)
        actual_min_idx = safe_argmin(q_msg)
        if abs(q_msg[idx] - q_msg[actual_min_idx]) > 1e-10:
            min_val = q_msg[actual_min_idx]
            idx_val = q_msg[idx]
            errors.append(
                f"Min idx {idx} for Q{q_key} has value {idx_val}, but minimum is {min_val} at index {actual_min_idx}"
            )

    return errors


def detect_winner_ties(
    snap: Snapshot,
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    tol: float = 1e-10,
) -> List[Tuple[str, str, str]]:
    """
    Detect ties in winner computation.

    A tie occurs when multiple assignments achieve the same minimum cost.

    Args:
        snap: Snapshot
        winners: Computed winners
        tol: Tolerance for considering costs equal

    Returns:
        List of (factor, variable, label) tuples with ties
    """
    ties = []

    for (factor_name, target_var, target_label), best_assignment in winners.items():
        if factor_name not in snap.N_fac:
            continue

        connected_vars = snap.N_fac[factor_name]
        other_vars = [v for v in connected_vars if v != target_var]
        cost_func = snap.cost.get(factor_name, lambda x: 0.0)

        # Compute the winning cost
        full_assignment = {**best_assignment, target_var: target_label}
        try:
            best_cost = cost_func(full_assignment)
            for other_var in other_vars:
                q_key = (other_var, factor_name)
                if q_key in snap.Q:
                    q_msg = snap.Q[q_key]
                    label_idx = snap.dom[other_var].index(best_assignment[other_var])
                    if 0 <= label_idx < len(q_msg):
                        best_cost += q_msg[label_idx]
        except (KeyError, ValueError, IndexError):
            continue

        # Check if any other assignment achieves the same cost
        tie_count = 0
        for assignment in product_labels(snap, other_vars):
            full_assignment = {**assignment, target_var: target_label}
            try:
                cost = cost_func(full_assignment)
                for other_var in other_vars:
                    q_key = (other_var, factor_name)
                    if q_key in snap.Q:
                        q_msg = snap.Q[q_key]
                        label_idx = snap.dom[other_var].index(assignment[other_var])
                        if 0 <= label_idx < len(q_msg):
                            cost += q_msg[label_idx]

                if abs(cost - best_cost) <= tol:
                    tie_count += 1
                    if tie_count > 1:  # Found a tie
                        ties.append((factor_name, target_var, target_label))
                        break

            except (KeyError, ValueError, IndexError):
                continue

    return ties


def detect_min_idx_ties(
    snap: Snapshot, min_idx: Dict[Tuple[str, str], int], tol: float = 1e-10
) -> List[Tuple[str, str]]:
    """
    Detect ties in min_idx computation.

    A tie occurs when multiple labels have the same minimum Q message value.

    Args:
        snap: Snapshot
        min_idx: Computed min indices
        tol: Tolerance for considering values equal

    Returns:
        List of (variable, factor) tuples with ties
    """
    ties = []

    for (var_name, factor_name), idx in min_idx.items():
        q_key = (var_name, factor_name)
        if q_key not in snap.Q:
            continue

        q_msg = snap.Q[q_key]
        if len(q_msg) == 0:
            continue

        min_value = q_msg[idx]

        # Count how many values are equal to minimum
        tie_count = sum(1 for val in q_msg if abs(val - min_value) <= tol)

        if tie_count > 1:
            ties.append((var_name, factor_name))

    return ties


def compute_winners_stats(
    winners: Dict[Tuple[str, str, str], Dict[str, str]]
) -> Dict[str, Any]:
    """
    Compute statistics about winner computation.

    Args:
        winners: Computed winners

    Returns:
        Dictionary with statistics
    """
    if not winners:
        return {"total_winners": 0}

    # Group by factor
    factors = set(factor_name for factor_name, _, _ in winners.keys())
    variables = set(var_name for _, var_name, _ in winners.keys())

    # Assignment size statistics
    assignment_sizes = [len(assignment) for assignment in winners.values()]

    return {
        "total_winners": len(winners),
        "unique_factors": len(factors),
        "unique_variables": len(variables),
        "assignment_sizes": {
            "min": min(assignment_sizes) if assignment_sizes else 0,
            "max": max(assignment_sizes) if assignment_sizes else 0,
            "mean": sum(assignment_sizes) / len(assignment_sizes)
            if assignment_sizes
            else 0,
        },
        "avg_winners_per_factor": len(winners) / len(factors) if factors else 0,
    }


def compute_min_idx_stats(min_idx: Dict[Tuple[str, str], int]) -> Dict[str, Any]:
    """
    Compute statistics about min_idx computation.

    Args:
        min_idx: Computed min indices

    Returns:
        Dictionary with statistics
    """
    if not min_idx:
        return {"total_min_indices": 0}

    # Index distribution
    indices = list(min_idx.values())
    variables = set(var_name for var_name, _ in min_idx.keys())
    factors = set(factor_name for _, factor_name in min_idx.keys())

    return {
        "total_min_indices": len(min_idx),
        "unique_variables": len(variables),
        "unique_factors": len(factors),
        "index_distribution": {
            "min": min(indices) if indices else 0,
            "max": max(indices) if indices else 0,
            "mean": sum(indices) / len(indices) if indices else 0,
        },
    }


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot
    import numpy as np

    print("=== Winners and Min Idx Examples ===")

    # Create test snapshot with simple factor graph
    test_snapshot = SimpleSnapshot(
        _lambda=0.5,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"]},
        _N_var={"x1": ["f1"], "x2": ["f1"]},
        _N_fac={"f1": ["x1", "x2"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.5]),  # min at index 0
            ("x2", "f1"): np.array([0.2, 0.0]),  # min at index 1
        },
        _R={("f1", "x1"): np.array([0.1, 0.2]), ("f1", "x2"): np.array([0.3, 0.0])},
        _unary={"x1": np.zeros(2), "x2": np.zeros(2)},
        _cost={
            "f1": lambda assign: 0.5 if assign.get("x1") == assign.get("x2") else 0.0
        },
        _split_map={},
    )

    # Compute winners
    winners = compute_winners(test_snapshot)
    print(f"Computed {len(winners)} winners:")
    for key, assignment in winners.items():
        print(f"  {key} -> {assignment}")

    # Compute min indices
    min_idx = compute_min_idx(test_snapshot)
    print(f"Computed {len(min_idx)} min indices:")
    for key, idx in min_idx.items():
        q_msg = test_snapshot.Q[key]
        print(f"  {key} -> index {idx} (value {q_msg[idx]:.3f})")

    # Validate results
    winner_errors = validate_winners(test_snapshot, winners)
    min_idx_errors = validate_min_idx(test_snapshot, min_idx)

    if winner_errors:
        print(f"Winner validation errors: {winner_errors}")
    else:
        print("✓ Winners are valid")

    if min_idx_errors:
        print(f"Min idx validation errors: {min_idx_errors}")
    else:
        print("✓ Min indices are valid")

    # Check for ties
    winner_ties = detect_winner_ties(test_snapshot, winners)
    min_idx_ties = detect_min_idx_ties(test_snapshot, min_idx)

    print(f"Winner ties: {len(winner_ties)}")
    print(f"Min idx ties: {len(min_idx_ties)}")

    # Get statistics
    winner_stats = compute_winners_stats(winners)
    min_idx_stats = compute_min_idx_stats(min_idx)
    print(f"Winner stats: {winner_stats}")
    print(f"Min idx stats: {min_idx_stats}")

    print("\n=== Ready for cycle analysis ===")
