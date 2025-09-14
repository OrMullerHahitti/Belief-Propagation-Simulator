"""
Margin computation and tie detection for belief propagation analysis.

Computes per-edge margins (best vs second-best objective) and detects ties
to validate fixed-region analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .snapshot import Snapshot
from .utils import product_labels
from .config import AnalysisConfig


def compute_edge_margins(
    snap: Snapshot, config: Optional[AnalysisConfig] = None
) -> Dict[str, Any]:
    """
    Compute per-edge margin per Lemma (best vs second-best objective).

    For each factor-variable edge, computes the margin between the best and
    second-best objective values across all possible assignments to other variables.

    Args:
        snap: Snapshot containing factor graph state
        config: Analysis configuration (optional)

    Returns:
        Dictionary with margin summary and tie information
    """
    if config is None:
        config = AnalysisConfig()

    ties = []
    margins = {}

    for factor_name in snap.N_fac:
        connected_vars = snap.N_fac[factor_name]
        cost_func = snap.cost.get(factor_name, lambda assign: 0.0)

        for target_var in connected_vars:
            other_vars = [v for v in connected_vars if v != target_var]

            # Collect all objective values for this edge
            all_objectives = []

            for target_label in snap.dom[target_var]:
                # For each assignment to other variables, compute objective
                for assignment in product_labels(snap, other_vars):
                    full_assignment = {**assignment, target_var: target_label}

                    try:
                        # Factor cost
                        objective = cost_func(full_assignment)

                        # Add Q message costs
                        for other_var in other_vars:
                            q_key = (other_var, factor_name)
                            if q_key in snap.Q:
                                q_msg = snap.Q[q_key]
                                label_idx = snap.dom[other_var].index(
                                    assignment[other_var]
                                )
                                if 0 <= label_idx < len(q_msg):
                                    objective += q_msg[label_idx]

                        all_objectives.append((objective, target_label, assignment))

                    except (KeyError, ValueError, IndexError):
                        # Skip invalid assignments
                        continue

            if len(all_objectives) < 2:
                # Need at least 2 values to compute margin
                margins[(factor_name, target_var)] = {
                    "margin": float("inf"),
                    "degree": len(connected_vars),
                    "num_assignments": len(all_objectives),
                }
                continue

            # Sort by objective value
            all_objectives.sort(key=lambda x: x[0])

            best_value = all_objectives[0][0]
            second_best_value = all_objectives[1][0]

            # Check for ties at minimum
            min_ties = []
            for obj_val, target_label, assignment in all_objectives:
                if abs(obj_val - best_value) <= config.tie_tolerance:
                    min_ties.append((factor_name, target_var, target_label))
                else:
                    break  # Since sorted, no more ties

            if len(min_ties) > 1:
                ties.extend(min_ties)

            # Compute margin
            margin = second_best_value - best_value
            margins[(factor_name, target_var)] = {
                "margin": margin,
                "degree": len(connected_vars),
                "num_assignments": len(all_objectives),
                "best_value": best_value,
                "second_best_value": second_best_value,
            }

    # Compute summary statistics
    margin_values = [
        m["margin"] for m in margins.values() if m["margin"] != float("inf")
    ]
    min_margin = min(margin_values) if margin_values else None

    summary = {
        "min_margin": min_margin,
        "num_ties": len(ties),
        "total_edges": len(margins),
        "infinite_margins": sum(
            1 for m in margins.values() if m["margin"] == float("inf")
        ),
    }

    if margin_values:
        summary.update(
            {
                "max_margin": max(margin_values),
                "mean_margin": sum(margin_values) / len(margin_values),
                "median_margin": np.median(margin_values),
            }
        )

    return {"summary": summary, "margins": margins, "ties": ties}


def detect_q_message_ties(
    snap: Snapshot, config: Optional[AnalysisConfig] = None
) -> List[Tuple[str, str]]:
    """
    Detect ties in Q messages (multiple labels with same minimum value).

    Args:
        snap: Snapshot containing Q messages
        config: Analysis configuration

    Returns:
        List of (variable, factor) pairs with Q message ties
    """
    if config is None:
        config = AnalysisConfig()

    ties = []

    for (var_name, factor_name), q_msg in snap.Q.items():
        if len(q_msg) <= 1:
            continue

        min_value = np.min(q_msg)

        # Count values equal to minimum
        tie_count = np.sum(np.abs(q_msg - min_value) <= config.tie_tolerance)

        if tie_count > 1:
            ties.append((var_name, factor_name))

    return ties


def detect_factor_ties(
    snap: Snapshot, config: Optional[AnalysisConfig] = None
) -> List[Tuple[str, str, str]]:
    """
    Detect ties in factor cost functions.

    For each factor and target variable, check if multiple assignments
    to other variables achieve the same minimum cost.

    Args:
        snap: Snapshot
        config: Analysis configuration

    Returns:
        List of (factor, variable, label) tuples with ties
    """
    if config is None:
        config = AnalysisConfig()

    ties = []

    for factor_name in snap.N_fac:
        connected_vars = snap.N_fac[factor_name]
        cost_func = snap.cost.get(factor_name, lambda assign: 0.0)

        for target_var in connected_vars:
            other_vars = [v for v in connected_vars if v != target_var]

            for target_label in snap.dom[target_var]:
                costs = []

                # Compute costs for all assignments to other variables
                for assignment in product_labels(snap, other_vars):
                    full_assignment = {**assignment, target_var: target_label}

                    try:
                        cost = cost_func(full_assignment)
                        costs.append(cost)
                    except (KeyError, ValueError, IndexError):
                        continue

                if len(costs) <= 1:
                    continue

                costs = np.array(costs)
                min_cost = np.min(costs)

                # Count costs equal to minimum
                tie_count = np.sum(np.abs(costs - min_cost) <= config.tie_tolerance)

                if tie_count > 1:
                    ties.append((factor_name, target_var, target_label))

    return ties


def validate_margin_computation(
    snap: Snapshot, margins: Dict[Tuple[str, str], Dict[str, Any]]
) -> List[str]:
    """
    Validate margin computation results.

    Args:
        snap: Snapshot
        margins: Computed margins

    Returns:
        List of validation errors
    """
    errors = []

    for (factor_name, var_name), margin_info in margins.items():
        # Check edge exists in snapshot
        if factor_name not in snap.N_fac:
            errors.append(f"Margin factor {factor_name} not in snapshot")
            continue

        if var_name not in snap.N_fac[factor_name]:
            errors.append(
                f"Margin variable {var_name} not connected to factor {factor_name}"
            )
            continue

        # Check margin properties
        margin = margin_info.get("margin")
        if margin is None:
            errors.append(f"Margin for ({factor_name}, {var_name}) is None")
            continue

        if margin < 0:
            errors.append(
                f"Margin for ({factor_name}, {var_name}) is negative: {margin}"
            )

        # Check degree consistency
        degree = margin_info.get("degree", 0)
        expected_degree = len(snap.N_fac[factor_name])
        if degree != expected_degree:
            errors.append(
                f"Margin degree {degree} != expected {expected_degree} for ({factor_name}, {var_name})"
            )

        # Check value consistency
        if "best_value" in margin_info and "second_best_value" in margin_info:
            best = margin_info["best_value"]
            second = margin_info["second_best_value"]
            computed_margin = second - best

            if abs(computed_margin - margin) > 1e-10:
                errors.append(
                    f"Margin inconsistency for ({factor_name}, {var_name}): {margin} != {computed_margin}"
                )

    return errors


def compute_margin_statistics(
    margins: Dict[Tuple[str, str], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute statistics about margin distribution.

    Args:
        margins: Margin computation results

    Returns:
        Dictionary with margin statistics
    """
    if not margins:
        return {"total_margins": 0}

    margin_values = []
    degrees = []
    assignment_counts = []

    for margin_info in margins.values():
        margin = margin_info.get("margin")
        if margin is not None and margin != float("inf"):
            margin_values.append(margin)

        degree = margin_info.get("degree", 0)
        degrees.append(degree)

        num_assignments = margin_info.get("num_assignments", 0)
        assignment_counts.append(num_assignments)

    stats = {
        "total_margins": len(margins),
        "finite_margins": len(margin_values),
        "infinite_margins": len(margins) - len(margin_values),
    }

    if margin_values:
        margin_array = np.array(margin_values)
        stats.update(
            {
                "min_margin": float(np.min(margin_array)),
                "max_margin": float(np.max(margin_array)),
                "mean_margin": float(np.mean(margin_array)),
                "std_margin": float(np.std(margin_array)),
                "median_margin": float(np.median(margin_array)),
                "zero_margins": int(np.sum(margin_array <= 1e-10)),
            }
        )

    if degrees:
        degree_array = np.array(degrees)
        stats.update(
            {
                "min_degree": int(np.min(degree_array)),
                "max_degree": int(np.max(degree_array)),
                "mean_degree": float(np.mean(degree_array)),
            }
        )

    if assignment_counts:
        assignment_array = np.array(assignment_counts)
        stats.update(
            {
                "min_assignments": int(np.min(assignment_array)),
                "max_assignments": int(np.max(assignment_array)),
                "mean_assignments": float(np.mean(assignment_array)),
            }
        )

    return stats


def analyze_tie_patterns(ties: List[Tuple], tie_type: str = "margin") -> Dict[str, Any]:
    """
    Analyze patterns in detected ties.

    Args:
        ties: List of tie tuples
        tie_type: Type of ties ("margin", "q_message", "factor")

    Returns:
        Dictionary with tie pattern analysis
    """
    if not ties:
        return {"total_ties": 0, "tie_type": tie_type}

    analysis = {"total_ties": len(ties), "tie_type": tie_type}

    if tie_type == "margin":
        # Ties are (factor, variable, label) tuples
        factors = [t[0] for t in ties]
        variables = [t[1] for t in ties]

        analysis.update(
            {
                "unique_factors": len(set(factors)),
                "unique_variables": len(set(variables)),
                "most_tied_factor": max(set(factors), key=factors.count)
                if factors
                else None,
                "most_tied_variable": max(set(variables), key=variables.count)
                if variables
                else None,
            }
        )

    elif tie_type == "q_message":
        # Ties are (variable, factor) tuples
        factors = [t[1] for t in ties]
        variables = [t[0] for t in ties]

        analysis.update(
            {
                "unique_factors": len(set(factors)),
                "unique_variables": len(set(variables)),
            }
        )

    elif tie_type == "factor":
        # Ties are (factor, variable, label) tuples
        factors = [t[0] for t in ties]
        variables = [t[1] for t in ties]
        labels = [t[2] for t in ties]

        analysis.update(
            {
                "unique_factors": len(set(factors)),
                "unique_variables": len(set(variables)),
                "unique_labels": len(set(labels)),
            }
        )

    return analysis


def check_margin_convergence_conditions(
    margins: Dict[Tuple[str, str], Dict[str, Any]],
    ties: List[Tuple],
    config: Optional[AnalysisConfig] = None,
) -> Dict[str, Any]:
    """
    Check conditions for margin-based convergence analysis.

    Args:
        margins: Computed margins
        ties: Detected ties
        config: Analysis configuration

    Returns:
        Dictionary with convergence condition analysis
    """
    if config is None:
        config = AnalysisConfig()

    # Extract finite margin values
    margin_values = [
        m["margin"] for m in margins.values() if m["margin"] != float("inf")
    ]

    conditions = {
        "has_finite_margins": len(margin_values) > 0,
        "has_ties": len(ties) > 0,
        "min_margin": min(margin_values) if margin_values else None,
        "ties_treated_as_error": config.treat_ties_as_error,
    }

    # Check strict positivity
    if margin_values:
        min_margin = min(margin_values)
        conditions.update(
            {
                "all_margins_positive": min_margin > config.tie_tolerance,
                "margin_threshold_met": min_margin > config.abs_tol,
                "smallest_margin": min_margin,
            }
        )

    # Overall assessment
    conditions["convergence_feasible"] = (
        conditions.get("has_finite_margins", False)
        and not conditions.get("has_ties", True)
        and conditions.get("all_margins_positive", False)
    )

    return conditions


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot
    import numpy as np

    print("=== Margin Analysis Examples ===")

    # Create test snapshot with margin structure
    test_snapshot = SimpleSnapshot(
        _lambda=0.5,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"]},
        _N_var={"x1": ["f1"], "x2": ["f1"]},
        _N_fac={"f1": ["x1", "x2"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.2]),  # Clear minimum at 0
            ("x2", "f1"): np.array([0.1, 0.1]),  # Tie between 0 and 1
        },
        _R={("f1", "x1"): np.array([0.1, 0.3]), ("f1", "x2"): np.array([0.2, 0.0])},
        _unary={"x1": np.zeros(2), "x2": np.zeros(2)},
        _cost={
            "f1": lambda assign: 0.5 if assign.get("x1") == assign.get("x2") else 0.0
        },
        _split_map={},
    )

    # Compute edge margins
    margin_results = compute_edge_margins(test_snapshot)

    print(f"Computed margins for {len(margin_results['margins'])} edges")
    print(f"Summary: {margin_results['summary']}")
    print(f"Found {len(margin_results['ties'])} ties")

    # Check Q message ties
    q_ties = detect_q_message_ties(test_snapshot)
    print(f"Q message ties: {q_ties}")

    # Check factor ties
    factor_ties = detect_factor_ties(test_snapshot)
    print(f"Factor ties: {len(factor_ties)}")

    # Validate margins
    errors = validate_margin_computation(test_snapshot, margin_results["margins"])
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Margin computation is valid")

    # Get statistics
    stats = compute_margin_statistics(margin_results["margins"])
    print(f"Margin statistics: {stats}")

    # Analyze tie patterns
    tie_analysis = analyze_tie_patterns(margin_results["ties"], "margin")
    print(f"Tie patterns: {tie_analysis}")

    # Check convergence conditions
    convergence_conditions = check_margin_convergence_conditions(
        margin_results["margins"], margin_results["ties"]
    )
    print(f"Convergence conditions: {convergence_conditions}")

    print("\n=== Ready for region detection ===")
