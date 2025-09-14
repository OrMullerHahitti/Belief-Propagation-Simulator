"""
Enforcement suggestions (epsilon) for belief propagation analysis.

Given cycles without an aligned hop, proposes specific hops and epsilon values
to tilt winners and achieve convergence certificates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .snapshot import Snapshot
from .utils import product_labels
from .config import AnalysisConfig


def suggest_enforcement(
    snap: Snapshot,
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    cycles_analysis: Dict[str, Any],
    config: Optional[AnalysisConfig] = None,
) -> Optional[Dict[str, Any]]:
    """
    Suggest epsilon enforcement to create aligned hops in uncertified cycles.

    Finds the shortest uncertified cycle (no aligned hop) and proposes
    one hop and the epsilon value needed for winner-tilt.

    Args:
        snap: Snapshot
        winners: Winner assignments
        min_idx: Minimum indices
        cycles_analysis: Cycle analysis results
        config: Analysis configuration

    Returns:
        Enforcement suggestion dictionary or None if not needed
    """
    if config is None:
        config = AnalysisConfig()

    # Check if enforcement is enabled
    if not config.include_enforcement_suggestions:
        return None

    # Check if there are uncertified cycles
    cycles_detail = cycles_analysis.get("detail", [])
    uncertified_cycles = [c for c in cycles_detail if not c.get("aligned", False)]

    if not uncertified_cycles:
        # All cycles are certified - no enforcement needed
        return None

    # Find shortest uncertified cycle
    shortest_cycle = min(
        uncertified_cycles, key=lambda c: c.get("length", float("inf"))
    )

    # Pick a hop from this cycle for enforcement
    enforcement_hop = pick_enforcement_hop(snap, winners, min_idx, shortest_cycle)

    if enforcement_hop is None:
        return None

    # Compute required epsilon
    epsilon = compute_required_epsilon(snap, enforcement_hop, winners, min_idx, config)

    return {
        "enforce_hop": enforcement_hop,
        "epsilon": epsilon,
        "cycle_length": shortest_cycle.get("length"),
        "reason": "shortest_uncertified_cycle",
        "confidence": "high" if epsilon < 1.0 else "medium",
    }


def pick_enforcement_hop(
    snap: Snapshot,
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    cycle_info: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """
    Pick a specific hop from a cycle for epsilon enforcement.

    Selects a (factor, variable, other_variable) triple where enforcement
    would be most effective.

    Args:
        snap: Snapshot
        winners: Winner assignments
        min_idx: Minimum indices
        cycle_info: Information about the cycle to enforce

    Returns:
        Enforcement hop specification or None if none found
    """
    # For simplicity, pick the first available Q->R hop that participates in BPA
    # In practice, this would extract from the actual cycle structure

    for factor_name in snap.N_fac:
        connected_vars = snap.N_fac[factor_name]

        if len(connected_vars) < 2:
            continue  # Need at least 2 variables for meaningful hop

        # Pick first variable as target, second as the enforced variable
        target_var = connected_vars[0]
        enforce_var = connected_vars[1]

        # Check if this hop has potential for alignment
        min_key = (enforce_var, factor_name)
        if min_key not in min_idx:
            continue

        min_label_idx = min_idx[min_key]

        # Check if winner disagrees with min for some target label
        for target_label in snap.dom[target_var]:
            winner_key = (factor_name, target_var, target_label)
            winner_assignment = winners.get(winner_key, {})

            if enforce_var in winner_assignment:
                winning_label = winner_assignment[enforce_var]
                try:
                    winning_label_idx = snap.dom[enforce_var].index(winning_label)
                except ValueError:
                    continue

                if winning_label_idx != min_label_idx:
                    # Found disagreement - this hop can be enforced
                    return {
                        "fcopy": factor_name,
                        "u": enforce_var,
                        "v": target_var,
                        "target_label": target_label,
                        "current_winner_label": winning_label,
                        "min_label_idx": min_label_idx,
                    }

    return None


def compute_required_epsilon(
    snap: Snapshot,
    enforcement_hop: Dict[str, str],
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    config: AnalysisConfig,
) -> float:
    """
    Compute epsilon value needed to tilt the winner towards alignment.

    Computes slack s_{f,u}(a) = best_unconstrained - best_constrained_u_eq_m
    and returns epsilon = margin_factor * max_slack.

    Args:
        snap: Snapshot
        enforcement_hop: Hop specification
        winners: Winner assignments
        min_idx: Minimum indices
        config: Analysis configuration

    Returns:
        Required epsilon value
    """
    factor_name = enforcement_hop["fcopy"]
    enforce_var = enforcement_hop["u"]
    target_var = enforcement_hop["v"]
    target_label = enforcement_hop["target_label"]

    # Get cost function and connected variables
    cost_func = snap.cost.get(factor_name, lambda assign: 0.0)
    connected_vars = snap.N_fac.get(factor_name, [])
    other_vars = [v for v in connected_vars if v not in [enforce_var, target_var]]

    min_key = (enforce_var, factor_name)
    min_label_idx = min_idx.get(min_key, 0)
    min_label = (
        snap.dom[enforce_var][min_label_idx]
        if min_label_idx < len(snap.dom[enforce_var])
        else "0"
    )

    max_slack = 0.0

    # For each assignment to other variables
    for other_assignment in product_labels(snap, other_vars):
        # Compute best unconstrained value (over all enforce_var labels)
        best_unconstrained = float("+inf")

        for enforce_label in snap.dom[enforce_var]:
            full_assignment = {
                **other_assignment,
                enforce_var: enforce_label,
                target_var: target_label,
            }

            try:
                value = cost_func(full_assignment)

                # Add Q message costs for other variables
                for other_var in other_vars:
                    q_key = (other_var, factor_name)
                    if q_key in snap.Q:
                        q_msg = snap.Q[q_key]
                        label_idx = snap.dom[other_var].index(
                            other_assignment[other_var]
                        )
                        if 0 <= label_idx < len(q_msg):
                            value += q_msg[label_idx]

                # Add Q message cost for enforce_var
                q_key = (enforce_var, factor_name)
                if q_key in snap.Q:
                    q_msg = snap.Q[q_key]
                    label_idx = snap.dom[enforce_var].index(enforce_label)
                    if 0 <= label_idx < len(q_msg):
                        value += q_msg[label_idx]

                best_unconstrained = min(best_unconstrained, value)

            except (KeyError, ValueError, IndexError):
                continue

        # Compute best constrained value (enforce_var = min_label)
        constrained_assignment = {
            **other_assignment,
            enforce_var: min_label,
            target_var: target_label,
        }

        try:
            constrained_value = cost_func(constrained_assignment)

            # Add Q message costs for other variables
            for other_var in other_vars:
                q_key = (other_var, factor_name)
                if q_key in snap.Q:
                    q_msg = snap.Q[q_key]
                    label_idx = snap.dom[other_var].index(other_assignment[other_var])
                    if 0 <= label_idx < len(q_msg):
                        constrained_value += q_msg[label_idx]

            # Add Q message cost for enforce_var (at minimum)
            q_key = (enforce_var, factor_name)
            if q_key in snap.Q:
                q_msg = snap.Q[q_key]
                if 0 <= min_label_idx < len(q_msg):
                    constrained_value += q_msg[min_label_idx]

            # Compute slack
            slack = best_unconstrained - constrained_value
            max_slack = max(max_slack, slack)

        except (KeyError, ValueError, IndexError):
            continue

    # Add safety margin
    epsilon = 1.05 * max_slack if max_slack > 0 else config.abs_tol

    return float(epsilon)


def validate_enforcement_suggestion(
    suggestion: Dict[str, Any], snap: Snapshot
) -> List[str]:
    """
    Validate enforcement suggestion.

    Args:
        suggestion: Enforcement suggestion
        snap: Snapshot

    Returns:
        List of validation errors
    """
    errors = []

    # Check required fields
    required_fields = ["enforce_hop", "epsilon"]
    for field in required_fields:
        if field not in suggestion:
            errors.append(f"Enforcement suggestion missing field '{field}'")

    if "enforce_hop" not in suggestion:
        return errors

    hop = suggestion["enforce_hop"]

    # Validate hop structure
    required_hop_fields = ["fcopy", "u", "v"]
    for field in required_hop_fields:
        if field not in hop:
            errors.append(f"Enforcement hop missing field '{field}'")

    # Validate hop references
    if "fcopy" in hop:
        factor_name = hop["fcopy"]
        if factor_name not in snap.N_fac:
            errors.append(f"Enforcement factor {factor_name} not in snapshot")
        else:
            connected_vars = snap.N_fac[factor_name]

            if "u" in hop and hop["u"] not in connected_vars:
                errors.append(
                    f"Enforcement variable {hop['u']} not connected to factor {factor_name}"
                )

            if "v" in hop and hop["v"] not in connected_vars:
                errors.append(
                    f"Target variable {hop['v']} not connected to factor {factor_name}"
                )

    # Validate epsilon
    if "epsilon" in suggestion:
        epsilon = suggestion["epsilon"]
        if not isinstance(epsilon, (int, float)):
            errors.append(f"Epsilon must be numeric, got {type(epsilon)}")
        elif epsilon < 0:
            errors.append(f"Epsilon must be non-negative, got {epsilon}")
        elif epsilon > 100:  # Sanity check
            errors.append(f"Epsilon seems too large: {epsilon}")

    return errors


def analyze_enforcement_impact(
    snap: Snapshot, enforcement_hop: Dict[str, str], epsilon: float
) -> Dict[str, Any]:
    """
    Analyze the potential impact of epsilon enforcement.

    Args:
        snap: Snapshot
        enforcement_hop: Hop specification
        epsilon: Enforcement epsilon value

    Returns:
        Dictionary with impact analysis
    """
    factor_name = enforcement_hop["fcopy"]
    enforce_var = enforcement_hop["u"]
    target_var = enforcement_hop["v"]

    connected_vars = snap.N_fac.get(factor_name, [])

    # Estimate affected message strength
    q_key = (enforce_var, factor_name)
    q_msg = snap.Q.get(q_key, np.array([]))

    if len(q_msg) > 0:
        q_range = np.max(q_msg) - np.min(q_msg)
        relative_epsilon = epsilon / q_range if q_range > 0 else float("inf")
    else:
        relative_epsilon = float("inf")

    # Count potentially affected winner decisions
    affected_decisions = 0
    total_decisions = 0

    for target_label in snap.dom.get(target_var, []):
        total_decisions += 1

        # This is a simplified check - in practice would need full recomputation
        if epsilon > 0.01:  # Threshold for "significant" change
            affected_decisions += 1

    impact = {
        "factor": factor_name,
        "enforce_variable": enforce_var,
        "target_variable": target_var,
        "epsilon": epsilon,
        "relative_epsilon": relative_epsilon,
        "affected_decisions": affected_decisions,
        "total_decisions": total_decisions,
        "impact_ratio": affected_decisions / total_decisions
        if total_decisions > 0
        else 0,
        "q_message_range": float(q_range) if len(q_msg) > 0 else 0.0,
        "factor_degree": len(connected_vars),
    }

    # Impact assessment
    if relative_epsilon < 0.1:
        impact["severity"] = "low"
    elif relative_epsilon < 0.5:
        impact["severity"] = "medium"
    else:
        impact["severity"] = "high"

    return impact


def generate_enforcement_alternatives(
    snap: Snapshot,
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    max_alternatives: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate alternative enforcement suggestions.

    Args:
        snap: Snapshot
        winners: Winner assignments
        min_idx: Minimum indices
        max_alternatives: Maximum number of alternatives to generate

    Returns:
        List of alternative enforcement suggestions
    """
    alternatives = []

    # Try different factors and variable pairs
    factor_var_pairs = []
    for factor_name in snap.N_fac:
        connected_vars = snap.N_fac[factor_name]
        if len(connected_vars) >= 2:
            for i, var1 in enumerate(connected_vars):
                for var2 in connected_vars[i + 1 :]:
                    factor_var_pairs.append((factor_name, var1, var2))

    # Sort by potential impact (prefer factors with more variables)
    factor_var_pairs.sort(key=lambda x: len(snap.N_fac.get(x[0], [])), reverse=True)

    for factor_name, enforce_var, target_var in factor_var_pairs[:max_alternatives]:
        # Create mock cycle info for this hop
        mock_cycle = {"length": 6, "aligned": False}

        hop = pick_enforcement_hop_specific(
            snap, winners, min_idx, factor_name, enforce_var, target_var
        )

        if hop:
            epsilon = compute_required_epsilon(
                snap, hop, winners, min_idx, AnalysisConfig()
            )

            alternative = {
                "enforce_hop": hop,
                "epsilon": epsilon,
                "reason": "alternative_hop",
                "confidence": "medium",
            }

            alternatives.append(alternative)

            if len(alternatives) >= max_alternatives:
                break

    return alternatives


def pick_enforcement_hop_specific(
    snap: Snapshot,
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
    factor_name: str,
    enforce_var: str,
    target_var: str,
) -> Optional[Dict[str, str]]:
    """
    Pick enforcement hop for specific factor and variable combination.

    Args:
        snap: Snapshot
        winners: Winner assignments
        min_idx: Minimum indices
        factor_name: Factor name
        enforce_var: Variable to enforce
        target_var: Target variable

    Returns:
        Enforcement hop specification or None
    """
    min_key = (enforce_var, factor_name)
    if min_key not in min_idx:
        return None

    min_label_idx = min_idx[min_key]

    # Find a target label where winner disagrees with min
    for target_label in snap.dom.get(target_var, []):
        winner_key = (factor_name, target_var, target_label)
        winner_assignment = winners.get(winner_key, {})

        if enforce_var in winner_assignment:
            winning_label = winner_assignment[enforce_var]
            try:
                winning_label_idx = snap.dom[enforce_var].index(winning_label)
            except ValueError:
                continue

            if winning_label_idx != min_label_idx:
                return {
                    "fcopy": factor_name,
                    "u": enforce_var,
                    "v": target_var,
                    "target_label": target_label,
                    "current_winner_label": winning_label,
                    "min_label_idx": min_label_idx,
                }

    return None


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot
    from .winners import compute_winners, compute_min_idx
    from .slot_indices import build_slot_indices
    from .matrices import build_A, build_P, build_B
    from .cycles import analyze_cycles
    import numpy as np

    print("=== Enforcement Suggestions Examples ===")

    # Create test snapshot with uncertified cycles
    test_snapshot = SimpleSnapshot(
        _lambda=0.8,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"], "x3": ["0", "1"]},
        _N_var={"x1": ["f1", "f2"], "x2": ["f1"], "x3": ["f2"]},
        _N_fac={"f1": ["x1", "x2"], "f2": ["x1", "x3"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.4]),  # min at 0
            ("x1", "f2"): np.array([0.2, 0.0]),  # min at 1
            ("x2", "f1"): np.array([0.0, 0.3]),  # min at 0
            ("x3", "f2"): np.array([0.1, 0.0]),  # min at 1
        },
        _R={
            ("f1", "x1"): np.array([0.1, 0.0]),
            ("f1", "x2"): np.array([0.0, 0.2]),
            ("f2", "x1"): np.array([0.2, 0.1]),
            ("f2", "x3"): np.array([0.0, 0.3]),
        },
        _unary={"x1": np.zeros(2), "x2": np.zeros(2), "x3": np.zeros(2)},
        _cost={
            "f1": lambda assign: 0.5 if assign.get("x1") != assign.get("x2") else 0.0,
            "f2": lambda assign: 0.3 if assign.get("x1") != assign.get("x3") else 0.0,
        },
        _split_map={},
    )

    # Compute analysis components
    winners = compute_winners(test_snapshot)
    min_idx = compute_min_idx(test_snapshot)

    print(f"Computed {len(winners)} winners, {len(min_idx)} min indices")

    # Build matrices and analyze cycles (simplified)
    idxQ, idxR = build_slot_indices(test_snapshot)
    A = build_A(test_snapshot, idxQ, idxR)
    P = build_P(test_snapshot, min_idx, idxQ)
    B = build_B(test_snapshot, winners, idxQ, idxR)

    cycles_analysis = analyze_cycles(
        test_snapshot, A, P, B, idxQ, idxR, winners, min_idx, AnalysisConfig()
    )

    print(f"Cycle analysis: {cycles_analysis['summary']}")

    # Generate enforcement suggestion
    enforcement = suggest_enforcement(test_snapshot, winners, min_idx, cycles_analysis)

    if enforcement:
        print(f"Enforcement suggestion: {enforcement}")

        # Validate suggestion
        errors = validate_enforcement_suggestion(enforcement, test_snapshot)
        if errors:
            print(f"Validation errors: {errors}")
        else:
            print("✓ Enforcement suggestion is valid")

        # Analyze impact
        impact = analyze_enforcement_impact(
            test_snapshot, enforcement["enforce_hop"], enforcement["epsilon"]
        )
        print(f"Impact analysis: {impact}")

    else:
        print("No enforcement needed - all cycles are certified")

    # Generate alternatives
    alternatives = generate_enforcement_alternatives(test_snapshot, winners, min_idx)
    print(f"Generated {len(alternatives)} alternative suggestions")

    for i, alt in enumerate(alternatives):
        print(f"  Alternative {i+1}: epsilon={alt['epsilon']:.4f}")

    print("\n=== Ready for validation module ===")
