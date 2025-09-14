"""
Fixed-region detection across iterations for belief propagation analysis.

Tracks winner and min_idx stability across iterations to detect when
the system has reached a fixed point in its selection behavior.
"""

from typing import Dict, Tuple, List, Optional, Any, Set
import hashlib
import json
from .snapshot import Snapshot
from .config import AnalysisConfig


def generate_region_keys(
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
) -> Dict[str, str]:
    """
    Generate stable keys for winners and min_idx for cross-iteration comparison.

    Args:
        winners: Winner assignments
        min_idx: Minimum indices

    Returns:
        Dictionary with stable hash keys
    """
    # Create stable representation of winners
    winners_items = []
    for key, assignment in sorted(winners.items()):
        # Sort assignment items for stability
        assignment_items = sorted(assignment.items())
        winners_items.append((key, tuple(assignment_items)))
    winners_key = str(tuple(winners_items))

    # Create stable representation of min_idx
    min_idx_items = sorted(min_idx.items())
    min_idx_key = str(tuple(min_idx_items))

    # Generate compact hashes for efficient comparison
    winners_hash = hashlib.md5(winners_key.encode()).hexdigest()
    min_idx_hash = hashlib.md5(min_idx_key.encode()).hexdigest()

    return {
        "winners": winners_hash,
        "min_idx": min_idx_hash,
        "combined": hashlib.md5((winners_hash + min_idx_hash).encode()).hexdigest(),
    }


def detect_fixed_region(
    prev_keys: Optional[Dict[str, str]],
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    """
    Detect fixed region by comparing current keys with previous iteration.

    Uses keys made from winners and min_idx to determine stability.
    A fixed region occurs when both winners and min_idx remain unchanged.

    Args:
        prev_keys: Keys from previous iteration (None for first iteration)
        winners: Current winner assignments
        min_idx: Current minimum indices

    Returns:
        Dictionary with fixed region analysis
    """
    # Generate keys for current state
    current_keys = generate_region_keys(winners, min_idx)

    if prev_keys is None:
        # First iteration - cannot detect fixedness yet
        return {
            "fixed": False,
            "changes": None,
            "keys": current_keys,
            "iteration_count": 1,
            "first_iteration": True,
        }

    # Compare with previous iteration
    winners_changed = prev_keys["winners"] != current_keys["winners"]
    min_idx_changed = prev_keys["min_idx"] != current_keys["min_idx"]

    fixed = not (winners_changed or min_idx_changed)

    changes = (
        {
            "winners_changed": winners_changed,
            "min_idx_changed": min_idx_changed,
            "combined_changed": winners_changed or min_idx_changed,
        }
        if not fixed
        else None
    )

    return {
        "fixed": fixed,
        "changes": changes,
        "keys": current_keys,
        "iteration_count": prev_keys.get("iteration_count", 1) + 1,
        "first_iteration": False,
    }


def analyze_region_stability(
    region_history: List[Dict[str, Any]], window_size: int = 5
) -> Dict[str, Any]:
    """
    Analyze stability patterns across multiple iterations.

    Args:
        region_history: List of region detection results from multiple iterations
        window_size: Number of recent iterations to analyze

    Returns:
        Dictionary with stability analysis
    """
    if not region_history:
        return {
            "total_iterations": 0,
            "stable_iterations": 0,
            "stability_ratio": 0.0,
            "converged": False,
        }

    total_iterations = len(region_history)
    recent_history = (
        region_history[-window_size:] if window_size > 0 else region_history
    )

    # Count stable iterations
    stable_iterations = sum(1 for h in region_history if h.get("fixed", False))
    recent_stable = sum(1 for h in recent_history if h.get("fixed", False))

    # Check for convergence (all recent iterations are stable)
    converged = len(recent_history) >= min(window_size, 3) and recent_stable == len(
        recent_history
    )

    # Analyze change patterns
    winner_changes = sum(
        1
        for h in region_history
        if h.get("changes") and h["changes"].get("winners_changed", False)
    )
    min_idx_changes = sum(
        1
        for h in region_history
        if h.get("changes") and h["changes"].get("min_idx_changed", False)
    )

    analysis = {
        "total_iterations": total_iterations,
        "stable_iterations": stable_iterations,
        "stability_ratio": stable_iterations / total_iterations
        if total_iterations > 0
        else 0.0,
        "converged": converged,
        "recent_stable_iterations": recent_stable,
        "recent_stability_ratio": recent_stable / len(recent_history)
        if recent_history
        else 0.0,
        "winner_changes": winner_changes,
        "min_idx_changes": min_idx_changes,
        "window_size": len(recent_history),
    }

    # Convergence iteration (first iteration where stability began)
    if converged:
        for i, h in enumerate(region_history):
            if h.get("fixed", False):
                # Check if stability continues from this point
                remaining = region_history[i:]
                if all(r.get("fixed", False) for r in remaining):
                    analysis["convergence_iteration"] = i
                    analysis["iterations_since_convergence"] = len(remaining)
                    break

    return analysis


def detect_oscillation_patterns(
    region_history: List[Dict[str, Any]],
    min_pattern_length: int = 2,
    max_pattern_length: int = 10,
) -> Dict[str, Any]:
    """
    Detect oscillation patterns in region keys.

    Args:
        region_history: List of region detection results
        min_pattern_length: Minimum oscillation period to detect
        max_pattern_length: Maximum oscillation period to detect

    Returns:
        Dictionary with oscillation analysis
    """
    if len(region_history) < 2 * min_pattern_length:
        return {
            "has_oscillation": False,
            "pattern_length": None,
            "pattern_start": None,
            "oscillation_type": None,
        }

    # Extract key sequences
    combined_keys = [h.get("keys", {}).get("combined", "") for h in region_history]
    winner_keys = [h.get("keys", {}).get("winners", "") for h in region_history]
    min_idx_keys = [h.get("keys", {}).get("min_idx", "") for h in region_history]

    # Check for periodic patterns
    for pattern_length in range(
        min_pattern_length, min(max_pattern_length + 1, len(combined_keys) // 2)
    ):
        # Check if the last few periods match
        if len(combined_keys) >= 3 * pattern_length:
            pattern = combined_keys[-pattern_length:]
            prev_pattern = combined_keys[-(2 * pattern_length) : -pattern_length]
            prev_prev_pattern = combined_keys[
                -(3 * pattern_length) : -(2 * pattern_length)
            ]

            if pattern == prev_pattern == prev_prev_pattern:
                # Found oscillation pattern
                oscillation_type = "combined"

                # Check if oscillation is in winners, min_idx, or both
                winner_pattern = winner_keys[-pattern_length:]
                winner_prev = winner_keys[-(2 * pattern_length) : -pattern_length]
                min_idx_pattern = min_idx_keys[-pattern_length:]
                min_idx_prev = min_idx_keys[-(2 * pattern_length) : -pattern_length]

                if winner_pattern == winner_prev and min_idx_pattern != min_idx_prev:
                    oscillation_type = "min_idx_only"
                elif winner_pattern != winner_prev and min_idx_pattern == min_idx_prev:
                    oscillation_type = "winners_only"
                elif winner_pattern == winner_prev and min_idx_pattern == min_idx_prev:
                    oscillation_type = (
                        "neither"  # Should not happen if combined oscillates
                    )
                else:
                    oscillation_type = "both"

                # Find when oscillation started
                pattern_start = None
                for start_idx in range(len(combined_keys) - pattern_length):
                    if combined_keys[start_idx : start_idx + pattern_length] == pattern:
                        pattern_start = start_idx
                        break

                return {
                    "has_oscillation": True,
                    "pattern_length": pattern_length,
                    "pattern_start": pattern_start,
                    "oscillation_type": oscillation_type,
                    "pattern": pattern,
                    "iterations_oscillating": len(combined_keys) - (pattern_start or 0),
                }

    return {
        "has_oscillation": False,
        "pattern_length": None,
        "pattern_start": None,
        "oscillation_type": None,
    }


def validate_region_detection(
    region_result: Dict[str, Any],
    winners: Dict[Tuple[str, str, str], Dict[str, str]],
    min_idx: Dict[Tuple[str, str], int],
) -> List[str]:
    """
    Validate region detection results.

    Args:
        region_result: Region detection result
        winners: Winner assignments used for detection
        min_idx: Min indices used for detection

    Returns:
        List of validation errors
    """
    errors = []

    # Check required fields
    required_fields = ["fixed", "keys"]
    for field in required_fields:
        if field not in region_result:
            errors.append(f"Region result missing field '{field}'")

    # Validate keys
    if "keys" in region_result:
        keys = region_result["keys"]
        required_key_fields = ["winners", "min_idx", "combined"]

        for key_field in required_key_fields:
            if key_field not in keys:
                errors.append(f"Region keys missing field '{key_field}'")
            elif not isinstance(keys[key_field], str):
                errors.append(f"Region key '{key_field}' is not a string")

    # Validate consistency
    if region_result.get("fixed", False):
        if region_result.get("changes") is not None:
            errors.append("Fixed region should not have changes")
    else:
        if region_result.get("changes") is None and not region_result.get(
            "first_iteration", False
        ):
            errors.append(
                "Non-fixed region should have changes (unless first iteration)"
            )

    # Validate iteration count
    iteration_count = region_result.get("iteration_count", 0)
    if not isinstance(iteration_count, int) or iteration_count < 1:
        errors.append(f"Invalid iteration count: {iteration_count}")

    # Check that keys can be regenerated
    try:
        regenerated_keys = generate_region_keys(winners, min_idx)
        if "keys" in region_result:
            for key_field in ["winners", "min_idx", "combined"]:
                if (
                    key_field in region_result["keys"]
                    and key_field in regenerated_keys
                    and region_result["keys"][key_field] != regenerated_keys[key_field]
                ):
                    errors.append(
                        f"Region key '{key_field}' does not match regenerated key"
                    )
    except Exception as e:
        errors.append(f"Failed to regenerate keys: {e}")

    return errors


def compute_region_statistics(region_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about region detection across iterations.

    Args:
        region_history: List of region detection results

    Returns:
        Dictionary with region statistics
    """
    if not region_history:
        return {"total_iterations": 0}

    total_iterations = len(region_history)
    fixed_count = sum(1 for h in region_history if h.get("fixed", False))

    # Analyze change types
    winner_changes = sum(
        1
        for h in region_history
        if h.get("changes") and h["changes"].get("winners_changed", False)
    )
    min_idx_changes = sum(
        1
        for h in region_history
        if h.get("changes") and h["changes"].get("min_idx_changed", False)
    )
    combined_changes = sum(
        1
        for h in region_history
        if h.get("changes") and h["changes"].get("combined_changed", False)
    )

    # Find longest stable streak
    max_stable_streak = 0
    current_streak = 0

    for h in region_history:
        if h.get("fixed", False):
            current_streak += 1
            max_stable_streak = max(max_stable_streak, current_streak)
        else:
            current_streak = 0

    # Find first stable iteration
    first_stable_iteration = None
    for i, h in enumerate(region_history):
        if h.get("fixed", False):
            first_stable_iteration = i
            break

    # Unique key counts
    unique_winner_keys = set()
    unique_min_idx_keys = set()
    unique_combined_keys = set()

    for h in region_history:
        keys = h.get("keys", {})
        if "winners" in keys:
            unique_winner_keys.add(keys["winners"])
        if "min_idx" in keys:
            unique_min_idx_keys.add(keys["min_idx"])
        if "combined" in keys:
            unique_combined_keys.add(keys["combined"])

    return {
        "total_iterations": total_iterations,
        "fixed_iterations": fixed_count,
        "stability_ratio": fixed_count / total_iterations,
        "winner_changes": winner_changes,
        "min_idx_changes": min_idx_changes,
        "combined_changes": combined_changes,
        "max_stable_streak": max_stable_streak,
        "first_stable_iteration": first_stable_iteration,
        "unique_winner_keys": len(unique_winner_keys),
        "unique_min_idx_keys": len(unique_min_idx_keys),
        "unique_combined_keys": len(unique_combined_keys),
    }


def suggest_convergence_acceleration(
    region_history: List[Dict[str, Any]], oscillation_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Suggest methods to accelerate convergence based on region analysis.

    Args:
        region_history: Region detection history
        oscillation_analysis: Oscillation pattern analysis

    Returns:
        Dictionary with convergence acceleration suggestions
    """
    suggestions = {"needs_acceleration": False, "suggestions": [], "confidence": "low"}

    if not region_history:
        return suggestions

    # Check if converged already
    recent_history = region_history[-3:] if len(region_history) >= 3 else region_history
    if all(h.get("fixed", False) for h in recent_history):
        suggestions["suggestions"].append("Already converged - no acceleration needed")
        return suggestions

    suggestions["needs_acceleration"] = True

    # Oscillation-based suggestions
    if oscillation_analysis.get("has_oscillation", False):
        pattern_length = oscillation_analysis.get("pattern_length", 0)
        oscillation_type = oscillation_analysis.get("oscillation_type", "")

        suggestions["suggestions"].append(
            f"Detected {oscillation_type} oscillation with period {pattern_length}"
        )

        if oscillation_type in ["winners_only", "both"]:
            suggestions["suggestions"].append(
                "Consider epsilon-enforcement to break winner oscillations"
            )

        if oscillation_type in ["min_idx_only", "both"]:
            suggestions["suggestions"].append(
                "Consider message damping to stabilize min indices"
            )

        suggestions["confidence"] = "high"

    # Non-convergence suggestions
    else:
        stability_ratio = sum(1 for h in region_history if h.get("fixed", False)) / len(
            region_history
        )

        if stability_ratio < 0.1:
            suggestions["suggestions"].append(
                "Very low stability - check for numerical issues or increase damping"
            )
            suggestions["confidence"] = "medium"
        elif stability_ratio < 0.5:
            suggestions["suggestions"].append(
                "Moderate instability - consider epsilon-enforcement or damping adjustment"
            )
            suggestions["confidence"] = "medium"
        else:
            suggestions["suggestions"].append(
                "High stability with occasional changes - minor parameter adjustment may help"
            )
            suggestions["confidence"] = "low"

    return suggestions


# Example usage and testing
if __name__ == "__main__":
    from .winners import compute_winners, compute_min_idx
    from .snapshot import SimpleSnapshot
    import numpy as np

    print("=== Region Detection Examples ===")

    # Create test snapshots for multiple iterations
    test_snapshots = []
    for i in range(5):
        # Slightly different Q messages to simulate iteration changes
        lambda_val = 0.5 + i * 0.01
        q1 = np.array([0.0, 0.2 + i * 0.1])
        q2 = np.array([0.1 - i * 0.02, 0.0])

        snapshot = SimpleSnapshot(
            _lambda=lambda_val,
            _dom={"x1": ["0", "1"], "x2": ["0", "1"]},
            _N_var={"x1": ["f1"], "x2": ["f1"]},
            _N_fac={"f1": ["x1", "x2"]},
            _Q={("x1", "f1"): q1, ("x2", "f1"): q2},
            _R={("f1", "x1"): np.array([0.1, 0.0]), ("f1", "x2"): np.array([0.0, 0.1])},
            _unary={"x1": np.zeros(2), "x2": np.zeros(2)},
            _cost={"f1": lambda assign: 0.0},
            _split_map={},
        )
        test_snapshots.append(snapshot)

    # Simulate region detection across iterations
    region_history = []
    prev_keys = None

    for i, snapshot in enumerate(test_snapshots):
        winners = compute_winners(snapshot)
        min_idx = compute_min_idx(snapshot)

        region_result = detect_fixed_region(prev_keys, winners, min_idx)
        region_history.append(region_result)

        print(f"Iteration {i}: Fixed={region_result['fixed']}")
        if region_result["changes"]:
            print(f"  Changes: {region_result['changes']}")

        prev_keys = region_result["keys"]

    # Analyze stability
    stability_analysis = analyze_region_stability(region_history)
    print(f"Stability analysis: {stability_analysis}")

    # Check for oscillations
    oscillation_analysis = detect_oscillation_patterns(region_history)
    print(f"Oscillation analysis: {oscillation_analysis}")

    # Get statistics
    stats = compute_region_statistics(region_history)
    print(f"Region statistics: {stats}")

    # Validate results
    if region_history:
        last_result = region_history[-1]
        last_winners = compute_winners(test_snapshots[-1])
        last_min_idx = compute_min_idx(test_snapshots[-1])

        errors = validate_region_detection(last_result, last_winners, last_min_idx)
        if errors:
            print(f"Validation errors: {errors}")
        else:
            print("✓ Region detection is valid")

    # Get convergence suggestions
    suggestions = suggest_convergence_acceleration(region_history, oscillation_analysis)
    print(f"Convergence suggestions: {suggestions}")

    print("\n=== Ready for enforcement suggestions ===")
