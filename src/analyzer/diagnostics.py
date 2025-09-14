"""
Main diagnostics orchestrator for belief propagation analysis.

Coordinates all analysis components and produces standardized JSON output
for step-by-step analysis of belief propagation convergence.
"""

from typing import Dict, Any, Optional, List
from .snapshot import Snapshot, has_winners, has_min_idx
from .config import AnalysisConfig, validate_config
from .slot_indices import build_slot_indices
from .winners import compute_winners, compute_min_idx
from .matrices import build_A, build_P, build_B, compute_block_norms
from .cycles import analyze_cycles
from .margins import compute_edge_margins
from .regions import detect_fixed_region
from .enforcement import suggest_enforcement
from .validation import run_all_invariant_checks, get_violation_summary


def run_diagnostics(
    snap: Snapshot,
    cfg: Optional[Dict[str, Any]] = None,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run complete diagnostic analysis on a belief propagation snapshot.

    This is the main orchestrator function that coordinates all analysis
    components and produces a standardized output format.

    Args:
        snap: Snapshot containing factor graph state
        cfg: Configuration dictionary (will be validated)
        prev: Previous analysis results for region detection

    Returns:
        Dictionary with complete analysis results
    """
    # Validate and normalize configuration
    try:
        config = validate_config(cfg)
    except (TypeError, ValueError) as e:
        return {
            "version": "analysis-v1",
            "error": f"Invalid configuration: {e}",
            "timestamp": None,
        }

    analysis_result = {"version": "analysis-v1", "config": config.to_dict()}

    try:
        # Step 1: Build slot indices
        idxQ, idxR = build_slot_indices(snap)
        analysis_result["sizes"] = {"nQ": len(idxQ), "nR": len(idxR)}

        # Step 2: Compute winners and min_idx (if missing)
        winners = snap.winners if has_winners(snap) else None
        min_idx = snap.min_idx if has_min_idx(snap) else None

        if winners is None and config.rebuild_winners_if_missing:
            winners = compute_winners(snap, config)

        if min_idx is None and config.rebuild_min_idx_if_missing:
            min_idx = compute_min_idx(snap)

        # Check if we have required data
        if winners is None:
            analysis_result["error"] = "Winners data missing and rebuild disabled"
            return analysis_result

        if min_idx is None:
            analysis_result["error"] = "Min_idx data missing and rebuild disabled"
            return analysis_result

        # Step 3: Build matrices A, P, B
        A = build_A(snap, idxQ, idxR)
        P = build_P(snap, min_idx, idxQ)
        B = build_B(snap, winners, idxQ, idxR)

        # Step 4: Compute block norms
        if config.include_matrix_norms:
            norms = compute_block_norms(A, P, B, snap.lambda_)
            analysis_result["norms"] = norms

        # Step 5: Analyze cycles
        cycles_analysis = analyze_cycles(
            snap, A, P, B, idxQ, idxR, winners, min_idx, config
        )
        analysis_result["cycles"] = cycles_analysis["summary"]

        if config.include_detailed_cycles:
            analysis_result["cycles_detail"] = cycles_analysis["detail"]

        analysis_result["aligned_hops_total"] = cycles_analysis["aligned_hops_total"]

        # Step 6: Compute edge margins
        margins_analysis = compute_edge_margins(snap, config)
        analysis_result["margins"] = margins_analysis["summary"]
        analysis_result["ties"] = margins_analysis["ties"]

        # Step 7: Detect fixed region
        prev_region_keys = prev.get("region_keys") if prev else None
        region_analysis = detect_fixed_region(prev_region_keys, winners, min_idx)

        analysis_result["region_fixed"] = region_analysis["fixed"]
        analysis_result["region_changes"] = region_analysis["changes"]
        analysis_result["region_keys"] = region_analysis["keys"]

        # Step 8: Suggest enforcement (if needed)
        if config.include_enforcement_suggestions:
            enforcement = suggest_enforcement(
                snap, winners, min_idx, cycles_analysis, config
            )
            analysis_result["enforcement_suggestion"] = enforcement

        # Step 9: Run invariant checks
        if config.check_invariants:
            violations = run_all_invariant_checks(
                snap, A, P, B, idxQ, idxR, cycles_analysis.get("detail", []), config
            )

            violation_summary = get_violation_summary(violations)
            analysis_result["invariant_violations"] = violation_summary

            # Include detailed violations if requested
            if config.verbose_warnings and violations:
                analysis_result["violation_details"] = violations

        # Additional metadata
        analysis_result["analysis_complete"] = True
        analysis_result["snapshot_lambda"] = snap.lambda_

    except Exception as e:
        analysis_result["error"] = f"Analysis failed: {str(e)}"
        analysis_result["analysis_complete"] = False

    return analysis_result


def validate_analysis_result(result: Dict[str, Any]) -> List[str]:
    """
    Validate analysis result structure and content.

    Args:
        result: Analysis result dictionary

    Returns:
        List of validation errors
    """
    errors = []

    # Check required top-level fields
    required_fields = ["version"]
    for field in required_fields:
        if field not in result:
            errors.append(f"Analysis result missing required field: {field}")

    # Check version
    if result.get("version") != "analysis-v1":
        errors.append(f"Unsupported analysis version: {result.get('version')}")

    # If there's an error, other validations don't apply
    if "error" in result:
        return errors

    # Validate sizes
    if "sizes" in result:
        sizes = result["sizes"]
        if not isinstance(sizes.get("nQ"), int) or sizes["nQ"] < 0:
            errors.append("Invalid nQ size")
        if not isinstance(sizes.get("nR"), int) or sizes["nR"] < 0:
            errors.append("Invalid nR size")

    # Validate norms (if present)
    if "norms" in result:
        norms = result["norms"]
        norm_keys = ["||BPA||_inf", "||B||_inf", "||PA||_inf", "||M||_inf_upper"]
        for key in norm_keys:
            if key in norms:
                value = norms[key]
                if value is not None and (
                    not isinstance(value, (int, float)) or value < 0
                ):
                    errors.append(f"Invalid norm value for {key}: {value}")

    # Validate cycles
    if "cycles" in result:
        cycles = result["cycles"]
        required_cycle_fields = [
            "num_cycles",
            "aligned_hops_total",
            "has_certified_contraction",
        ]
        for field in required_cycle_fields:
            if field not in cycles:
                errors.append(f"Cycles summary missing field: {field}")

    # Validate margins
    if "margins" in result:
        margins = result["margins"]
        if "min_margin" in margins and margins["min_margin"] is not None:
            if not isinstance(margins["min_margin"], (int, float)):
                errors.append("Invalid min_margin type")

        if "num_ties" in margins:
            if not isinstance(margins["num_ties"], int) or margins["num_ties"] < 0:
                errors.append("Invalid num_ties value")

    # Validate region analysis
    if "region_fixed" in result:
        if not isinstance(result["region_fixed"], bool):
            errors.append("region_fixed must be boolean")

    return errors


def create_analysis_summary(result: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of analysis results.

    Args:
        result: Analysis result dictionary

    Returns:
        Formatted summary string
    """
    if "error" in result:
        return f"❌ Analysis failed: {result['error']}"

    lines = ["=== Belief Propagation Analysis Summary ==="]

    # Basic info
    sizes = result.get("sizes", {})
    lines.append(
        f"Slots: {sizes.get('nQ', 0)} Q-messages, {sizes.get('nR', 0)} R-messages"
    )
    lines.append(f"Lambda: {result.get('snapshot_lambda', 'unknown')}")

    # Convergence status
    region_fixed = result.get("region_fixed", False)
    cycles_summary = result.get("cycles", {})
    has_certification = cycles_summary.get("has_certified_contraction", False)

    lines.append("")
    lines.append("Convergence Analysis:")
    lines.append(f"  Region fixed: {'✓' if region_fixed else '✗'}")
    lines.append(f"  Certified contraction: {'✓' if has_certification else '✗'}")
    lines.append(f"  Aligned hops: {result.get('aligned_hops_total', 0)}")

    # Cycles
    num_cycles = cycles_summary.get("num_cycles", 0)
    lines.append(f"  Cycles found: {num_cycles}")

    # Margins and ties
    margins = result.get("margins", {})
    min_margin = margins.get("min_margin")
    num_ties = margins.get("num_ties", 0)

    lines.append("")
    lines.append("Margin Analysis:")
    if min_margin is not None:
        lines.append(f"  Minimum margin: {min_margin:.6f}")
    else:
        lines.append("  Minimum margin: undefined")
    lines.append(f"  Ties detected: {num_ties}")

    # Enforcement suggestion
    enforcement = result.get("enforcement_suggestion")
    if enforcement:
        lines.append("")
        lines.append("Enforcement Suggestion:")
        hop = enforcement.get("enforce_hop", {})
        epsilon = enforcement.get("epsilon", 0)
        lines.append(f"  Factor: {hop.get('fcopy', 'unknown')}")
        lines.append(f"  Variable: {hop.get('u', 'unknown')}")
        lines.append(f"  Epsilon: {epsilon:.6f}")

    # Invariant violations
    violations = result.get("invariant_violations")
    if violations:
        total_violations = violations.get("total_violations", 0)
        lines.append("")
        if total_violations > 0:
            lines.append(f"⚠️  Invariant violations: {total_violations}")
            severity = violations.get("severity", "unknown")
            lines.append(f"  Severity: {severity}")
        else:
            lines.append("✓ All invariants satisfied")

    # Matrix norms (if available)
    norms = result.get("norms")
    if norms:
        lines.append("")
        lines.append("Matrix Norms:")
        for norm_name, norm_value in norms.items():
            if norm_value is not None:
                lines.append(f"  {norm_name}: {norm_value:.6f}")

    lines.append("")
    lines.append("=== End Summary ===")

    return "\n".join(lines)


def extract_convergence_indicators(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key convergence indicators from analysis result.

    Args:
        result: Analysis result dictionary

    Returns:
        Dictionary with convergence indicators
    """
    indicators = {"converged": False, "confidence": "unknown"}

    if "error" in result:
        return indicators

    # Main convergence indicators
    region_fixed = result.get("region_fixed", False)
    has_certification = result.get("cycles", {}).get("has_certified_contraction", False)
    aligned_hops = result.get("aligned_hops_total", 0)

    # Determine convergence status
    if region_fixed and has_certification:
        indicators["converged"] = True
        indicators["confidence"] = "high"
    elif region_fixed:
        indicators["converged"] = True
        indicators["confidence"] = "medium"
    elif has_certification:
        indicators["converged"] = False  # Certified but not fixed yet
        indicators["confidence"] = "medium"
    else:
        indicators["converged"] = False
        indicators["confidence"] = "low"

    # Additional details
    indicators.update(
        {
            "region_fixed": region_fixed,
            "has_certification": has_certification,
            "aligned_hops": aligned_hops,
            "num_cycles": result.get("cycles", {}).get("num_cycles", 0),
            "min_margin": result.get("margins", {}).get("min_margin"),
            "num_ties": result.get("margins", {}).get("num_ties", 0),
            "needs_enforcement": result.get("enforcement_suggestion") is not None,
        }
    )

    # Violations affect confidence
    violations = result.get("invariant_violations", {})
    if violations.get("total_violations", 0) > 0:
        if indicators["confidence"] == "high":
            indicators["confidence"] = "medium"
        elif indicators["confidence"] == "medium":
            indicators["confidence"] = "low"

    return indicators


# Example usage and testing
if __name__ == "__main__":
    from .snapshot import SimpleSnapshot, create_engine_snapshot
    import numpy as np

    print("=== Diagnostics Orchestrator Examples ===")

    # Create comprehensive test snapshot
    test_snapshot = SimpleSnapshot(
        _lambda=0.6,
        _dom={"x1": ["0", "1"], "x2": ["0", "1"], "x3": ["0", "1"]},
        _N_var={"x1": ["f1", "f2"], "x2": ["f1"], "x3": ["f2"]},
        _N_fac={"f1": ["x1", "x2"], "f2": ["x1", "x3"]},
        _Q={
            ("x1", "f1"): np.array([0.0, 0.3]),
            ("x1", "f2"): np.array([0.1, 0.0]),
            ("x2", "f1"): np.array([0.0, 0.2]),
            ("x3", "f2"): np.array([0.2, 0.0]),
        },
        _R={
            ("f1", "x1"): np.array([0.1, 0.0]),
            ("f1", "x2"): np.array([0.0, 0.2]),
            ("f2", "x1"): np.array([0.2, 0.1]),
            ("f2", "x3"): np.array([0.0, 0.3]),
        },
        _unary={"x1": np.zeros(2), "x2": np.zeros(2), "x3": np.zeros(2)},
        _cost={
            "f1": lambda assign: 0.1 if assign.get("x1") == assign.get("x2") else 0.0,
            "f2": lambda assign: 0.2 if assign.get("x1") == assign.get("x3") else 0.0,
        },
        _split_map={},
    )

    print("Running complete diagnostics analysis...")

    # Run diagnostics with comprehensive config
    config = {
        "max_cycle_len": 10,
        "compute_numeric_cycle_gain": True,
        "include_detailed_cycles": True,
        "check_invariants": True,
        "validate_normalization": True,
        "include_enforcement_suggestions": True,
    }

    # First iteration (no previous data)
    result1 = run_diagnostics(test_snapshot, config, prev=None)

    print("First iteration analysis:")
    print(f"  Analysis complete: {result1.get('analysis_complete', False)}")
    print(f"  Sizes: {result1.get('sizes', {})}")
    print(f"  Region fixed: {result1.get('region_fixed', False)}")
    print(f"  Cycles: {result1.get('cycles', {})}")
    print(f"  Enforcement needed: {result1.get('enforcement_suggestion') is not None}")

    # Validate result
    validation_errors = validate_analysis_result(result1)
    if validation_errors:
        print(f"  Validation errors: {validation_errors}")
    else:
        print("  ✓ Analysis result is valid")

    # Second iteration (with previous data)
    result2 = run_diagnostics(test_snapshot, config, prev=result1)

    print("\nSecond iteration analysis:")
    print(f"  Region fixed: {result2.get('region_fixed', False)}")
    region_changes = result2.get("region_changes")
    if region_changes:
        print(f"  Changes: {region_changes}")

    # Generate summary
    summary = create_analysis_summary(result2)
    print(f"\nAnalysis Summary:\n{summary}")

    # Extract convergence indicators
    convergence = extract_convergence_indicators(result2)
    print(f"\nConvergence Indicators: {convergence}")

    print("\n=== Ready for decorator integration ===")
