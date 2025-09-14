"""
Belief Propagation Analysis Framework

A comprehensive analysis framework for belief propagation simulations that provides
slot-cycle analysis, convergence certificates, and enforcement suggestions.

Main Components:
- Snapshot interface for simulator integration
- Analysis decorator for automatic diagnostics
- Jacobian matrices (A, P, B) for mathematical analysis
- Cycle detection with alignment analysis
- Margin computation and tie detection
- Fixed-region detection across iterations
- Epsilon enforcement suggestions
- Comprehensive invariant validation

Usage Examples:

1. Basic Analysis with Decorator:

    from analyzer import analyze_step

    @analyze_step(return_state=True)
    def step(engine, iteration=0):
        return engine.step(iteration)

    state, analysis = step(engine, 5)

2. Manual Analysis:

    from analyzer import run_diagnostics, create_engine_snapshot

    snapshot = create_engine_snapshot(engine)
    analysis = run_diagnostics(snapshot)

3. Convergence Monitoring:

    from analyzer import monitor_convergence

    results = monitor_convergence(engine, max_steps=50)

4. Analysis Wrapper:

    from analyzer import AnalysisWrapper

    wrapper = AnalysisWrapper(engine)
    step_result, analysis = wrapper.step(iteration=0)
"""

# Main interface exports
from .decorator import (
    analyze_step,
    AnalysisWrapper,
    create_analyzing_step_function,
    batch_analyze_steps,
    monitor_convergence,
)

from .diagnostics import (
    run_diagnostics,
    validate_analysis_result,
    create_analysis_summary,
    extract_convergence_indicators,
)

from .snapshot import (
    Snapshot,
    SimpleSnapshot,
    BPEngineSnapshot,
    create_engine_snapshot,
    has_winners,
    has_min_idx,
    validate_snapshot,
)

from .config import (
    AnalysisConfig,
    get_default_config,
    create_config,
    get_preset_config,
    list_presets,
)

# Component exports for advanced usage
from .slot_indices import build_slot_indices, get_slot_dimensions
from .winners import compute_winners, compute_min_idx
from .matrices import build_A, build_P, build_B, compute_block_norms
from .cycles import analyze_cycles, build_slot_graph, cycle_has_aligned_hop
from .margins import compute_edge_margins, detect_q_message_ties, detect_factor_ties
from .regions import (
    detect_fixed_region,
    analyze_region_stability,
    detect_oscillation_patterns,
)
from .enforcement import suggest_enforcement, analyze_enforcement_impact
from .validation import (
    run_all_invariant_checks,
    generate_invariant_report,
    check_normalization_invariants,
    check_projector_invariants,
)

# Utility exports
from .utils import (
    product_labels,
    normalize_to_min_zero,
    check_min_normalized,
    format_analysis_dict,
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Belief Propagation Analysis Framework"

# Main API - Most common usage patterns
__all__ = [
    # Main interface
    "analyze_step",
    "run_diagnostics",
    "create_engine_snapshot",
    "AnalysisWrapper",
    "monitor_convergence",
    # Configuration
    "AnalysisConfig",
    "get_default_config",
    "create_config",
    "get_preset_config",
    # Snapshot interface
    "Snapshot",
    "SimpleSnapshot",
    "BPEngineSnapshot",
    "validate_snapshot",
    # Analysis utilities
    "create_analysis_summary",
    "extract_convergence_indicators",
    "validate_analysis_result",
    # Advanced components
    "compute_winners",
    "compute_min_idx",
    "build_slot_indices",
    "analyze_cycles",
    "compute_edge_margins",
    "detect_fixed_region",
    "suggest_enforcement",
    "run_all_invariant_checks",
]


def quick_analysis(engine, config=None):
    """
    Perform quick analysis on an engine's current state.

    Args:
        engine: BP engine to analyze
        config: Analysis configuration (optional)

    Returns:
        Analysis result dictionary
    """
    snapshot = create_engine_snapshot(engine)
    return run_diagnostics(snapshot, config)


def analyze_engine_convergence(engine, max_steps=50, config=None):
    """
    Analyze engine convergence over multiple steps.

    Args:
        engine: BP engine to analyze
        max_steps: Maximum steps to run
        config: Analysis configuration (optional)

    Returns:
        Convergence analysis results
    """
    return monitor_convergence(engine, max_steps, config)


def create_comprehensive_report(engine, max_steps=20, config=None):
    """
    Create a comprehensive analysis report for an engine.

    Args:
        engine: BP engine to analyze
        max_steps: Maximum steps to analyze
        config: Analysis configuration (optional)

    Returns:
        Formatted report string
    """
    # Get comprehensive config
    if config is None:
        config = get_preset_config("comprehensive")

    # Run convergence analysis
    convergence_results = monitor_convergence(engine, max_steps, config)

    # Create report
    lines = ["=== COMPREHENSIVE BELIEF PROPAGATION ANALYSIS REPORT ==="]
    lines.append("")

    # Summary
    converged = convergence_results.get("converged", False)
    total_steps = convergence_results.get("total_steps", 0)
    convergence_step = convergence_results.get("convergence_step")

    lines.append("CONVERGENCE SUMMARY:")
    lines.append(f"  Converged: {' YES' if converged else ' NO'}")
    lines.append(f"  Total steps analyzed: {total_steps}")
    if converged and convergence_step is not None:
        lines.append(f"  Convergence achieved at step: {convergence_step}")
    lines.append("")

    # Final state analysis
    final_analysis = convergence_results.get("final_analysis")
    if final_analysis:
        lines.append("FINAL STATE ANALYSIS:")
        summary = create_analysis_summary(final_analysis)
        lines.append(summary)
        lines.append("")

    # Trajectory analysis
    trajectory = convergence_results.get("convergence_trajectory", [])
    if trajectory:
        lines.append("CONVERGENCE TRAJECTORY:")
        lines.append("Step | Fixed | Cert | Aligned | Margin    | Ties")
        lines.append("-" * 50)

        for data in trajectory[-10:]:  # Last 10 steps
            step = data.get("step", 0)
            fixed = "" if data.get("region_fixed", False) else ""
            cert = "" if data.get("has_certification", False) else ""
            aligned = data.get("aligned_hops", 0)
            margin = data.get("min_margin")
            ties = data.get("num_ties", 0)

            margin_str = f"{margin:.6f}" if margin is not None else "None"
            lines.append(
                f"{step:4d} | {fixed:5s} | {cert:4s} | {aligned:7d} | {margin_str:9s} | {ties:4d}"
            )

    lines.append("")
    lines.append("=== END REPORT ===")

    return "\n".join(lines)


# Convenience aliases for common operations
analyze = run_diagnostics
create_snapshot = create_engine_snapshot
check_convergence = monitor_convergence


# Framework information
def get_framework_info():
    """Get information about the analysis framework."""
    return {
        "name": "Belief Propagation Analysis Framework",
        "version": __version__,
        "description": "Comprehensive analysis for belief propagation algorithms",
        "features": [
            "Slot-cycle analysis with alignment detection",
            "Convergence certificates based on mathematical theory",
            "Epsilon enforcement suggestions",
            "Matrix norm analysis (A, P, B matrices)",
            "Margin computation and tie detection",
            "Fixed-region detection across iterations",
            "Comprehensive invariant validation",
            "Flexible configuration system",
            "Multiple integration patterns",
        ],
        "main_functions": [
            "analyze_step - Decorator for automatic analysis",
            "run_diagnostics - Core analysis function",
            "monitor_convergence - Multi-step convergence analysis",
            "create_engine_snapshot - Snapshot extraction",
            "AnalysisWrapper - Engine wrapper for analysis",
        ],
    }


def print_framework_info():
    """Print framework information."""
    info = get_framework_info()
    print(f"{info['name']} v{info['version']}")
    print(f"{info['description']}")
    print("\nKey Features:")
    for feature in info["features"]:
        print(f"  - {feature}")
    print("\nMain Functions:")
    for func in info["main_functions"]:
        print(f"  - {func}")


# Example usage function
def example_usage():
    """Print example usage patterns."""
    print(
        """
=== BELIEF PROPAGATION ANALYSIS FRAMEWORK EXAMPLES ===

1. BASIC DECORATOR USAGE:

    from analyzer import analyze_step

    @analyze_step(return_state=True, config={"max_cycle_len": 10})
    def step(engine, iteration=0):
        return engine.step(iteration)

    # Use it
    state, analysis = step(engine, 5)
    print(f"Converged: {analysis.get('region_fixed', False)}")

2. MANUAL ANALYSIS:

    from analyzer import run_diagnostics, create_engine_snapshot

    snapshot = create_engine_snapshot(engine)
    analysis = run_diagnostics(snapshot, {"check_invariants": True})

3. CONVERGENCE MONITORING:

    from analyzer import monitor_convergence

    results = monitor_convergence(engine, max_steps=50)
    print(f"Converged after {results['convergence_step']} steps")

4. COMPREHENSIVE REPORT:

    from analyzer import create_comprehensive_report

    report = create_comprehensive_report(engine, max_steps=30)
    print(report)

5. WRAPPER USAGE:

    from analyzer import AnalysisWrapper

    wrapper = AnalysisWrapper(engine, config="comprehensive")

    for i in range(20):
        result, analysis = wrapper.step(i)
        if analysis.get('region_fixed', False):
            print(f"Converged at step {i}")
            break

For more examples, see the documentation or run help() on specific functions.
"""
    )


if __name__ == "__main__":
    print_framework_info()
    print()
    example_usage()
