"""
Analysis Framework Integration Example

Demonstrates how to integrate the belief propagation analysis framework
with existing BPEngine code for comprehensive convergence analysis.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, "/mnt/c/users/ormul/PycharmProjects/Belief-Propagation-Simulator")

try:
    # Import existing BP components
    from src.propflow.bp.engines import DampingEngine, Engine
    from src.propflow.bp.engine_base import BPEngine
    from src.propflow.utils import FGBuilder
    from src.propflow.configs import CTFactory

    # Import analysis framework
    from src.analyzer import (
        analyze_step,
        run_diagnostics,
        create_engine_snapshot,
        monitor_convergence,
        AnalysisWrapper,
        create_comprehensive_report,
        AnalysisConfig,
    )

    print("✓ All imports successful")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_test_factor_graph():
    """Create a test factor graph for analysis demonstrations."""
    builder = FGBuilder()

    # Add variables with different domain sizes
    builder.add_variables(names=["x1", "x2", "x3"], domain_sizes=[2, 2, 3])

    # Add factors with interesting cost structures
    # Factor connecting x1 and x2 (prefers different values)
    builder.add_factor(["x1", "x2"], cost_table=np.array([[0.5, 0.0], [0.0, 0.3]]))

    # Factor connecting x2 and x3 (more complex interaction)
    builder.add_factor(["x2", "x3"], cost_table=np.random.rand(2, 3) * 0.4)

    # Unary factor on x1 (slight bias)
    builder.add_factor(["x1"], cost_table=np.array([0.1, 0.2]))

    return builder.build()


def example_1_basic_decorator():
    """Example 1: Using the @analyze_step decorator."""
    print("\n=== Example 1: Basic Decorator Usage ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()
    engine = DampingEngine(factor_graph=fg, damping_factor=0.7)

    # Define step function with analysis
    @analyze_step(
        return_state=True, config={"max_cycle_len": 8, "check_invariants": True}
    )
    def analyzed_step(eng, iteration=0):
        """Step function with automatic analysis."""
        return eng.step(iteration)

    print(f"Running analysis on {engine.__class__.__name__}")

    # Run several analyzed steps
    for i in range(5):
        step_result, analysis = analyzed_step(engine, i)

        # Extract key information
        region_fixed = analysis.get("region_fixed", False)
        has_certification = analysis.get("cycles", {}).get(
            "has_certified_contraction", False
        )
        aligned_hops = analysis.get("aligned_hops_total", 0)
        min_margin = analysis.get("margins", {}).get("min_margin")

        print(
            f"  Step {i}: Fixed={region_fixed}, Certified={has_certification}, "
            f"AlignedHops={aligned_hops}, MinMargin={min_margin:.6f if min_margin else 'None'}"
        )

        if region_fixed:
            print(f"  → Converged at step {i}")
            break


def example_2_manual_analysis():
    """Example 2: Manual analysis without decorators."""
    print("\n=== Example 2: Manual Analysis ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()
    engine = Engine(factor_graph=fg)

    print(f"Analyzing {engine.__class__.__name__}")

    # Run a few steps manually
    for i in range(3):
        engine.step(i)

    # Create snapshot and analyze
    snapshot = create_engine_snapshot(engine)

    # Configure comprehensive analysis
    config = {
        "max_cycle_len": 12,
        "compute_numeric_cycle_gain": True,
        "include_detailed_cycles": True,
        "check_invariants": True,
        "include_enforcement_suggestions": True,
    }

    analysis = run_diagnostics(snapshot, config)

    # Display results
    print(f"  Analysis complete: {analysis.get('analysis_complete', False)}")
    print(f"  Snapshot lambda: {analysis.get('snapshot_lambda', 'unknown')}")
    print(
        f"  Slots: Q={analysis.get('sizes', {}).get('nQ', 0)}, R={analysis.get('sizes', {}).get('nR', 0)}"
    )

    # Matrix norms
    norms = analysis.get("norms", {})
    if norms:
        print(f"  Matrix norms: {norms}")

    # Cycles
    cycles = analysis.get("cycles", {})
    print(
        f"  Cycles: {cycles.get('num_cycles', 0)} found, "
        f"{cycles.get('aligned_hops_total', 0)} aligned"
    )

    # Enforcement suggestion
    enforcement = analysis.get("enforcement_suggestion")
    if enforcement:
        hop = enforcement["enforce_hop"]
        print(
            f"  Enforcement suggested: factor={hop['fcopy']}, "
            f"var={hop['u']}, epsilon={enforcement['epsilon']:.6f}"
        )
    else:
        print("  No enforcement needed")


def example_3_convergence_monitoring():
    """Example 3: Comprehensive convergence monitoring."""
    print("\n=== Example 3: Convergence Monitoring ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()
    engine = DampingEngine(factor_graph=fg, damping_factor=0.8)

    print(f"Monitoring convergence of {engine.__class__.__name__}")

    # Monitor convergence with detailed config
    config = AnalysisConfig.comprehensive_config()
    results = monitor_convergence(engine, max_steps=25, config=config)

    # Display results
    converged = results.get("converged", False)
    total_steps = results.get("total_steps", 0)
    convergence_step = results.get("convergence_step")

    print(f"  Result: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}")
    print(f"  Steps analyzed: {total_steps}")

    if converged and convergence_step is not None:
        print(f"  Convergence achieved at step: {convergence_step}")

    # Show convergence trajectory (last few steps)
    trajectory = results.get("convergence_trajectory", [])
    if trajectory:
        print("  \nLast few steps:")
        print("    Step | Fixed | Certified | Aligned | Ties")
        for data in trajectory[-5:]:
            step = data["step"]
            fixed = "✓" if data["region_fixed"] else "✗"
            cert = "✓" if data["has_certification"] else "✗"
            aligned = data["aligned_hops"]
            ties = data["num_ties"]
            print(
                f"    {step:4d} |   {fixed}   |     {cert}     |   {aligned:3d}   | {ties:3d}"
            )


def example_4_analysis_wrapper():
    """Example 4: Using AnalysisWrapper for step-by-step control."""
    print("\n=== Example 4: Analysis Wrapper ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()
    engine = DampingEngine(factor_graph=fg, damping_factor=0.6)

    # Create analysis wrapper
    wrapper = AnalysisWrapper(engine, config={"max_cycle_len": 10})

    print(f"Using AnalysisWrapper on {engine.__class__.__name__}")

    # Run steps with detailed control
    for i in range(8):
        step_result, analysis = wrapper.step(i)

        # Check specific conditions
        region_fixed = analysis.get("region_fixed", False)
        enforcement = analysis.get("enforcement_suggestion")
        violations = analysis.get("invariant_violations", {})

        print(f"  Step {i}:")
        print(f"    Region fixed: {region_fixed}")
        print(f"    Enforcement needed: {'Yes' if enforcement else 'No'}")
        print(f"    Invariant violations: {violations.get('total_violations', 0)}")

        if region_fixed:
            print("    → Convergence detected, stopping")
            break

    # Get analysis history
    history = wrapper.get_analysis_history()
    print(f"  Total analyses performed: {len(history)}")


def example_5_comprehensive_report():
    """Example 5: Generate comprehensive analysis report."""
    print("\n=== Example 5: Comprehensive Report ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()
    engine = DampingEngine(factor_graph=fg, damping_factor=0.9)

    print(f"Generating comprehensive report for {engine.__class__.__name__}")

    # Generate report with comprehensive analysis
    report = create_comprehensive_report(engine, max_steps=15)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


def example_6_configuration_options():
    """Example 6: Different configuration options."""
    print("\n=== Example 6: Configuration Options ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()
    engine = Engine(factor_graph=fg)

    # Test different configuration presets
    presets = ["default", "fast", "comprehensive", "debug"]

    for preset_name in presets:
        print(f"\n  Testing {preset_name} configuration:")

        try:
            from src.analyzer import get_preset_config

            config = get_preset_config(preset_name)

            # Run single analysis
            engine.step(0)  # Execute one step
            snapshot = create_engine_snapshot(engine)
            analysis = run_diagnostics(snapshot, config.to_dict())

            # Show key results
            analysis_complete = analysis.get("analysis_complete", False)
            sizes = analysis.get("sizes", {})
            cycles = analysis.get("cycles", {})

            print(f"    Complete: {analysis_complete}")
            print(f"    Slots: Q={sizes.get('nQ', 0)}, R={sizes.get('nR', 0)}")
            print(f"    Cycles: {cycles.get('num_cycles', 0)}")

        except Exception as e:
            print(f"    Error: {e}")


def example_7_integration_patterns():
    """Example 7: Different integration patterns with existing code."""
    print("\n=== Example 7: Integration Patterns ===")

    # Create factor graph and engine
    fg = create_test_factor_graph()

    print("Pattern 1: Modify existing engine run method")

    class AnalyzingEngine(DampingEngine):
        """Example of engine subclass with integrated analysis."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.analysis_history = []

        @analyze_step(return_state=False)  # Return analysis only
        def analyzed_step(self, iteration=0):
            return self.step(iteration)

        def run_with_analysis(self, max_iter=10):
            """Modified run method with analysis."""
            print("    Running with integrated analysis...")

            for i in range(max_iter):
                analysis = self.analyzed_step(i)
                self.analysis_history.append(analysis)

                if analysis.get("region_fixed", False):
                    print(f"    Converged at iteration {i}")
                    break

            return self.analysis_history

    # Test the analyzing engine
    analyzing_engine = AnalyzingEngine(factor_graph=fg, damping_factor=0.7)
    history = analyzing_engine.run_with_analysis(max_iter=8)
    print(f"    Analysis history length: {len(history)}")

    print("\nPattern 2: Wrapper around existing engine")

    def create_analyzing_engine(base_engine):
        """Factory function to create analyzing version of any engine."""
        wrapper = AnalysisWrapper(base_engine)

        def run_until_convergence(max_steps=20):
            for i in range(max_steps):
                _, analysis = wrapper.step(i)
                if analysis.get("region_fixed", False):
                    print(f"    Wrapper detected convergence at step {i}")
                    return analysis
            print(f"    Wrapper did not converge in {max_steps} steps")
            return None

        wrapper.run_until_convergence = run_until_convergence
        return wrapper

    # Test wrapper pattern
    regular_engine = Engine(factor_graph=fg)
    analyzing_wrapper = create_analyzing_engine(regular_engine)
    final_analysis = analyzing_wrapper.run_until_convergence(max_steps=10)


def main():
    """Run all examples."""
    print("=== BELIEF PROPAGATION ANALYSIS FRAMEWORK EXAMPLES ===")
    print("Demonstrating integration with existing BPEngine code")

    examples = [
        example_1_basic_decorator,
        example_2_manual_analysis,
        example_3_convergence_monitoring,
        example_4_analysis_wrapper,
        example_5_comprehensive_report,
        example_6_configuration_options,
        example_7_integration_patterns,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example {example.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== INTEGRATION EXAMPLES COMPLETE ===")
    print("\nKey Integration Points:")
    print("1. @analyze_step decorator for automatic analysis")
    print("2. Manual run_diagnostics() for custom integration")
    print("3. AnalysisWrapper for existing engines")
    print("4. monitor_convergence() for convergence studies")
    print("5. Configuration presets for different analysis levels")
    print("6. Comprehensive reporting for detailed analysis")

    print("\nNext Steps:")
    print("• Integrate analysis into your existing simulation loops")
    print("• Use different configuration presets based on your needs")
    print("• Monitor convergence for long-running simulations")
    print("• Generate comprehensive reports for research")


if __name__ == "__main__":
    main()
