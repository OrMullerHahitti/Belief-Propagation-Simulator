"""
Analysis decorator for belief propagation step functions.

Provides the @analyze_step decorator that wraps simulator step functions
to automatically perform diagnostic analysis after each iteration.
"""

import functools
from typing import Dict, Any, Optional, Callable, Union, Tuple
from .snapshot import Snapshot, create_engine_snapshot, BPEngineSnapshot
from .diagnostics import run_diagnostics
from .config import AnalysisConfig, validate_config


def analyze_step(
    return_state: bool = True,
    config: Optional[Union[Dict[str, Any], AnalysisConfig]] = None,
):
    """
    Decorator that wraps a simulator step function with diagnostic analysis.

    After running the step, computes diagnostics on the returned snapshot(s)
    and returns either (state, analysis) or just analysis based on return_state.

    Usage:
        @analyze_step(return_state=True, config={"max_cycle_len": 10})
        def step(engine, iteration=0):
            # Your normal step logic
            return engine.step(iteration)

        state, analysis = step(engine, 5)

    Args:
        return_state: If True, return (state, analysis). If False, return analysis only.
        config: Analysis configuration (dict or AnalysisConfig object)

    Returns:
        Decorator function
    """

    def decorator(step_fn: Callable) -> Callable:
        @functools.wraps(step_fn)
        def wrapper(*args, **kwargs) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
            # Execute the original step function
            state = step_fn(*args, **kwargs)

            # Extract or create snapshot from the returned state
            snap = extract_snapshot_from_state(state, args, kwargs)

            if snap is None:
                # Could not create snapshot - return original result
                if return_state:
                    return state, {"error": "Could not extract snapshot from state"}
                else:
                    return {"error": "Could not extract snapshot from state"}

            # Get previous analysis for region detection
            prev_analysis = (
                getattr(state, "_prev_diag", None)
                if hasattr(state, "__dict__")
                else None
            )

            # Run diagnostic analysis
            analysis = run_diagnostics(snap, config, prev=prev_analysis)

            # Store analysis for next iteration (if state supports it)
            if hasattr(state, "__dict__"):
                state._prev_diag = analysis.get("region_keys", None)

            # Return based on return_state flag
            if return_state:
                return state, analysis
            else:
                return analysis

        return wrapper

    return decorator


def extract_snapshot_from_state(
    state: Any, args: tuple, kwargs: Dict[str, Any]
) -> Optional[Snapshot]:
    """
    Extract or create a snapshot from the step function's return value and arguments.

    Args:
        state: Return value from step function
        args: Arguments passed to step function
        kwargs: Keyword arguments passed to step function

    Returns:
        Snapshot object or None if extraction failed
    """
    # Method 1: State has a snapshot attribute
    if hasattr(state, "snapshot") and isinstance(state.snapshot, Snapshot):
        return state.snapshot

    # Method 2: State is itself a snapshot
    if isinstance(state, Snapshot):
        return state

    # Method 3: Try to extract from engine in arguments
    engine = None

    # Look for engine in args (typically first argument)
    if args:
        potential_engine = args[0]
        if hasattr(potential_engine, "graph") and hasattr(
            potential_engine, "var_nodes"
        ):
            engine = potential_engine

    # Look for engine in kwargs
    if engine is None:
        for key in ["engine", "bp_engine", "simulator"]:
            if key in kwargs:
                potential_engine = kwargs[key]
                if hasattr(potential_engine, "graph") and hasattr(
                    potential_engine, "var_nodes"
                ):
                    engine = potential_engine
                    break

    # Method 4: State contains engine reference
    if engine is None and hasattr(state, "engine"):
        engine = state.engine

    # If we found an engine, create snapshot from it
    if engine is not None:
        try:
            # Try to detect damping factor
            damping_factor = getattr(engine, "damping_factor", 0.0)
            return create_engine_snapshot(engine, damping_factor)
        except Exception:
            # Fallback: try without damping factor detection
            try:
                return BPEngineSnapshot(engine, 0.0)
            except Exception:
                pass

    return None


class AnalysisWrapper:
    """
    Wrapper class that adds analysis capabilities to existing engines.

    This can be used when you can't modify the original step function
    but want to add analysis capabilities.
    """

    def __init__(
        self, engine, config: Optional[Union[Dict[str, Any], AnalysisConfig]] = None
    ):
        """
        Initialize analysis wrapper.

        Args:
            engine: The BP engine to wrap
            config: Analysis configuration
        """
        self.engine = engine
        self.config = validate_config(config) if config else AnalysisConfig()
        self._prev_analysis = None
        self._analysis_history = []

    def step(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute engine step with analysis.

        Returns:
            Tuple of (step_result, analysis)
        """
        # Execute original step
        step_result = self.engine.step(*args, **kwargs)

        # Create snapshot
        snapshot = create_engine_snapshot(self.engine)

        # Run analysis
        analysis = run_diagnostics(
            snapshot, self.config.to_dict(), prev=self._prev_analysis
        )

        # Store for next iteration
        self._prev_analysis = analysis.get("region_keys", None)
        self._analysis_history.append(analysis)

        return step_result, analysis

    def run_with_analysis(self, max_iterations: int = 50) -> Dict[str, Any]:
        """
        Run multiple steps with analysis tracking.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary with complete run analysis
        """
        iteration_analyses = []

        for i in range(max_iterations):
            step_result, analysis = self.step(i)
            iteration_analyses.append(analysis)

            # Check for convergence
            if analysis.get("region_fixed", False):
                break

        # Compile summary
        run_summary = {
            "total_iterations": len(iteration_analyses),
            "converged": iteration_analyses[-1].get("region_fixed", False)
            if iteration_analyses
            else False,
            "final_analysis": iteration_analyses[-1] if iteration_analyses else None,
            "iteration_analyses": iteration_analyses,
            "convergence_iteration": None,
        }

        # Find convergence iteration
        for i, analysis in enumerate(iteration_analyses):
            if analysis.get("region_fixed", False):
                run_summary["convergence_iteration"] = i
                break

        return run_summary

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of all analyses performed."""
        return self._analysis_history.copy()

    def clear_history(self):
        """Clear analysis history."""
        self._analysis_history.clear()
        self._prev_analysis = None


def create_analyzing_step_function(
    original_step_fn: Callable,
    config: Optional[Union[Dict[str, Any], AnalysisConfig]] = None,
) -> Callable:
    """
    Create an analyzing version of a step function without using decorators.

    This is useful when you can't modify the original function definition
    but want to add analysis capabilities.

    Args:
        original_step_fn: The original step function
        config: Analysis configuration

    Returns:
        New function that includes analysis
    """

    @analyze_step(return_state=True, config=config)
    def analyzing_step(*args, **kwargs):
        return original_step_fn(*args, **kwargs)

    return analyzing_step


def batch_analyze_steps(
    engine,
    num_steps: int = 10,
    config: Optional[Union[Dict[str, Any], AnalysisConfig]] = None,
) -> List[Dict[str, Any]]:
    """
    Run multiple steps with analysis on an engine.

    Args:
        engine: BP engine to analyze
        num_steps: Number of steps to run
        config: Analysis configuration

    Returns:
        List of analysis results, one per step
    """
    wrapper = AnalysisWrapper(engine, config)
    analyses = []

    for i in range(num_steps):
        _, analysis = wrapper.step(i)
        analyses.append(analysis)

        # Stop if converged
        if analysis.get("region_fixed", False):
            break

    return analyses


def monitor_convergence(
    engine,
    max_steps: int = 100,
    config: Optional[Union[Dict[str, Any], AnalysisConfig]] = None,
) -> Dict[str, Any]:
    """
    Monitor an engine until convergence or max_steps reached.

    Args:
        engine: BP engine to monitor
        max_steps: Maximum steps to run
        config: Analysis configuration

    Returns:
        Convergence monitoring results
    """
    wrapper = AnalysisWrapper(engine, config)

    convergence_data = {
        "converged": False,
        "convergence_step": None,
        "total_steps": 0,
        "final_analysis": None,
        "convergence_trajectory": [],
    }

    for step in range(max_steps):
        _, analysis = wrapper.step(step)
        convergence_data["total_steps"] = step + 1
        convergence_data["final_analysis"] = analysis

        # Track convergence indicators
        convergence_indicators = {
            "step": step,
            "region_fixed": analysis.get("region_fixed", False),
            "has_certification": analysis.get("cycles", {}).get(
                "has_certified_contraction", False
            ),
            "aligned_hops": analysis.get("aligned_hops_total", 0),
            "min_margin": analysis.get("margins", {}).get("min_margin"),
            "num_ties": analysis.get("margins", {}).get("num_ties", 0),
        }
        convergence_data["convergence_trajectory"].append(convergence_indicators)

        # Check convergence
        if analysis.get("region_fixed", False):
            convergence_data["converged"] = True
            convergence_data["convergence_step"] = step
            break

    return convergence_data


# Example usage and testing
if __name__ == "__main__":
    print("=== Analysis Decorator Examples ===")

    # Example 1: Decorator usage
    print("Example 1: Using @analyze_step decorator")

    @analyze_step(return_state=True, config={"max_cycle_len": 8})
    def example_step(engine, iteration=0):
        """Example step function with analysis."""
        # Simulate step execution
        result = (
            engine.step(iteration) if hasattr(engine, "step") else f"step_{iteration}"
        )
        return result

    # Example 2: Wrapper class usage
    print("\nExample 2: Using AnalysisWrapper class")

    class MockEngine:
        def __init__(self):
            self.iteration = 0
            self.graph = "mock_graph"
            self.var_nodes = []
            self.factor_nodes = []

        def step(self, i=0):
            self.iteration = i
            return f"mock_step_{i}"

    mock_engine = MockEngine()

    # Test analysis wrapper
    try:
        wrapper = AnalysisWrapper(mock_engine)
        print("✓ AnalysisWrapper created successfully")

        # Note: This would fail without a real engine, but demonstrates the API
        print("  (Analysis would run on real engine)")

    except Exception as e:
        print(f"Expected error with mock engine: {type(e).__name__}")

    # Example 3: Function creation
    print("\nExample 3: Creating analyzing function")

    def original_step(engine, iteration):
        return f"original_step_{iteration}"

    analyzing_step = create_analyzing_step_function(
        original_step, config={"check_invariants": True}
    )

    print("✓ Analyzing step function created")

    print("\n=== Decorator ready for integration ===")
    print("Usage patterns:")
    print("1. @analyze_step decorator on existing functions")
    print("2. AnalysisWrapper for engines")
    print("3. Batch analysis and convergence monitoring")
    print("4. Integration with existing simulation loops")
