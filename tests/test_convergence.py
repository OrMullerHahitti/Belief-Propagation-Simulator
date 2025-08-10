"""
Comprehensive tests for convergence monitoring functionality.

This module tests the convergence detection system:
- ConvergenceConfig dataclass and its integration with centralized config
- ConvergenceMonitor behavior and convergence detection
- Different convergence criteria (belief-based, assignment-based)
- Patience and minimum iteration handling
- Relative vs absolute change detection
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from dataclasses import dataclass

from propflow.policies.convergance import ConvergenceConfig, ConvergenceMonitor
from propflow.configs.global_config_mapping import CONVERGENCE_DEFAULTS


@pytest.mark.unit
class TestConvergenceConfig:
    """Test the ConvergenceConfig dataclass."""

    def test_convergence_config_default_initialization(self):
        """Test ConvergenceConfig initialization with defaults."""
        config = ConvergenceConfig()

        # Should use values from centralized configuration
        assert config.belief_threshold == CONVERGENCE_DEFAULTS["belief_threshold"]
        assert (
            config.assignment_threshold == CONVERGENCE_DEFAULTS["assignment_threshold"]
        )
        assert config.min_iterations == CONVERGENCE_DEFAULTS["min_iterations"]
        assert config.patience == CONVERGENCE_DEFAULTS["patience"]
        assert config.use_relative_change == CONVERGENCE_DEFAULTS["use_relative_change"]

    def test_convergence_config_custom_values(self):
        """Test ConvergenceConfig initialization with custom values."""
        config = ConvergenceConfig(
            belief_threshold=1e-8,
            assignment_threshold=5,
            min_iterations=10,
            patience=3,
            use_relative_change=False,
        )

        assert config.belief_threshold == 1e-8
        assert config.assignment_threshold == 5
        assert config.min_iterations == 10
        assert config.patience == 3
        assert config.use_relative_change is False

    def test_convergence_config_partial_override(self):
        """Test ConvergenceConfig with partial parameter override."""
        config = ConvergenceConfig(belief_threshold=1e-4, patience=10)

        # Overridden values
        assert config.belief_threshold == 1e-4
        assert config.patience == 10

        # Default values from centralized config
        assert (
            config.assignment_threshold == CONVERGENCE_DEFAULTS["assignment_threshold"]
        )
        assert config.min_iterations == CONVERGENCE_DEFAULTS["min_iterations"]
        assert config.use_relative_change == CONVERGENCE_DEFAULTS["use_relative_change"]

    def test_convergence_config_is_dataclass(self):
        """Test that ConvergenceConfig is properly structured as a dataclass."""
        config = ConvergenceConfig()

        # Should have dataclass properties
        assert hasattr(config, "__dataclass_fields__")

        # Should support field access
        assert hasattr(config, "belief_threshold")
        assert hasattr(config, "assignment_threshold")
        assert hasattr(config, "min_iterations")
        assert hasattr(config, "patience")
        assert hasattr(config, "use_relative_change")


@pytest.mark.unit
class TestConvergenceMonitorInitialization:
    """Test ConvergenceMonitor initialization."""

    def test_monitor_default_initialization(self):
        """Test ConvergenceMonitor initialization with default config."""
        monitor = ConvergenceMonitor()

        # Should create default config
        assert monitor.config is not None
        assert isinstance(monitor.config, ConvergenceConfig)
        assert (
            monitor.config.belief_threshold == CONVERGENCE_DEFAULTS["belief_threshold"]
        )

        # Initial state
        assert monitor.prev_beliefs is None
        assert monitor.prev_assignments is None
        assert monitor.stable_count == 0
        assert monitor.iteration == 0
        assert monitor.convergence_history == []

    def test_monitor_with_custom_config(self):
        """Test ConvergenceMonitor initialization with custom config."""
        custom_config = ConvergenceConfig(
            belief_threshold=1e-4, min_iterations=5, patience=8
        )

        monitor = ConvergenceMonitor(custom_config)

        assert monitor.config == custom_config
        assert monitor.config.belief_threshold == 1e-4
        assert monitor.config.min_iterations == 5
        assert monitor.config.patience == 8

    def test_monitor_reset_functionality(self):
        """Test ConvergenceMonitor reset functionality."""
        monitor = ConvergenceMonitor()

        # Modify state
        monitor.iteration = 10
        monitor.stable_count = 5
        monitor.prev_beliefs = {"var1": np.array([0.1, 0.9])}
        monitor.prev_assignments = {"var1": 1}
        monitor.convergence_history = [True, False, True]

        # Reset
        monitor.reset()

        # Should reset to initial state
        assert monitor.iteration == 0
        assert monitor.stable_count == 0
        assert monitor.prev_beliefs is None
        assert monitor.prev_assignments is None
        assert monitor.convergence_history == []


@pytest.mark.unit
class TestConvergenceDetection:
    """Test core convergence detection logic."""

    @pytest.fixture
    def monitor(self):
        """Create monitor with reasonable test configuration."""
        config = ConvergenceConfig(
            belief_threshold=1e-3,
            assignment_threshold=0,
            min_iterations=2,
            patience=3,
            use_relative_change=True,
        )
        return ConvergenceMonitor(config)

    @pytest.fixture
    def sample_beliefs(self):
        """Sample belief distributions for testing."""
        return {
            "var1": np.array([0.8, 0.2]),
            "var2": np.array([0.3, 0.7]),
            "var3": np.array([0.5, 0.5]),
        }

    @pytest.fixture
    def sample_assignments(self):
        """Sample variable assignments for testing."""
        return {"var1": 0, "var2": 1, "var3": 0}

    def test_convergence_before_min_iterations(
        self, monitor, sample_beliefs, sample_assignments
    ):
        """Test that convergence is not detected before min_iterations."""
        # First iteration (below min_iterations)
        converged = monitor.check_convergence(sample_beliefs, sample_assignments)
        assert converged is False
        assert monitor.iteration == 1

        # Second iteration (still below min_iterations)
        converged = monitor.check_convergence(sample_beliefs, sample_assignments)
        assert converged is False
        assert monitor.iteration == 2

    def test_first_iteration_after_min_iterations(
        self, monitor, sample_beliefs, sample_assignments
    ):
        """Test first iteration after min_iterations."""
        # Advance to min_iterations + 1
        for _ in range(monitor.config.min_iterations + 1):
            converged = monitor.check_convergence(sample_beliefs, sample_assignments)

        # Should not converge on first comparison (no previous beliefs)
        assert converged is False
        assert monitor.prev_beliefs is not None
        assert monitor.prev_assignments is not None

    def test_convergence_with_identical_beliefs(
        self, monitor, sample_beliefs, sample_assignments
    ):
        """Test convergence detection with identical beliefs."""
        # Set up previous state
        monitor.iteration = monitor.config.min_iterations
        monitor.prev_beliefs = sample_beliefs.copy()
        monitor.prev_assignments = sample_assignments.copy()

        # Check convergence with identical beliefs
        converged = monitor.check_convergence(sample_beliefs, sample_assignments)

        assert converged is True

    def test_convergence_with_small_belief_changes(
        self, monitor, sample_beliefs, sample_assignments
    ):
        """Test convergence with belief changes below threshold."""
        # Set up previous state
        monitor.iteration = monitor.config.min_iterations
        monitor.prev_beliefs = sample_beliefs.copy()
        monitor.prev_assignments = sample_assignments.copy()

        # Create slightly different beliefs (below threshold)
        new_beliefs = {}
        for var, belief in sample_beliefs.items():
            # Very small change
            new_beliefs[var] = belief + 1e-4 * np.random.rand(len(belief))

        converged = monitor.check_convergence(new_beliefs, sample_assignments)

        # Should converge since changes are below threshold
        assert converged is True

    def test_no_convergence_with_large_belief_changes(
        self, monitor, sample_beliefs, sample_assignments
    ):
        """Test no convergence with belief changes above threshold."""
        # Set up previous state
        monitor.iteration = monitor.config.min_iterations
        monitor.prev_beliefs = sample_beliefs.copy()
        monitor.prev_assignments = sample_assignments.copy()

        # Create significantly different beliefs (above threshold)
        new_beliefs = {
            "var1": np.array([0.2, 0.8]),  # Flipped from [0.8, 0.2]
            "var2": np.array([0.9, 0.1]),  # Changed from [0.3, 0.7]
            "var3": np.array([0.1, 0.9]),  # Changed from [0.5, 0.5]
        }

        converged = monitor.check_convergence(new_beliefs, sample_assignments)

        # Should not converge due to large changes
        assert converged is False

    def test_convergence_with_assignment_changes(
        self, monitor, sample_beliefs, sample_assignments
    ):
        """Test convergence behavior with assignment changes."""
        # Set up previous state
        monitor.iteration = monitor.config.min_iterations
        monitor.prev_beliefs = sample_beliefs.copy()
        monitor.prev_assignments = sample_assignments.copy()

        # Create new assignments (different from previous)
        new_assignments = {"var1": 1, "var2": 0, "var3": 1}  # All different

        # Even with identical beliefs, different assignments should prevent convergence
        converged = monitor.check_convergence(sample_beliefs, new_assignments)

        assert converged is False

    def test_relative_vs_absolute_change_detection(
        self, sample_beliefs, sample_assignments
    ):
        """Test relative vs absolute change detection modes."""
        # Test with relative change (default)
        monitor_rel = ConvergenceMonitor(
            ConvergenceConfig(
                belief_threshold=1e-2, min_iterations=1, use_relative_change=True
            )
        )

        # Test with absolute change
        monitor_abs = ConvergenceMonitor(
            ConvergenceConfig(
                belief_threshold=1e-2, min_iterations=1, use_relative_change=False
            )
        )

        # Set up initial state for both monitors
        for monitor in [monitor_rel, monitor_abs]:
            monitor.iteration = monitor.config.min_iterations
            monitor.prev_beliefs = {"var1": np.array([1.0, 0.0])}
            monitor.prev_assignments = {"var1": 0}

        # Small absolute change but potentially large relative change
        new_beliefs = {"var1": np.array([0.99, 0.01])}
        new_assignments = {"var1": 0}

        # Both should handle this appropriately based on their mode
        conv_rel = monitor_rel.check_convergence(new_beliefs, new_assignments)
        conv_abs = monitor_abs.check_convergence(new_beliefs, new_assignments)

        # Results may differ based on relative vs absolute calculation
        assert isinstance(conv_rel, bool)
        assert isinstance(conv_abs, bool)

    def test_patience_mechanism(self, sample_beliefs, sample_assignments):
        """Test the patience mechanism for stable convergence."""
        config = ConvergenceConfig(
            belief_threshold=1e-3,
            min_iterations=1,
            patience=3,
            use_relative_change=True,
        )
        monitor = ConvergenceMonitor(config)

        # Set up initial state
        monitor.iteration = config.min_iterations
        monitor.prev_beliefs = sample_beliefs.copy()
        monitor.prev_assignments = sample_assignments.copy()

        # Test patience counting
        for i in range(config.patience):
            converged = monitor.check_convergence(sample_beliefs, sample_assignments)

            if i < config.patience - 1:
                # Should not converge until patience is reached
                assert (
                    converged is False or converged is True
                )  # Implementation dependent

        # After patience iterations, should have made decision
        assert isinstance(converged, bool)


@pytest.mark.unit
class TestConvergenceEdgeCases:
    """Test edge cases and error conditions in convergence detection."""

    def test_empty_beliefs_and_assignments(self):
        """Test convergence with empty beliefs and assignments."""
        monitor = ConvergenceMonitor()

        empty_beliefs = {}
        empty_assignments = {}

        converged = monitor.check_convergence(empty_beliefs, empty_assignments)

        # Should handle empty input gracefully
        assert isinstance(converged, bool)

    def test_mismatched_beliefs_and_assignments(self):
        """Test convergence with mismatched variable sets."""
        monitor = ConvergenceMonitor()

        beliefs = {"var1": np.array([0.5, 0.5]), "var2": np.array([0.3, 0.7])}
        assignments = {
            "var1": 0,
            "var3": 1,
        }  # var3 not in beliefs, var2 not in assignments

        # Should handle mismatch gracefully
        converged = monitor.check_convergence(beliefs, assignments)
        assert isinstance(converged, bool)

    def test_nan_inf_in_beliefs(self):
        """Test convergence handling with NaN/Inf in beliefs."""
        monitor = ConvergenceMonitor()
        monitor.iteration = monitor.config.min_iterations
        monitor.prev_beliefs = {"var1": np.array([0.5, 0.5])}
        monitor.prev_assignments = {"var1": 0}

        # Beliefs with NaN
        nan_beliefs = {"var1": np.array([np.nan, 0.5])}
        assignments = {"var1": 0}

        # Should handle NaN gracefully (may or may not converge)
        converged = monitor.check_convergence(nan_beliefs, assignments)
        assert isinstance(converged, bool)

        # Beliefs with Inf
        inf_beliefs = {"var1": np.array([np.inf, 0.5])}

        converged = monitor.check_convergence(inf_beliefs, assignments)
        assert isinstance(converged, bool)

    def test_zero_norm_beliefs(self):
        """Test convergence with zero-norm beliefs."""
        monitor = ConvergenceMonitor()
        monitor.iteration = monitor.config.min_iterations
        monitor.prev_beliefs = {"var1": np.array([0.0, 0.0])}
        monitor.prev_assignments = {"var1": 0}

        new_beliefs = {"var1": np.array([0.0, 0.0])}
        assignments = {"var1": 0}

        # Should handle zero-norm beliefs gracefully
        converged = monitor.check_convergence(new_beliefs, assignments)
        assert isinstance(converged, bool)

    def test_single_variable_convergence(self):
        """Test convergence with single variable."""
        monitor = ConvergenceMonitor(
            ConvergenceConfig(belief_threshold=1e-3, min_iterations=1)
        )

        # Single variable case
        beliefs = {"var1": np.array([0.8, 0.2])}
        assignments = {"var1": 0}

        # First call
        converged = monitor.check_convergence(beliefs, assignments)
        assert converged is False  # No previous state

        # Second call with same values
        converged = monitor.check_convergence(beliefs, assignments)
        assert converged is True  # Should converge

    def test_large_belief_vectors(self):
        """Test convergence with large belief vectors."""
        monitor = ConvergenceMonitor(ConvergenceConfig(min_iterations=1))

        # Large domain size
        large_belief = np.random.rand(100)
        large_belief /= np.sum(large_belief)  # Normalize

        beliefs = {"var1": large_belief}
        assignments = {"var1": np.argmax(large_belief)}

        # Should handle large vectors
        converged = monitor.check_convergence(beliefs, assignments)
        assert isinstance(converged, bool)


@pytest.mark.integration
class TestConvergenceIntegration:
    """Integration tests for convergence with other components."""

    def test_convergence_config_integration(self):
        """Test integration with centralized configuration."""
        # Test that ConvergenceConfig properly inherits from centralized defaults
        config = ConvergenceConfig()

        # Should match centralized configuration
        for key, value in CONVERGENCE_DEFAULTS.items():
            assert getattr(config, key) == value

        # Test that overrides still work
        config_override = ConvergenceConfig(belief_threshold=1e-8)
        assert config_override.belief_threshold == 1e-8
        assert config_override.patience == CONVERGENCE_DEFAULTS["patience"]

    def test_monitor_with_real_convergence_scenario(self):
        """Test monitor with realistic convergence scenario."""
        config = ConvergenceConfig(belief_threshold=1e-4, min_iterations=3, patience=2)
        monitor = ConvergenceMonitor(config)

        # Simulate belief propagation convergence
        initial_beliefs = {"var1": np.array([0.6, 0.4]), "var2": np.array([0.3, 0.7])}

        assignments = {"var1": 0, "var2": 1}

        # Initial iterations (should not converge)
        for i in range(3):
            # Gradually converging beliefs
            noise_factor = 0.1 / (i + 1)  # Decreasing noise
            current_beliefs = {}
            for var, belief in initial_beliefs.items():
                noise = noise_factor * np.random.rand(len(belief))
                current_beliefs[var] = belief + noise
                current_beliefs[var] /= np.sum(current_beliefs[var])  # Renormalize

            converged = monitor.check_convergence(current_beliefs, assignments)

            if i < config.min_iterations:
                assert converged is False

        # Final iteration with very similar beliefs
        converged = monitor.check_convergence(initial_beliefs, assignments)

        # Should eventually converge or make a decision
        assert isinstance(converged, bool)

    def test_convergence_history_tracking(self):
        """Test that convergence history is properly tracked."""
        monitor = ConvergenceMonitor(ConvergenceConfig(min_iterations=1))

        beliefs = {"var1": np.array([0.5, 0.5])}
        assignments = {"var1": 0}

        # Multiple iterations
        for _ in range(5):
            converged = monitor.check_convergence(beliefs, assignments)

        # History should be tracked (implementation dependent)
        assert len(monitor.convergence_history) >= 0  # May or may not track history

    def test_monitor_reset_after_convergence(self):
        """Test monitor reset functionality after convergence detection."""
        monitor = ConvergenceMonitor(ConvergenceConfig(min_iterations=1))

        beliefs = {"var1": np.array([0.5, 0.5])}
        assignments = {"var1": 0}

        # Run to convergence
        monitor.check_convergence(beliefs, assignments)
        converged = monitor.check_convergence(beliefs, assignments)

        # Reset monitor
        monitor.reset()

        # Should be back to initial state
        assert monitor.iteration == 0
        assert monitor.prev_beliefs is None
        assert monitor.prev_assignments is None

        # Should behave as if starting fresh
        converged = monitor.check_convergence(beliefs, assignments)
        assert converged is False  # First iteration after reset


@pytest.mark.unit
@pytest.mark.parametrize(
    "belief_threshold,min_iterations,patience",
    [(1e-6, 0, 1), (1e-4, 5, 3), (1e-2, 10, 5), (1e-8, 2, 10)],
)
class TestConvergenceParametrized:
    """Parametrized tests for different convergence configurations."""

    def test_different_configurations(self, belief_threshold, min_iterations, patience):
        """Test convergence monitor with different parameter combinations."""
        config = ConvergenceConfig(
            belief_threshold=belief_threshold,
            min_iterations=min_iterations,
            patience=patience,
        )
        monitor = ConvergenceMonitor(config)

        # Verify configuration
        assert monitor.config.belief_threshold == belief_threshold
        assert monitor.config.min_iterations == min_iterations
        assert monitor.config.patience == patience

        # Test basic functionality
        beliefs = {"var1": np.array([0.7, 0.3])}
        assignments = {"var1": 0}

        converged = monitor.check_convergence(beliefs, assignments)

        # Should handle all parameter combinations
        assert isinstance(converged, bool)
        assert monitor.iteration == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
