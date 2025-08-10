"""
Comprehensive tests for configuration validation functionality.

This module focuses specifically on testing the validation functions
and error handling in the centralized configuration system:
- Input validation for all configuration types
- Edge case handling and boundary conditions
- Error message clarity and specificity
- Type checking and conversion
- Integration with validation pipeline
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import patch

from propflow.configs.global_config_mapping import (
    validate_engine_config,
    validate_policy_config,
    validate_convergence_config,
    get_validated_config,
    ENGINE_DEFAULTS,
    POLICY_DEFAULTS,
    CONVERGENCE_DEFAULTS,
    SIMULATOR_DEFAULTS,
    LOGGING_CONFIG,
    SEARCH_DEFAULTS,
)


@pytest.mark.validation
class TestEngineConfigValidation:
    """Test validation for engine configurations."""

    def test_validate_engine_config_with_valid_defaults(self):
        """Test validation passes with default engine configuration."""
        result = validate_engine_config(ENGINE_DEFAULTS)
        assert result is True

    @pytest.mark.parametrize("max_iterations", [1, 10, 100, 1000, 10000])
    def test_validate_engine_max_iterations_valid_values(self, max_iterations):
        """Test validation with various valid max_iterations values."""
        config = ENGINE_DEFAULTS.copy()
        config["max_iterations"] = max_iterations

        result = validate_engine_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_max_iter", [0, -1, -100, 0.5, "10", None, float("inf")]
    )
    def test_validate_engine_max_iterations_invalid_values(self, invalid_max_iter):
        """Test validation rejects invalid max_iterations values."""
        config = ENGINE_DEFAULTS.copy()
        config["max_iterations"] = invalid_max_iter

        with pytest.raises(
            ValueError, match="max_iterations must be a positive integer"
        ):
            validate_engine_config(config)

    @pytest.mark.parametrize("timeout", [1, 60, 3600, 86400, 0.1, 10.5])
    def test_validate_engine_timeout_valid_values(self, timeout):
        """Test validation with various valid timeout values."""
        config = ENGINE_DEFAULTS.copy()
        config["timeout"] = timeout

        result = validate_engine_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_timeout", [0, -1, -10.5, "60", None, float("inf")]
    )
    def test_validate_engine_timeout_invalid_values(self, invalid_timeout):
        """Test validation rejects invalid timeout values."""
        config = ENGINE_DEFAULTS.copy()
        config["timeout"] = invalid_timeout

        with pytest.raises(ValueError, match="timeout must be a positive number"):
            validate_engine_config(config)

    def test_validate_engine_config_missing_required_keys(self):
        """Test validation fails when required keys are missing."""
        required_keys = [
            "max_iterations",
            "timeout",
            "normalize_messages",
            "monitor_performance",
            "anytime",
            "use_bct_history",
        ]

        for key_to_remove in required_keys:
            config = ENGINE_DEFAULTS.copy()
            del config[key_to_remove]

            with pytest.raises(
                ValueError, match=f"Missing required engine config key: {key_to_remove}"
            ):
                validate_engine_config(config)

    def test_validate_engine_config_extra_keys(self):
        """Test validation handles extra keys gracefully."""
        config = ENGINE_DEFAULTS.copy()
        config["extra_key"] = "extra_value"
        config["another_extra"] = 42

        # Should still pass validation (extra keys are allowed)
        result = validate_engine_config(config)
        assert result is True

    def test_validate_engine_config_boolean_fields(self):
        """Test validation of boolean fields."""
        boolean_fields = [
            "normalize_messages",
            "monitor_performance",
            "anytime",
            "use_bct_history",
        ]

        for field in boolean_fields:
            # Test valid boolean values
            for value in [True, False]:
                config = ENGINE_DEFAULTS.copy()
                config[field] = value
                result = validate_engine_config(config)
                assert result is True

        # Note: The current implementation doesn't validate boolean types,
        # so invalid values like strings or numbers would pass.
        # This could be enhanced in the future.


@pytest.mark.validation
class TestPolicyConfigValidation:
    """Test validation for policy configurations."""

    def test_validate_policy_config_with_valid_defaults(self):
        """Test validation passes with default policy configuration."""
        result = validate_policy_config(POLICY_DEFAULTS)
        assert result is True

    @pytest.mark.parametrize("damping_factor", [0.001, 0.1, 0.5, 0.9, 0.999, 1.0])
    def test_validate_policy_damping_factor_valid_values(self, damping_factor):
        """Test validation with valid damping factor values."""
        config = {"damping_factor": damping_factor}

        result = validate_policy_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_damping", [0.0, -0.1, -1.0, 1.1, 2.0, "0.5", None, float("inf")]
    )
    def test_validate_policy_damping_factor_invalid_values(self, invalid_damping):
        """Test validation rejects invalid damping factor values."""
        config = {"damping_factor": invalid_damping}

        with pytest.raises(
            ValueError, match="damping_factor must be in range \\(0, 1\\]"
        ):
            validate_policy_config(config)

    @pytest.mark.parametrize("split_factor", [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999])
    def test_validate_policy_split_factor_valid_values(self, split_factor):
        """Test validation with valid split factor values."""
        config = {"split_factor": split_factor}

        result = validate_policy_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_split", [0.0, -0.1, 1.0, 1.1, 2.0, "0.5", None, float("inf")]
    )
    def test_validate_policy_split_factor_invalid_values(self, invalid_split):
        """Test validation rejects invalid split factor values."""
        config = {"split_factor": invalid_split}

        with pytest.raises(
            ValueError, match="split_factor must be in range \\(0, 1\\)"
        ):
            validate_policy_config(config)

    @pytest.mark.parametrize("pruning_threshold", [0.0, 0.001, 0.1, 1.0, 10.0])
    def test_validate_policy_pruning_threshold_valid_values(self, pruning_threshold):
        """Test validation with valid pruning threshold values."""
        config = {"pruning_threshold": pruning_threshold}

        result = validate_policy_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_pruning", [-0.001, -1.0, "0.1", None, float("inf")]
    )
    def test_validate_policy_pruning_threshold_invalid_values(self, invalid_pruning):
        """Test validation rejects invalid pruning threshold values."""
        config = {"pruning_threshold": invalid_pruning}

        with pytest.raises(
            ValueError, match="pruning_threshold must be a non-negative number"
        ):
            validate_policy_config(config)

    def test_validate_policy_config_empty_config(self):
        """Test validation passes with empty policy config."""
        result = validate_policy_config({})
        assert result is True

    def test_validate_policy_config_partial_config(self):
        """Test validation with partial policy configurations."""
        partial_configs = [
            {"damping_factor": 0.8},
            {"split_factor": 0.3},
            {"pruning_threshold": 0.05},
            {"damping_factor": 0.7, "split_factor": 0.4},
            {"split_factor": 0.6, "pruning_threshold": 0.2},
            {"damping_factor": 0.9, "pruning_threshold": 0.1},
        ]

        for config in partial_configs:
            result = validate_policy_config(config)
            assert result is True

    def test_validate_policy_config_boundary_values(self):
        """Test validation with boundary values."""
        # Test exact boundary conditions
        boundary_tests = [
            {"damping_factor": 1.0},  # Upper bound (inclusive)
            {"split_factor": 0.999999},  # Just below upper bound
            {"pruning_threshold": 0.0},  # Lower bound (inclusive)
        ]

        for config in boundary_tests:
            result = validate_policy_config(config)
            assert result is True

        # Test just outside boundaries
        invalid_boundary_tests = [
            {"damping_factor": 0.0},  # Lower bound (exclusive)
            {"split_factor": 1.0},  # Upper bound (exclusive)
            {"pruning_threshold": -0.000001},  # Below lower bound
        ]

        for config in invalid_boundary_tests:
            with pytest.raises(ValueError):
                validate_policy_config(config)


@pytest.mark.validation
class TestConvergenceConfigValidation:
    """Test validation for convergence configurations."""

    def test_validate_convergence_config_with_valid_defaults(self):
        """Test validation passes with default convergence configuration."""
        result = validate_convergence_config(CONVERGENCE_DEFAULTS)
        assert result is True

    @pytest.mark.parametrize("belief_threshold", [1e-10, 1e-6, 1e-4, 0.01, 0.1, 1.0])
    def test_validate_convergence_belief_threshold_valid_values(self, belief_threshold):
        """Test validation with valid belief threshold values."""
        config = {"belief_threshold": belief_threshold}

        result = validate_convergence_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_threshold", [0, -1e-6, -0.1, "1e-6", None, float("inf")]
    )
    def test_validate_convergence_belief_threshold_invalid_values(
        self, invalid_threshold
    ):
        """Test validation rejects invalid belief threshold values."""
        config = {"belief_threshold": invalid_threshold}

        with pytest.raises(
            ValueError, match="belief_threshold must be a positive number"
        ):
            validate_convergence_config(config)

    @pytest.mark.parametrize("min_iterations", [0, 1, 5, 10, 100])
    def test_validate_convergence_min_iterations_valid_values(self, min_iterations):
        """Test validation with valid min_iterations values."""
        config = {"min_iterations": min_iterations}

        result = validate_convergence_config(config)
        assert result is True

    @pytest.mark.parametrize(
        "invalid_min_iter", [-1, -10, 1.5, "5", None, float("inf")]
    )
    def test_validate_convergence_min_iterations_invalid_values(self, invalid_min_iter):
        """Test validation rejects invalid min_iterations values."""
        config = {"min_iterations": invalid_min_iter}

        with pytest.raises(
            ValueError, match="min_iterations must be a non-negative integer"
        ):
            validate_convergence_config(config)

    @pytest.mark.parametrize("patience", [0, 1, 3, 5, 10])
    def test_validate_convergence_patience_valid_values(self, patience):
        """Test validation with valid patience values."""
        config = {"patience": patience}

        result = validate_convergence_config(config)
        assert result is True

    @pytest.mark.parametrize("invalid_patience", [-1, -5, 2.5, "3", None, float("inf")])
    def test_validate_convergence_patience_invalid_values(self, invalid_patience):
        """Test validation rejects invalid patience values."""
        config = {"patience": invalid_patience}

        with pytest.raises(ValueError, match="patience must be a non-negative integer"):
            validate_convergence_config(config)

    def test_validate_convergence_config_empty_config(self):
        """Test validation passes with empty convergence config."""
        result = validate_convergence_config({})
        assert result is True

    def test_validate_convergence_config_comprehensive(self):
        """Test validation with comprehensive convergence configurations."""
        comprehensive_configs = [
            {"belief_threshold": 1e-8, "min_iterations": 10, "patience": 5},
            {"belief_threshold": 0.001, "min_iterations": 0, "patience": 1},
            {"belief_threshold": 0.1, "min_iterations": 50, "patience": 0},
        ]

        for config in comprehensive_configs:
            result = validate_convergence_config(config)
            assert result is True


@pytest.mark.validation
class TestGetValidatedConfig:
    """Test the get_validated_config function comprehensive validation."""

    def test_get_validated_config_all_types(self):
        """Test get_validated_config with all configuration types."""
        config_types = [
            "engine",
            "policy",
            "convergence",
            "simulator",
            "logging",
            "search",
        ]

        for config_type in config_types:
            config = get_validated_config(config_type)
            assert isinstance(config, dict)
            assert len(config) > 0

    def test_get_validated_config_unknown_type(self):
        """Test get_validated_config with unknown configuration type."""
        with pytest.raises(ValueError, match="Unknown config type: unknown_type"):
            get_validated_config("unknown_type")

    def test_get_validated_config_engine_with_overrides(self):
        """Test get_validated_config for engine with various overrides."""
        test_overrides = [
            {"max_iterations": 500},
            {"timeout": 1800},
            {"normalize_messages": False},
            {"max_iterations": 1000, "timeout": 7200, "anytime": True},
        ]

        for override in test_overrides:
            config = get_validated_config("engine", override)

            # Should contain all default keys plus overrides
            for key in ENGINE_DEFAULTS:
                assert key in config

            # Overridden values should be present
            for key, value in override.items():
                assert config[key] == value

    def test_get_validated_config_policy_with_overrides(self):
        """Test get_validated_config for policy with various overrides."""
        test_overrides = [
            {"damping_factor": 0.7},
            {"split_factor": 0.3},
            {"pruning_threshold": 0.05},
            {"damping_factor": 0.8, "split_factor": 0.4},
        ]

        for override in test_overrides:
            config = get_validated_config("policy", override)

            # Should contain all default keys plus overrides
            for key in POLICY_DEFAULTS:
                assert key in config

            # Overridden values should be present
            for key, value in override.items():
                assert config[key] == value

    def test_get_validated_config_convergence_with_overrides(self):
        """Test get_validated_config for convergence with various overrides."""
        test_overrides = [
            {"belief_threshold": 1e-8},
            {"min_iterations": 10},
            {"patience": 2},
            {"belief_threshold": 1e-4, "patience": 5},
        ]

        for override in test_overrides:
            config = get_validated_config("convergence", override)

            # Should contain all default keys plus overrides
            for key in CONVERGENCE_DEFAULTS:
                assert key in config

            # Overridden values should be present
            for key, value in override.items():
                assert config[key] == value

    def test_get_validated_config_validation_enforcement(self):
        """Test that get_validated_config enforces validation."""
        # Test engine validation enforcement
        with pytest.raises(ValueError):
            get_validated_config("engine", {"max_iterations": -1})

        # Test policy validation enforcement
        with pytest.raises(ValueError):
            get_validated_config("policy", {"damping_factor": 2.0})

        # Test convergence validation enforcement
        with pytest.raises(ValueError):
            get_validated_config("convergence", {"belief_threshold": 0})

    def test_get_validated_config_non_validated_types(self):
        """Test get_validated_config with non-validated configuration types."""
        # These types currently don't have specific validators
        non_validated_types = ["simulator", "logging", "search"]

        for config_type in non_validated_types:
            # Should work without validation
            config = get_validated_config(config_type)
            assert isinstance(config, dict)

            # Should allow any overrides (no validation)
            config_with_override = get_validated_config(
                config_type, {"custom_key": "custom_value"}
            )
            assert config_with_override["custom_key"] == "custom_value"

    def test_get_validated_config_immutability(self):
        """Test that get_validated_config returns independent copies."""
        config1 = get_validated_config("engine")
        config2 = get_validated_config("engine")

        # Modify one config
        config1["max_iterations"] = 9999

        # Other config should be unchanged
        assert config2["max_iterations"] == ENGINE_DEFAULTS["max_iterations"]

    def test_get_validated_config_deep_copy_behavior(self):
        """Test that nested structures are properly copied."""
        config1 = get_validated_config("logging")
        config2 = get_validated_config("logging")

        # Modify nested dictionary if it exists
        if "console_colors" in config1 and isinstance(config1["console_colors"], dict):
            config1["console_colors"]["INFO"] = "modified_color"

            # Should not affect other config
            assert config2["console_colors"]["INFO"] != "modified_color"


@pytest.mark.validation
class TestValidationEdgeCases:
    """Test edge cases and unusual inputs in validation."""

    def test_validation_with_numpy_types(self):
        """Test validation with numpy numeric types."""
        # Test engine config with numpy types
        config = {
            "max_iterations": np.int32(100),
            "timeout": np.float64(3600.0),
            "normalize_messages": True,
            "monitor_performance": False,
            "anytime": False,
            "use_bct_history": False,
        }

        result = validate_engine_config(config)
        assert result is True

        # Test policy config with numpy types
        policy_config = {
            "damping_factor": np.float32(0.8),
            "split_factor": np.float64(0.5),
            "pruning_threshold": np.float32(0.1),
        }

        result = validate_policy_config(policy_config)
        assert result is True

    def test_validation_with_extreme_values(self):
        """Test validation with extreme but valid values."""
        # Very large max_iterations
        config = ENGINE_DEFAULTS.copy()
        config["max_iterations"] = 1000000
        result = validate_engine_config(config)
        assert result is True

        # Very small belief threshold
        conv_config = {"belief_threshold": 1e-20}
        result = validate_convergence_config(conv_config)
        assert result is True

        # Very large timeout
        eng_config = ENGINE_DEFAULTS.copy()
        eng_config["timeout"] = 1e6
        result = validate_engine_config(eng_config)
        assert result is True

    def test_validation_with_special_float_values(self):
        """Test validation behavior with special float values."""
        # Test with NaN
        config = ENGINE_DEFAULTS.copy()
        config["timeout"] = float("nan")

        # Should reject NaN values
        with pytest.raises(ValueError):
            validate_engine_config(config)

        # Test with infinity
        config["timeout"] = float("inf")
        with pytest.raises(ValueError):
            validate_engine_config(config)

        config["timeout"] = float("-inf")
        with pytest.raises(ValueError):
            validate_engine_config(config)

    def test_validation_error_message_specificity(self):
        """Test that validation error messages are specific and helpful."""
        # Test engine validation error messages
        config = ENGINE_DEFAULTS.copy()
        config["max_iterations"] = -5

        try:
            validate_engine_config(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max_iterations" in str(e)
            assert "positive integer" in str(e)

        # Test policy validation error messages
        policy_config = {"damping_factor": 1.5}

        try:
            validate_policy_config(policy_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "damping_factor" in str(e)
            assert "range" in str(e)

        # Test convergence validation error messages
        conv_config = {"belief_threshold": -1e-6}

        try:
            validate_convergence_config(conv_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "belief_threshold" in str(e)
            assert "positive" in str(e)

    def test_validation_with_mixed_valid_invalid(self):
        """Test validation with mixed valid and invalid parameters."""
        # Config with some valid and some invalid parameters
        config = {
            "max_iterations": 100,  # Valid
            "timeout": -1,  # Invalid
            "normalize_messages": True,  # Valid
            "monitor_performance": False,  # Valid
            "anytime": False,  # Valid
            "use_bct_history": False,  # Valid
        }

        # Should fail due to invalid timeout
        with pytest.raises(ValueError, match="timeout"):
            validate_engine_config(config)

    def test_validation_performance_with_large_configs(self):
        """Test validation performance with large configuration objects."""
        # Create config with many extra keys
        config = ENGINE_DEFAULTS.copy()
        for i in range(1000):
            config[f"extra_key_{i}"] = f"extra_value_{i}"

        # Should still validate quickly
        import time

        start_time = time.time()
        result = validate_engine_config(config)
        end_time = time.time()

        assert result is True
        assert (end_time - start_time) < 1.0  # Should be fast


@pytest.mark.validation
@pytest.mark.parametrize(
    "config_type,invalid_overrides",
    [
        (
            "engine",
            [
                {"max_iterations": 0},
                {"max_iterations": -1},
                {"timeout": 0},
                {"timeout": -1},
            ],
        ),
        (
            "policy",
            [
                {"damping_factor": 0.0},
                {"damping_factor": 1.1},
                {"split_factor": 0.0},
                {"split_factor": 1.0},
                {"pruning_threshold": -0.1},
            ],
        ),
        (
            "convergence",
            [
                {"belief_threshold": 0},
                {"belief_threshold": -1e-6},
                {"min_iterations": -1},
                {"patience": -1},
            ],
        ),
    ],
)
class TestValidationParametrized:
    """Parametrized tests for validation with various invalid configurations."""

    def test_validation_rejects_invalid_configs(self, config_type, invalid_overrides):
        """Test that validation properly rejects various invalid configurations."""
        for invalid_override in invalid_overrides:
            with pytest.raises(ValueError):
                get_validated_config(config_type, invalid_override)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
