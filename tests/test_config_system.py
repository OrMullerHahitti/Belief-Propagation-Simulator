"""
Comprehensive tests for the centralized configuration system.

This module tests the global_config_mapping functionality, including:
- Default configuration values
- Configuration validation
- Configuration merging and overrides
- Type safety and error handling
"""

import pytest
import logging
from typing import Dict, Any
from unittest.mock import patch

from propflow.configs.global_config_mapping import (
    # Configuration dictionaries
    ENGINE_DEFAULTS,
    POLICY_DEFAULTS,
    CONVERGENCE_DEFAULTS,
    SIMULATOR_DEFAULTS,
    LOGGING_CONFIG,
    LOG_LEVELS,
    SEARCH_DEFAULTS,
    # Validation functions
    validate_engine_config,
    validate_policy_config,
    validate_convergence_config,
    get_validated_config,
    # Constants
    MESSAGE_DOMAIN_SIZE,
    PROJECT_ROOT,
    Dirs,
    # Legacy support
    VERBOSE_LOGGING,
)


@pytest.mark.config
class TestConfigurationDefaults:
    """Test that all default configurations are properly structured."""

    def test_engine_defaults_structure(self):
        """Test that ENGINE_DEFAULTS contains all required keys with correct types."""
        expected_keys = {
            "max_iterations",
            "timeout",
            "normalize_messages",
            "monitor_performance",
            "anytime",
            "use_bct_history",
        }

        assert set(ENGINE_DEFAULTS.keys()) == expected_keys
        assert isinstance(ENGINE_DEFAULTS["max_iterations"], int)
        assert isinstance(ENGINE_DEFAULTS["timeout"], (int, float))
        assert isinstance(ENGINE_DEFAULTS["normalize_messages"], bool)
        assert isinstance(ENGINE_DEFAULTS["monitor_performance"], bool)
        assert isinstance(ENGINE_DEFAULTS["anytime"], bool)
        assert isinstance(ENGINE_DEFAULTS["use_bct_history"], bool)

    def test_engine_defaults_values(self):
        """Test that ENGINE_DEFAULTS values are reasonable."""
        assert ENGINE_DEFAULTS["max_iterations"] > 0
        assert ENGINE_DEFAULTS["timeout"] > 0
        assert isinstance(ENGINE_DEFAULTS["normalize_messages"], bool)

    def test_policy_defaults_structure(self):
        """Test that POLICY_DEFAULTS contains all expected keys."""
        expected_keys = {
            "damping_factor",
            "damping_diameter",
            "split_factor",
            "pruning_threshold",
            "pruning_magnitude_factor",
            "cost_reduction_enabled",
        }

        assert set(POLICY_DEFAULTS.keys()) == expected_keys

    def test_policy_defaults_values(self):
        """Test that POLICY_DEFAULTS values are within valid ranges."""
        assert 0.0 < POLICY_DEFAULTS["damping_factor"] <= 1.0
        assert POLICY_DEFAULTS["damping_diameter"] >= 1
        assert 0.0 < POLICY_DEFAULTS["split_factor"] < 1.0
        assert POLICY_DEFAULTS["pruning_threshold"] >= 0
        assert POLICY_DEFAULTS["pruning_magnitude_factor"] >= 0
        assert isinstance(POLICY_DEFAULTS["cost_reduction_enabled"], bool)

    def test_convergence_defaults_structure(self):
        """Test that CONVERGENCE_DEFAULTS contains all expected keys."""
        expected_keys = {
            "belief_threshold",
            "assignment_threshold",
            "min_iterations",
            "patience",
            "use_relative_change",
        }

        assert set(CONVERGENCE_DEFAULTS.keys()) == expected_keys

    def test_convergence_defaults_values(self):
        """Test that CONVERGENCE_DEFAULTS values are reasonable."""
        assert CONVERGENCE_DEFAULTS["belief_threshold"] > 0
        assert CONVERGENCE_DEFAULTS["assignment_threshold"] >= 0
        assert CONVERGENCE_DEFAULTS["min_iterations"] >= 0
        assert CONVERGENCE_DEFAULTS["patience"] >= 0
        assert isinstance(CONVERGENCE_DEFAULTS["use_relative_change"], bool)

    def test_simulator_defaults_structure(self):
        """Test that SIMULATOR_DEFAULTS contains all expected keys."""
        expected_keys = {
            "default_max_iter",
            "default_log_level",
            "timeout",
            "cpu_count_multiplier",
        }

        assert set(SIMULATOR_DEFAULTS.keys()) == expected_keys

    def test_simulator_defaults_values(self):
        """Test that SIMULATOR_DEFAULTS values are reasonable."""
        assert SIMULATOR_DEFAULTS["default_max_iter"] > 0
        assert SIMULATOR_DEFAULTS["default_log_level"] in LOG_LEVELS
        assert SIMULATOR_DEFAULTS["timeout"] > 0
        assert SIMULATOR_DEFAULTS["cpu_count_multiplier"] > 0

    def test_logging_config_structure(self):
        """Test that LOGGING_CONFIG is properly structured."""
        expected_keys = {
            "default_level",
            "verbose_logging",
            "file_logging",
            "log_dir",
            "console_colors",
            "log_format",
            "console_format",
            "file_format",
        }

        assert set(LOGGING_CONFIG.keys()) == expected_keys
        assert isinstance(LOGGING_CONFIG["default_level"], int)
        assert isinstance(LOGGING_CONFIG["verbose_logging"], bool)
        assert isinstance(LOGGING_CONFIG["file_logging"], bool)
        assert isinstance(LOGGING_CONFIG["console_colors"], dict)

    def test_log_levels_mapping(self):
        """Test that LOG_LEVELS mapping is correct."""
        expected_levels = {"HIGH", "INFORMATIVE", "VERBOSE", "MILD", "MINIMAL"}
        assert set(LOG_LEVELS.keys()) == expected_levels

        # All values should be valid logging levels
        valid_levels = {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }
        for level in LOG_LEVELS.values():
            assert level in valid_levels


@pytest.mark.config
class TestConfigurationValidation:
    """Test the configuration validation functions."""

    def test_validate_engine_config_valid(self):
        """Test that validate_engine_config accepts valid configurations."""
        valid_config = ENGINE_DEFAULTS.copy()
        assert validate_engine_config(valid_config) is True

        # Test with custom valid values
        custom_config = {
            "max_iterations": 2000,
            "timeout": 7200,
            "normalize_messages": False,
            "monitor_performance": True,
            "anytime": True,
            "use_bct_history": True,
        }
        assert validate_engine_config(custom_config) is True

    def test_validate_engine_config_missing_keys(self):
        """Test that validate_engine_config rejects configs with missing keys."""
        incomplete_config = ENGINE_DEFAULTS.copy()
        del incomplete_config["max_iterations"]

        with pytest.raises(ValueError, match="Missing required engine config key"):
            validate_engine_config(incomplete_config)

    def test_validate_engine_config_invalid_values(self):
        """Test that validate_engine_config rejects invalid values."""
        # Test negative max_iterations
        invalid_config = ENGINE_DEFAULTS.copy()
        invalid_config["max_iterations"] = -1

        with pytest.raises(
            ValueError, match="max_iterations must be a positive integer"
        ):
            validate_engine_config(invalid_config)

        # Test zero timeout
        invalid_config = ENGINE_DEFAULTS.copy()
        invalid_config["timeout"] = 0

        with pytest.raises(ValueError, match="timeout must be a positive number"):
            validate_engine_config(invalid_config)

        # Test wrong type for max_iterations
        invalid_config = ENGINE_DEFAULTS.copy()
        invalid_config["max_iterations"] = "not_an_int"

        with pytest.raises(
            ValueError, match="max_iterations must be a positive integer"
        ):
            validate_engine_config(invalid_config)

    def test_validate_policy_config_valid(self):
        """Test that validate_policy_config accepts valid configurations."""
        valid_config = POLICY_DEFAULTS.copy()
        assert validate_policy_config(valid_config) is True

        # Test with minimal config
        minimal_config = {"damping_factor": 0.5}
        assert validate_policy_config(minimal_config) is True

    def test_validate_policy_config_invalid_damping_factor(self):
        """Test that validate_policy_config rejects invalid damping factors."""
        # Test damping factor = 0
        with pytest.raises(ValueError, match="damping_factor must be in range"):
            validate_policy_config({"damping_factor": 0.0})

        # Test damping factor > 1
        with pytest.raises(ValueError, match="damping_factor must be in range"):
            validate_policy_config({"damping_factor": 1.5})

        # Test negative damping factor
        with pytest.raises(ValueError, match="damping_factor must be in range"):
            validate_policy_config({"damping_factor": -0.1})

    def test_validate_policy_config_invalid_split_factor(self):
        """Test that validate_policy_config rejects invalid split factors."""
        # Test split factor = 0
        with pytest.raises(ValueError, match="split_factor must be in range"):
            validate_policy_config({"split_factor": 0.0})

        # Test split factor = 1
        with pytest.raises(ValueError, match="split_factor must be in range"):
            validate_policy_config({"split_factor": 1.0})

        # Test split factor > 1
        with pytest.raises(ValueError, match="split_factor must be in range"):
            validate_policy_config({"split_factor": 1.5})

    def test_validate_policy_config_invalid_pruning_threshold(self):
        """Test that validate_policy_config rejects negative pruning thresholds."""
        with pytest.raises(
            ValueError, match="pruning_threshold must be a non-negative number"
        ):
            validate_policy_config({"pruning_threshold": -0.1})

        with pytest.raises(
            ValueError, match="pruning_threshold must be a non-negative number"
        ):
            validate_policy_config({"pruning_threshold": "invalid"})

    def test_validate_convergence_config_valid(self):
        """Test that validate_convergence_config accepts valid configurations."""
        valid_config = CONVERGENCE_DEFAULTS.copy()
        assert validate_convergence_config(valid_config) is True

    def test_validate_convergence_config_invalid_belief_threshold(self):
        """Test that validate_convergence_config rejects invalid belief thresholds."""
        with pytest.raises(
            ValueError, match="belief_threshold must be a positive number"
        ):
            validate_convergence_config({"belief_threshold": 0})

        with pytest.raises(
            ValueError, match="belief_threshold must be a positive number"
        ):
            validate_convergence_config({"belief_threshold": -1e-6})

    def test_validate_convergence_config_invalid_min_iterations(self):
        """Test that validate_convergence_config rejects invalid min_iterations."""
        with pytest.raises(
            ValueError, match="min_iterations must be a non-negative integer"
        ):
            validate_convergence_config({"min_iterations": -1})

        with pytest.raises(
            ValueError, match="min_iterations must be a non-negative integer"
        ):
            validate_convergence_config({"min_iterations": 1.5})

    def test_validate_convergence_config_invalid_patience(self):
        """Test that validate_convergence_config rejects invalid patience values."""
        with pytest.raises(ValueError, match="patience must be a non-negative integer"):
            validate_convergence_config({"patience": -1})

        with pytest.raises(ValueError, match="patience must be a non-negative integer"):
            validate_convergence_config({"patience": 2.5})


@pytest.mark.config
class TestGetValidatedConfig:
    """Test the get_validated_config function for merging configurations."""

    def test_get_validated_config_engine_defaults(self):
        """Test getting engine config with no overrides."""
        config = get_validated_config("engine")
        assert config == ENGINE_DEFAULTS

    def test_get_validated_config_engine_with_overrides(self):
        """Test getting engine config with user overrides."""
        overrides = {"max_iterations": 2000, "timeout": 7200}
        config = get_validated_config("engine", overrides)

        expected = ENGINE_DEFAULTS.copy()
        expected.update(overrides)
        assert config == expected

    def test_get_validated_config_policy_defaults(self):
        """Test getting policy config with no overrides."""
        config = get_validated_config("policy")
        assert config == POLICY_DEFAULTS

    def test_get_validated_config_policy_with_overrides(self):
        """Test getting policy config with user overrides."""
        overrides = {"damping_factor": 0.7, "split_factor": 0.3}
        config = get_validated_config("policy", overrides)

        expected = POLICY_DEFAULTS.copy()
        expected.update(overrides)
        assert config == expected

    def test_get_validated_config_convergence_defaults(self):
        """Test getting convergence config with no overrides."""
        config = get_validated_config("convergence")
        assert config == CONVERGENCE_DEFAULTS

    def test_get_validated_config_unknown_type(self):
        """Test that get_validated_config rejects unknown config types."""
        with pytest.raises(ValueError, match="Unknown config type"):
            get_validated_config("unknown_type")

    def test_get_validated_config_with_invalid_overrides(self):
        """Test that get_validated_config validates merged configuration."""
        # Test with invalid engine override
        with pytest.raises(
            ValueError, match="max_iterations must be a positive integer"
        ):
            get_validated_config("engine", {"max_iterations": -1})

        # Test with invalid policy override
        with pytest.raises(ValueError, match="damping_factor must be in range"):
            get_validated_config("policy", {"damping_factor": 2.0})

        # Test with invalid convergence override
        with pytest.raises(
            ValueError, match="belief_threshold must be a positive number"
        ):
            get_validated_config("convergence", {"belief_threshold": 0})

    def test_get_validated_config_non_validated_types(self):
        """Test that get_validated_config works with non-validated config types."""
        # Simulator config doesn't have validation (currently)
        config = get_validated_config("simulator")
        assert config == SIMULATOR_DEFAULTS

        # Logging config doesn't have validation
        config = get_validated_config("logging")
        assert config == LOGGING_CONFIG

        # Search config doesn't have validation
        config = get_validated_config("search")
        assert config == SEARCH_DEFAULTS


@pytest.mark.config
class TestConstants:
    """Test configuration constants and derived values."""

    def test_message_domain_size(self):
        """Test that MESSAGE_DOMAIN_SIZE is a positive integer."""
        assert isinstance(MESSAGE_DOMAIN_SIZE, int)
        assert MESSAGE_DOMAIN_SIZE > 0

    def test_project_root(self):
        """Test that PROJECT_ROOT is a string."""
        assert isinstance(PROJECT_ROOT, str)
        assert len(PROJECT_ROOT) > 0

    def test_dirs_enum(self):
        """Test that Dirs enum contains expected directory names."""
        expected_dirs = {
            "LOGS",
            "TEST_LOGS",
            "TEST_DATA",
            "TEST_RESULTS",
            "TEST_CONFIGS",
            "TEST_PLOTS",
            "TEST_PLOTS_DATA",
            "TEST_PLOTS_FIGURES",
            "TEST_PLOTS_FIGURES_DATA",
        }

        actual_dirs = {d.name for d in Dirs}
        assert actual_dirs == expected_dirs

        # All values should be strings
        for d in Dirs:
            assert isinstance(d.value, str)

    def test_verbose_logging_legacy_support(self):
        """Test that VERBOSE_LOGGING matches LOGGING_CONFIG setting."""
        assert VERBOSE_LOGGING == LOGGING_CONFIG["verbose_logging"]


@pytest.mark.config
@pytest.mark.parametrize(
    "config_type,expected_keys",
    [
        (
            "engine",
            {
                "max_iterations",
                "timeout",
                "normalize_messages",
                "monitor_performance",
                "anytime",
                "use_bct_history",
            },
        ),
        (
            "policy",
            {
                "damping_factor",
                "damping_diameter",
                "split_factor",
                "pruning_threshold",
                "pruning_magnitude_factor",
                "cost_reduction_enabled",
            },
        ),
        (
            "convergence",
            {
                "belief_threshold",
                "assignment_threshold",
                "min_iterations",
                "patience",
                "use_relative_change",
            },
        ),
        (
            "simulator",
            {
                "default_max_iter",
                "default_log_level",
                "timeout",
                "cpu_count_multiplier",
            },
        ),
    ],
)
class TestParametrizedConfigs:
    """Parametrized tests for configuration structures."""

    def test_config_has_expected_keys(self, config_type, expected_keys):
        """Test that each configuration type has all expected keys."""
        config = get_validated_config(config_type)
        assert set(config.keys()) == expected_keys


@pytest.mark.config
class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_all_default_configs_are_valid(self):
        """Test that all default configurations pass their validators."""
        # This should not raise any exceptions
        validate_engine_config(ENGINE_DEFAULTS)
        validate_policy_config(POLICY_DEFAULTS)
        validate_convergence_config(CONVERGENCE_DEFAULTS)

    def test_config_merging_preserves_types(self):
        """Test that configuration merging preserves data types."""
        original_config = get_validated_config("engine")
        override_config = {"max_iterations": 2000}
        merged_config = get_validated_config("engine", override_config)

        # Type of overridden value should be preserved
        assert type(merged_config["max_iterations"]) == type(
            original_config["max_iterations"]
        )

        # Types of non-overridden values should be preserved
        for key in original_config:
            if key not in override_config:
                assert type(merged_config[key]) == type(original_config[key])

    def test_configuration_immutability(self):
        """Test that getting configurations returns independent copies."""
        config1 = get_validated_config("engine")
        config2 = get_validated_config("engine")

        # Modify one config
        config1["max_iterations"] = 9999

        # Other config should be unchanged
        assert config2["max_iterations"] == ENGINE_DEFAULTS["max_iterations"]

    @patch("propflow.configs.global_config_mapping.validate_engine_config")
    def test_validation_is_called(self, mock_validate):
        """Test that validation functions are called during config merging."""
        mock_validate.return_value = True

        get_validated_config("engine", {"max_iterations": 1000})

        mock_validate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
