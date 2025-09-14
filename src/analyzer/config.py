"""
Configuration management for the analysis framework.

Provides centralized configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class AnalysisConfig:
    """
    Configuration class for analysis framework with all tunable parameters.

    All parameters have sensible defaults and can be overridden as needed.
    """

    # Cycle analysis parameters
    max_cycle_len: int = 12
    compute_numeric_cycle_gain: bool = True
    estimate_global_norm_bound: bool = True

    # Winner/min_idx computation
    rebuild_winners_if_missing: bool = True
    rebuild_min_idx_if_missing: bool = True

    # Tie handling
    treat_ties_as_error: bool = False  # if True, flag CRITICAL when ties occur
    tie_tolerance: float = 1e-10

    # Numerical tolerances
    abs_tol: float = 1e-10
    rel_tol: float = 1e-9

    # Matrix computation
    use_sparse_matrices: bool = True
    validate_matrix_structure: bool = True

    # Performance optimization
    max_brute_force_domain_size: int = 10  # for winner computation
    cache_matrix_computations: bool = True

    # Output control
    include_detailed_cycles: bool = False  # can be memory intensive
    include_matrix_norms: bool = True
    include_enforcement_suggestions: bool = True

    # Validation and debugging
    check_invariants: bool = True
    validate_normalization: bool = True
    validate_projector: bool = True
    verbose_warnings: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        self.validate()

    def validate(self):
        """Validate configuration parameters and raise errors for invalid values."""
        if self.max_cycle_len < 1:
            raise ValueError("max_cycle_len must be at least 1")

        if self.abs_tol <= 0 or self.rel_tol <= 0:
            raise ValueError("Tolerances must be positive")

        if self.tie_tolerance <= 0:
            raise ValueError("tie_tolerance must be positive")

        if self.max_brute_force_domain_size < 2:
            raise ValueError("max_brute_force_domain_size must be at least 2")

    def update(self, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        """
        Update configuration with provided dictionary and return new instance.

        Args:
            config_dict: Dictionary of parameter updates

        Returns:
            New AnalysisConfig instance with updated parameters
        """
        updates = {}
        for key, value in config_dict.items():
            if hasattr(self, key):
                updates[key] = value
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Create new instance with updates
        current_values = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        current_values.update(updates)

        return AnalysisConfig(**current_values)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        """Create configuration from dictionary."""
        # Filter to only known parameters
        known_params = {
            key: value
            for key, value in config_dict.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**known_params)

    @classmethod
    def default(cls) -> "AnalysisConfig":
        """Get default configuration."""
        return cls()

    @classmethod
    def fast_config(cls) -> "AnalysisConfig":
        """Get configuration optimized for speed over completeness."""
        return cls(
            max_cycle_len=8,
            compute_numeric_cycle_gain=False,
            include_detailed_cycles=False,
            check_invariants=False,
            validate_matrix_structure=False,
            verbose_warnings=False,
        )

    @classmethod
    def comprehensive_config(cls) -> "AnalysisConfig":
        """Get configuration for comprehensive analysis."""
        return cls(
            max_cycle_len=16,
            compute_numeric_cycle_gain=True,
            include_detailed_cycles=True,
            check_invariants=True,
            validate_matrix_structure=True,
            verbose_warnings=True,
        )

    @classmethod
    def debug_config(cls) -> "AnalysisConfig":
        """Get configuration for debugging with extensive validation."""
        return cls(
            max_cycle_len=10,
            check_invariants=True,
            validate_matrix_structure=True,
            validate_normalization=True,
            validate_projector=True,
            verbose_warnings=True,
            treat_ties_as_error=True,
        )


# Global default configuration instance
DEFAULT_CONFIG = AnalysisConfig.default()


def get_default_config() -> AnalysisConfig:
    """Get the default configuration instance."""
    return DEFAULT_CONFIG


def create_config(overrides: Optional[Dict[str, Any]] = None) -> AnalysisConfig:
    """
    Create a configuration with optional overrides.

    Args:
        overrides: Dictionary of parameter overrides

    Returns:
        AnalysisConfig instance
    """
    if overrides is None:
        return DEFAULT_CONFIG

    return DEFAULT_CONFIG.update(overrides)


def validate_config(config: Any) -> AnalysisConfig:
    """
    Validate and convert configuration input to AnalysisConfig.

    Args:
        config: Configuration input (dict, AnalysisConfig, or None)

    Returns:
        Valid AnalysisConfig instance

    Raises:
        TypeError: If config type is not supported
        ValueError: If config values are invalid
    """
    if config is None:
        return DEFAULT_CONFIG

    if isinstance(config, AnalysisConfig):
        config.validate()  # Ensure it's still valid
        return config

    if isinstance(config, dict):
        return create_config(config)

    raise TypeError(f"Config must be dict, AnalysisConfig, or None, got {type(config)}")


# Configuration presets for common scenarios
PRESETS = {
    "default": AnalysisConfig.default,
    "fast": AnalysisConfig.fast_config,
    "comprehensive": AnalysisConfig.comprehensive_config,
    "debug": AnalysisConfig.debug_config,
}


def get_preset_config(preset_name: str) -> AnalysisConfig:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of preset ('default', 'fast', 'comprehensive', 'debug')

    Returns:
        AnalysisConfig instance

    Raises:
        ValueError: If preset name is unknown
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return PRESETS[preset_name]()


def list_presets() -> List[str]:
    """Get list of available preset names."""
    return list(PRESETS.keys())


# Example usage and documentation
if __name__ == "__main__":
    # Basic usage examples
    print("=== Analysis Configuration Examples ===")

    # Default configuration
    cfg = AnalysisConfig()
    print(f"Default max_cycle_len: {cfg.max_cycle_len}")

    # Configuration with overrides
    cfg_fast = AnalysisConfig(max_cycle_len=6, compute_numeric_cycle_gain=False)
    print(f"Fast config max_cycle_len: {cfg_fast.max_cycle_len}")

    # Update existing configuration
    cfg_updated = cfg.update({"max_cycle_len": 20, "treat_ties_as_error": True})
    print(f"Updated max_cycle_len: {cfg_updated.max_cycle_len}")

    # Preset configurations
    for preset_name in list_presets():
        preset_cfg = get_preset_config(preset_name)
        print(f"{preset_name} preset max_cycle_len: {preset_cfg.max_cycle_len}")

    # Dictionary conversion
    cfg_dict = cfg.to_dict()
    cfg_from_dict = AnalysisConfig.from_dict(cfg_dict)
    print(f"Roundtrip successful: {cfg == cfg_from_dict}")

    print("\n=== Configuration validation ===")

    # Test validation
    try:
        invalid_cfg = AnalysisConfig(max_cycle_len=-1)
    except ValueError as e:
        print(f"Validation caught error: {e}")

    print("\n=== Ready for use in analysis framework ===")
