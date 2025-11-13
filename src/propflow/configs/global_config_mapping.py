"""
Global Configuration and Registries for the PropFlow Library.

This module serves as a centralized location for default configurations,
registries, and factory functions used throughout the application. It ensures
consistent and manageable settings for various components like the simulation
engine, logging, policies, and more.

Key Sections:
- Core Configurations: Basic settings like domain size and project root.
- Component Defaults: Dictionaries holding default parameters for the engine,
  logging, convergence criteria, policies, etc.
- Configuration Validation: Functions to validate configuration dictionaries.
- Registries and Factories: Mappings for dynamically loading graph builders,
  computators, and cost table factories.
"""
from typing import Dict, Callable, Any, Optional
from enum import Enum
import logging

from narwhals import Field

from ..bp.computators import MinSumComputator
from ..utils.path_utils import find_project_root

########################################################################
# ---- Core Configuration Sections ------------------------------------
########################################################################

# Default domain size for messages if not otherwise specified.
MESSAGE_DOMAIN_SIZE = 3

# Default computator instance used across the application.
COMPUTATOR = MinSumComputator()

# Automatically determine the project's root directory.
PROJECT_ROOT: str = find_project_root()


class Dirs(Enum):
    """Standardized names for common project directories."""

    LOGS = "logs"
    TEST_LOGS = "test_logs"
    TEST_DATA = "test_data"
    TEST_RESULTS = "test_results"
    TEST_CONFIGS = "test_configs"
    TEST_PLOTS = "test_plots"
    TEST_PLOTS_DATA = "test_plots_data"
    TEST_PLOTS_FIGURES = "test_plots_figures"
    TEST_PLOTS_FIGURES_DATA = "test_plots_figures_data"


########################################################################
# ---- Engine Configuration -------------------------------------------
########################################################################

# Default parameters for the belief propagation engine.
class EngineDefaults(Enum):
    """Default parameters for the belief propagation engine."""
    MAX_ITERATIONS = 2000
    NORMALIZE_MESSAGES = True
    MONITOR_PERFORMANCE = False
    ANYTIME = False
    USE_BCT_HISTORY = False
    TIMEOUT = 600  # seconds


class LoggingDefaults(Enum):
    """Default configuration for the logging system."""
    DEFAULT_LEVEL = logging.INFO
    VERBOSE_LOGGING = False
    FILE_LOGGING = True
    LOG_DIR = "configs/logs"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    CONSOLE_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(message)s"
    FILE_FORMAT = "%(asctime)s - %(name)s  - %(message)s"


# For complex nested structures like console_colors, keep as a separate constant
CONSOLE_COLORS: Dict[str, str] = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}


########################################################################
# ---- Convergence Configuration --------------------------------------
########################################################################


class ConvergenceDefaults(Enum):
    """Default parameters for the convergence monitor."""
    BELIEF_THRESHOLD = 1e-6
    ASSIGNMENT_THRESHOLD = 0
    MIN_ITERATIONS = 0
    PATIENCE = 5
    USE_RELATIVE_CHANGE = True
    TIMEOUT = 600  # seconds


########################################################################
# ---- Policy Configuration -------------------------------------------
########################################################################


class PolicyDefaults(Enum):
    """Default parameters for various belief propagation policies."""
    DAMPING_FACTOR = 0.9
    DAMPING_DIAMETER = 1
    SPLIT_FACTOR = 0.5
    PRUNING_THRESHOLD = 0.1
    PRUNING_MAGNITUDE_FACTOR = 0.1
    COST_REDUCTION_ENABLED = True


########################################################################
# ---- Simulator Configuration ----------------------------------------
########################################################################


class SimulatorDefaults(Enum):
    """Default parameters for the multi-simulation runner."""
    DEFAULT_MAX_ITER = 5000
    DEFAULT_LOG_LEVEL = "INFORMATIVE"
    TIMEOUT = 3600
    CPU_COUNT_MULTIPLIER = 1.0  # Fraction of CPU cores to use


########################################################################
# ---- Search Engine Configuration ------------------------------------
########################################################################


class SearchDefaults(Enum):
    """Default parameters for search-based algorithms."""
    MAX_ITERATIONS = 100
    SEARCH_TIMEOUT = 1800  # 30 minutes
    BEAM_WIDTH = 10
    EXPLORATION_FACTOR = 0.1

# Legacy dict interfaces for backward compatibility
ENGINE_DEFAULTS: Dict[str, Any] = {
    "max_iterations": EngineDefaults.MAX_ITERATIONS.value,
    "normalize_messages": EngineDefaults.NORMALIZE_MESSAGES.value,
    "monitor_performance": EngineDefaults.MONITOR_PERFORMANCE.value,
    "anytime": EngineDefaults.ANYTIME.value,
    "use_bct_history": EngineDefaults.USE_BCT_HISTORY.value,
    "timeout": EngineDefaults.TIMEOUT.value,
}

########################################################################
# ---- Logging Configuration ------------------------------------------
########################################################################

# Default configuration for the logging system.
LOGGING_CONFIG: Dict[str, Any] = {
    "default_level": LoggingDefaults.DEFAULT_LEVEL.value,
    "verbose_logging": LoggingDefaults.VERBOSE_LOGGING.value,
    "file_logging": LoggingDefaults.FILE_LOGGING.value,
    "log_dir": LoggingDefaults.LOG_DIR.value,
    "console_colors": CONSOLE_COLORS,
    "log_format": LoggingDefaults.LOG_FORMAT.value,
    "console_format": LoggingDefaults.CONSOLE_FORMAT.value,
    "file_format": LoggingDefaults.FILE_FORMAT.value,
}

# Mapping of descriptive log level names to `logging` module constants.
LOG_LEVELS: Dict[str, int] = {
    "HIGH": logging.DEBUG,
    "INFORMATIVE": logging.INFO,
    "VERBOSE": logging.WARNING,
    "MILD": logging.ERROR,
    "MINIMAL": logging.CRITICAL,
}

# Default parameters for the convergence monitor (backward compatibility).
CONVERGENCE_DEFAULTS: Dict[str, Any] = {
    "belief_threshold": ConvergenceDefaults.BELIEF_THRESHOLD.value,
    "assignment_threshold": ConvergenceDefaults.ASSIGNMENT_THRESHOLD.value,
    "min_iterations": ConvergenceDefaults.MIN_ITERATIONS.value,
    "patience": ConvergenceDefaults.PATIENCE.value,
    "use_relative_change": ConvergenceDefaults.USE_RELATIVE_CHANGE.value,
    "timeout": ConvergenceDefaults.TIMEOUT.value,
}

# Default parameters for various belief propagation policies (backward compatibility).
POLICY_DEFAULTS: Dict[str, Any] = {
    "damping_factor": PolicyDefaults.DAMPING_FACTOR.value,
    "damping_diameter": PolicyDefaults.DAMPING_DIAMETER.value,
    "split_factor": PolicyDefaults.SPLIT_FACTOR.value,
    "pruning_threshold": PolicyDefaults.PRUNING_THRESHOLD.value,
    "pruning_magnitude_factor": PolicyDefaults.PRUNING_MAGNITUDE_FACTOR.value,
    "cost_reduction_enabled": PolicyDefaults.COST_REDUCTION_ENABLED.value,
}

# Default parameters for the multi-simulation runner (backward compatibility).
SIMULATOR_DEFAULTS: Dict[str, Any] = {
    "default_max_iter": SimulatorDefaults.DEFAULT_MAX_ITER.value,
    "default_log_level": SimulatorDefaults.DEFAULT_LOG_LEVEL.value,
    "timeout": SimulatorDefaults.TIMEOUT.value,
    "cpu_count_multiplier": SimulatorDefaults.CPU_COUNT_MULTIPLIER.value,
}

# Default parameters for search-based algorithms (backward compatibility).
SEARCH_DEFAULTS: Dict[str, Any] = {
    "max_iterations": SearchDefaults.MAX_ITERATIONS.value,
    "search_timeout": SearchDefaults.SEARCH_TIMEOUT.value,
    "beam_width": SearchDefaults.BEAM_WIDTH.value,
    "exploration_factor": SearchDefaults.EXPLORATION_FACTOR.value,
}

# Legacy support for verbose logging flag.
VERBOSE_LOGGING = LOGGING_CONFIG["verbose_logging"]

########################################################################
# ---- Configuration Validation ---------------------------------------
########################################################################


def validate_engine_config(config: Dict[str, Any]) -> bool:
    """Validate engine configuration. Raises ValueError if invalid."""
    required_keys = [
        "max_iterations",
        "normalize_messages",
        "monitor_performance",
        "anytime",
        "use_bct_history",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required engine config key: {key}")
    if not isinstance(config["max_iterations"], int) or config["max_iterations"] <= 0:
        raise ValueError("max_iterations must be a positive integer")
    return True


def validate_policy_config(config: Dict[str, Any]) -> bool:
    """Validate policy configuration. Raises ValueError if invalid."""
    if "damping_factor" in config and not 0.0 < config["damping_factor"] <= 1.0:
        raise ValueError("damping_factor must be in range (0, 1]")
    if "split_factor" in config and not 0.0 < config["split_factor"] < 1.0:
        raise ValueError("split_factor must be in range (0, 1)")
    if "pruning_threshold" in config and (
        not isinstance(config["pruning_threshold"], (int, float))
        or config["pruning_threshold"] < 0
    ):
        raise ValueError("pruning_threshold must be a non-negative number")
    return True


def validate_convergence_config(config: Dict[str, Any]) -> bool:
    """Validates convergence configuration parameters.

    Args:
        config: A dictionary containing convergence configuration.

    Returns:
        True if the configuration is valid.

    Raises:
        ValueError: If a value is invalid or outside its expected range.
    """
    if "belief_threshold" in config and (
        not isinstance(config["belief_threshold"], (int, float))
        or config["belief_threshold"] <= 0
    ):
        raise ValueError("belief_threshold must be a positive number")
    if "min_iterations" in config and (
        not isinstance(config["min_iterations"], int) or config["min_iterations"] < 0
    ):
        raise ValueError("min_iterations must be a non-negative integer")
    if "patience" in config and (
        not isinstance(config["patience"], int) or config["patience"] < 0
    ):
        raise ValueError("patience must be a non-negative integer")
    return True


def get_validated_config(
    config_type: str, user_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merges a user configuration with defaults and validates it.

    Args:
        config_type: The type of configuration to retrieve (e.g., 'engine', 'policy').
        user_config: An optional dictionary of user-provided overrides.

    Returns:
        A dictionary containing the final, validated configuration.

    Raises:
        ValueError: If the config_type is unknown or validation fails.
    """
    base_configs = {
        "engine": ENGINE_DEFAULTS,
        "policy": POLICY_DEFAULTS,
        "convergence": CONVERGENCE_DEFAULTS,
        "simulator": SIMULATOR_DEFAULTS,
        "logging": LOGGING_CONFIG,
        "search": SEARCH_DEFAULTS,
    }
    if config_type not in base_configs:
        raise ValueError(f"Unknown config type: {config_type}")

    final_config = base_configs[config_type].copy()
    if user_config:
        final_config.update(user_config)

    validators = {
        "engine": validate_engine_config,
        "policy": validate_policy_config,
        "convergence": validate_convergence_config,
    }
    if config_type in validators:
        validators[config_type](final_config)

    return final_config


# Validate default configurations on import
try:
    validate_engine_config(ENGINE_DEFAULTS)
    validate_policy_config(POLICY_DEFAULTS)
    validate_convergence_config(CONVERGENCE_DEFAULTS)
except ValueError as e:
    raise RuntimeError(f"Invalid default configuration: {e}")

########################################################################
# ---- Registry and Factory Mappings ----------------------------------
########################################################################

# all dotted paths are relative to the project root and will be resolved using importlib
GRAPH_TYPES: Dict[str, str] = {  # str  -> dotted‑path or import key
    # all the graph types builders will be registered here, and with build_... typing
    "cycle": "utils.create_factor_graphs_from_config.build_cycle_graph",
    "octet-variable": "my_bp.graph_builders.build_octet_variable",  # TODO : implemnt octec-variable and octec-factor, not for MST
    "octet-factor": "my_bp.graph_builders.build_octet_factor",  # TODO : implemnt octec-variable and octec-factor, not for MST
    "random": "utils.create_factor_graphs_from_config.build_random_graph",
}

COMPUTATORS: Dict[str, str] = {  # str -> dotted‑path to BPComputator subclass
    "max-sum": "bp.computators.MaxSumComputator",
    "min-sum": "bp.computators.MinSumComputator",
    "sum-product": "my_bp.computators.SumProductComputator",
}

ENGINE_MAPPING: Dict[str, str] = {
    "engine.search.a_star_fg": "propflow.search.algorithms:a_star_factor_graph",
    "engine.search.greedy_fg": "propflow.search.algorithms:greedy_best_first_factor_graph",
    "engine.search.beam_fg": "propflow.search.algorithms:beam_search_factor_graph",
}

########################################################################
# ---- Cost Table Factory Functions -----------------------------------
########################################################################


def create_poisson_table(
    n: int, domain: int, rate: float = 1.0, strength: float | None = None
):
    """Generate cost table with Poisson-distributed values.

    When using FGBuilder, pass `rate` or `strength` via ct_params.
    """
    import numpy as np

    lam = strength if strength is not None else rate
    shape = (domain,) * n
    return np.random.poisson(lam=lam, size=shape)


def create_random_int_table(n: int, domain: int, low: int = 0, high: int = 10):
    """Generate cost table with random integer values.

    When using FGBuilder, pass `low` and `high` via ct_params.
    """
    import numpy as np

    shape = (domain,) * n
    return np.random.randint(low=low, high=high, size=shape)


def create_uniform_float_table(
    n: int, domain: int, low: float = 0.0, high: float = 1.0
):
    """Generate cost table with uniform float values.

    When using FGBuilder, pass `low` and `high` via ct_params.
    """
    import numpy as np

    shape = (domain,) * n
    return np.random.uniform(low=low, high=high, size=shape)


# Registry of factory functions for string-based lookups and config file support
CT_FACTORIES: Dict[str, Callable] = {
    "poisson": create_poisson_table,
    "random_int": create_random_int_table,
    "uniform_float": create_uniform_float_table,
}


########################################################################
# ---- CT Factory Namespace + helpers ---------------------------------
########################################################################


class CTFactories:
    """Cost table factory namespace. Use: CTFactories.RANDOM_INT"""

    UNIFORM = create_uniform_float_table
    RANDOM_INT = create_random_int_table
    POISSON = create_poisson_table


def get_ct_factory(factory: Callable | str) -> Callable:
    """Resolve factory identifier to callable.

    Accepts CTFactories.RANDOM_INT, "random_int", or raw function.
    """
    if callable(factory):
        return factory  # type: ignore[return-value]
    if isinstance(factory, str):
        return CT_FACTORIES[factory]
    raise TypeError(f"Unsupported ct_factory type: {type(factory).__name__}")


########################################################################
# ---- 2. Project root determination -----------------------------------
########################################################################
