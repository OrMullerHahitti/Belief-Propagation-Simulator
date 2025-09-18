########################################################################
# ----  Internal registries, global configs  ----------------------------------------
########################################################################
from socket import timeout
from typing import Dict, Callable, Any, Optional
from enum import Enum
import logging

from ..bp.computators import MinSumComputator
from ..utils.path_utils import find_project_root

########################################################################
# ---- Core Configuration Sections ------------------------------------
########################################################################

# Message and Domain Configuration
MESSAGE_DOMAIN_SIZE = 3

# Default computator
COMPUTATOR = MinSumComputator()

# Project root path
PROJECT_ROOT: str = find_project_root()


# Directory structure configuration
class Dirs(Enum):
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

ENGINE_DEFAULTS: Dict[str, Any] = {
    "max_iterations": 2000,
    "normalize_messages": True,
    "monitor_performance": False,
    "anytime": False,
    "use_bct_history": False,
    'timeout': 600,  # seconds
}

########################################################################
# ---- Logging Configuration ------------------------------------------
########################################################################

LOGGING_CONFIG: Dict[str, Any] = {
    "default_level": logging.INFO,
    "verbose_logging": False,
    "file_logging": True,
    "log_dir": "configs/logs",
    "console_colors": {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console_format": "%(log_color)s%(asctime)s - %(name)s - %(message)s",
    "file_format": "%(asctime)s - %(name)s  - %(message)s",
}

LOG_LEVELS: Dict[str, int] = {
    "HIGH": logging.DEBUG,
    "INFORMATIVE": logging.INFO,
    "VERBOSE": logging.WARNING,
    "MILD": logging.ERROR,
    "MINIMAL": logging.CRITICAL,
}

########################################################################
# ---- Convergence Configuration --------------------------------------
########################################################################

CONVERGENCE_DEFAULTS: Dict[str, Any] = {
    "belief_threshold": 1e-6,
    "assignment_threshold": 0,
    "min_iterations": 0,
    "patience": 5,
    "use_relative_change": True,
    "timeout": 600,  # seconds
}

########################################################################
# ---- Policy Configuration -------------------------------------------
########################################################################

POLICY_DEFAULTS: Dict[str, Any] = {
    # Damping policy defaults
    "damping_factor": 0.9,
    "damping_diameter": 1,
    # Splitting policy defaults
    "split_factor": 0.5,
    # Message pruning defaults
    "pruning_threshold": 0.1,
    "pruning_magnitude_factor": 0.1,
    # Cost reduction defaults
    "cost_reduction_enabled": True,
}

########################################################################
# ---- Simulator Configuration ----------------------------------------
########################################################################

SIMULATOR_DEFAULTS: Dict[str, Any] = {
    "default_max_iter": 5000,
    "default_log_level": "INFORMATIVE",
    "timeout": 3600,
    "cpu_count_multiplier": 1.0,  # Fraction of CPU cores to use
}

########################################################################
# ---- Search Engine Configuration ------------------------------------
########################################################################

SEARCH_DEFAULTS: Dict[str, Any] = {
    "max_iterations": 100,
    "search_timeout": 1800,  # 30 minutes
    "beam_width": 10,
    "exploration_factor": 0.1,
}

# Legacy support
VERBOSE_LOGGING = LOGGING_CONFIG["verbose_logging"]

########################################################################
# ---- Configuration Validation ---------------------------------------
########################################################################


def validate_engine_config(config: Dict[str, Any]) -> bool:
    """Validate engine configuration parameters."""
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
    """Validate policy configuration parameters."""
    if "damping_factor" in config:
        if not 0.0 < config["damping_factor"] <= 1.0:
            raise ValueError("damping_factor must be in range (0, 1]")

    if "split_factor" in config:
        if not 0.0 < config["split_factor"] < 1.0:
            raise ValueError("split_factor must be in range (0, 1)")

    if "pruning_threshold" in config:
        if (
            not isinstance(config["pruning_threshold"], (int, float))
            or config["pruning_threshold"] < 0
        ):
            raise ValueError("pruning_threshold must be a non-negative number")

    return True


def validate_convergence_config(config: Dict[str, Any]) -> bool:
    """Validate convergence configuration parameters."""
    if "belief_threshold" in config:
        if (
            not isinstance(config["belief_threshold"], (int, float))
            or config["belief_threshold"] <= 0
        ):
            raise ValueError("belief_threshold must be a positive number")

    if "min_iterations" in config:
        if (
            not isinstance(config["min_iterations"], int)
            or config["min_iterations"] < 0
        ):
            raise ValueError("min_iterations must be a non-negative integer")

    if "patience" in config:
        if not isinstance(config["patience"], int) or config["patience"] < 0:
            raise ValueError("patience must be a non-negative integer")

    return True


def get_validated_config(
    config_type: str, user_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get a validated configuration by merging user overrides with defaults."""

    # Get base configuration
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

    # Start with base configuration
    final_config = base_configs[config_type].copy()

    # Apply user overrides
    if user_config:
        final_config.update(user_config)

    # Validate the final configuration
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

CT_FACTORIES: Dict[str, Callable] = {}  # filled in by decorator below


def ct_factory(name: str):
    """Decorator to register a cost‑table factory under a short name."""

    def decorator(fn: Callable):
        CT_FACTORIES[name] = fn
        return fn

    return decorator


########################################################################
# ------ all the function creators here:
#########################################################################
# TODO : add the rest of the cost table factories which i already made, i think it would be better if theyre al in one place, can still leave the other one for future uses
# TODO: add docstrings to the functions
@ct_factory("poisson")
def create_poisson_table(n: int, domain: int, rate: float = 1.0):
    """
    Creates a cost table with Poisson-distributed values.

    Args:
        n: Number of dimensions for the cost table (number of connected variables).
        domain: Size of the domain for each variable/dimension.
        rate: The rate parameter for the Poisson distribution.

    Returns:
        A numpy ndarray representing the cost table with shape (domain,) * n.
    """
    import numpy as np

    shape = (domain,) * n
    return np.random.poisson(lam=rate, size=shape)


@ct_factory("random_int")
def create_random_int_table(n: int, domain: int, low: int = 0, high: int = 10):
    """
    Creates a cost table with random integer values.

    Args:
        n: Number of dimensions for the cost table (number of connected variables).
        domain: Size of the domain for each variable/dimension.
        low: The lower bound for random integer generation (inclusive).
        high: The upper bound for random integer generation (exclusive).

    Returns:
        A numpy ndarray representing the cost table with shape (domain,) * n.
    """
    import numpy as np

    shape = (domain,) * n
    return np.random.randint(low=low, high=high, size=shape)


@ct_factory("uniform_float")
def create_uniform_float_table(
    n: int, domain: int, low: float = 0.0, high: float = 1.0
):
    """
    Creates a cost table with random float values from a uniform distribution.

    Args:
        n: Number of dimensions for the cost table (number of connected variables).
        domain: Size of the domain for each variable/dimension.
        low: The lower bound for the uniform distribution (inclusive).
        high: The upper bound for the uniform distribution (exclusive).

    Returns:
        A numpy ndarray representing the cost table with shape (domain,) * n.
    """
    import numpy as np

    shape = (domain,) * n
    return np.random.uniform(low=low, high=high, size=shape)


########################################################################
# ---- CT Factory Enum + helpers --------------------------------------
########################################################################

# Build an Enum from the registered factory names to enable IDE completion
# (e.g., CTFactory.random_int)
CTFactory = Enum("CTFactory", {name: name for name in CT_FACTORIES.keys()})


def get_ct_factory(factory: "CTFactory | str | Callable") -> Callable:
    """
    Resolve a factory identifier (Enum, str, or callable) into a callable.

    - CTFactory.random_int -> corresponding function
    - "random_int" -> function
    - callable -> returned as-is
    """
    # Already a callable (backward compatible)
    if callable(factory):
        return factory  # type: ignore[return-value]

    # Enum member
    if isinstance(factory, CTFactory):
        key = factory.value  # type: ignore[attr-defined]
        return CT_FACTORIES[key]

    # String key
    if isinstance(factory, str):
        return CT_FACTORIES[factory]

    raise TypeError(
        f"Unsupported ct_factory type: {type(factory).__name__}. "
        f"Use CTFactory enum, a registered name, or a callable."
    )


# Convenience: attach `.fn` attribute to enum members for quick access
for _m in CTFactory:  # type: ignore[not-an-iterable]
    try:
        setattr(_m, "fn", CT_FACTORIES[_m.value])  # e.g., CTFactory.random_int.fn
    except Exception:
        pass


########################################################################
# ---- 2. Project root determination -----------------------------------
########################################################################
