########################################################################
# ----  Internal registries, global configs  ----------------------------------------
########################################################################
from typing import Dict, Callable
from enum import Enum

from bp_base.computators import MaxSumComputator, MinSumComputator
from utils.path_utils import find_project_root

# message parameters
MESSAGE_DOMAIN_SIZE = 3
# Cost Table Creation Function and parameters


# computator
COMPUTATOR = MinSumComputator()


PROJECT_ROOT: str = find_project_root()

#add here global directories, could be also used as ENUM
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


VERBOSE_LOGGING = False  # Set to False to disable verbose logging

# all dotted paths are relative to the project root and will be resolved using importlib
GRAPH_TYPES: Dict[str, str] = {  # str  -> dotted‑path or import key
    # all the graph types builders will be registered here, and with build_... typing
    "cycle": "utils.create_factor_graphs_from_config.build_cycle_graph",
    "octet-variable": "my_bp.graph_builders.build_octet_variable",  # TODO : implemnt octec-variable and octec-factor, not for MST
    "octet-factor": "my_bp.graph_builders.build_octet_factor",  # TODO : implemnt octec-variable and octec-factor, not for MST
    "random": "utils.create_factor_graphs_from_config.build_random_graph",
}

COMPUTATORS: Dict[str, str] = {  # str -> dotted‑path to BPComputator subclass
    "max-sum": "bp_base.computators.MaxSumComputator",
    "min-sum": "bp_base.computators.MinSumComputator",
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
# ---- 2. Project root determination -----------------------------------
########################################################################
