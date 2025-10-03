"""Utility functions for creating example factor graphs.

This module provides high-level helper functions to easily generate or load
factor graphs for use in demonstrations, tests, or experiments.
"""
from ..bp.factor_graph import FactorGraph
from typing import Callable, Optional, Dict, Any
import os
import sys

from ..configs.global_config_mapping import CTFactory, get_ct_factory
from .create.create_factor_graphs_from_config import FactorGraphBuilder
from .fg_utils import FGBuilder
from .path_utils import find_project_root


def create_simple_factor_graph_cycle() -> FactorGraph:
    """Loads a simple, pre-defined factor graph with a cycle structure.

    This function is primarily for quick demonstrations and tests. It loads the
    graph configuration from a specific pickle file located in the project's
    `configs` directory.

    Returns:
        A `FactorGraph` instance loaded from the configuration file.
    """
    project_root = find_project_root()
    sys.path.append(str(project_root))
    cfg_path = os.path.join(
        project_root, "configs", "factor_graph_configs", "simple_example.pkl"
    )
    return FactorGraphBuilder().build_and_return(cfg_path)


def create_factor_graph(
    graph_type: str = "cycle",
    num_vars: int = 5,
    domain_size: int = 3,
    ct_factory: str | CTFactory | Callable = "random_int",
    ct_params: Optional[Dict[str, Any]] = None,
    density: float = 0.5,
) -> FactorGraph:
    """Creates a factor graph programmatically based on specified parameters.

    This is a flexible factory function that can generate different types of
    graphs (e.g., 'cycle', 'random') without needing a configuration file.

    Args:
        graph_type: The type of graph to create. Supported types are
            "cycle" and "random". Defaults to "cycle".
        num_vars: The number of variable nodes in the graph. Defaults to 5.
        domain_size: The size of the domain for each variable. Defaults to 3.
        ct_factory: The factory used to generate cost tables for the factors.
            Can be a registered string name, a `CTFactory` enum member, or a
            callable. Defaults to "random_int".
        ct_params: A dictionary of parameters to pass to the `ct_factory`.
            Defaults to `{"low": 1, "high": 100}`.
        density: The density of the graph, used for the "random" graph type.
            Represents the probability of an edge existing between a variable
            and a factor. Defaults to 0.5.

    Returns:
        The generated `FactorGraph` instance.

    Raises:
        ValueError: If an unsupported `graph_type` is provided.
    """
    if ct_params is None:
        ct_params = {"low": 1, "high": 100}

    ct_factory_fn = get_ct_factory(ct_factory)

    if graph_type == "cycle":
        return FGBuilder.build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory_fn,
            ct_params=ct_params,
            density=density,
        )
    elif graph_type == "random":
        return FGBuilder.build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory_fn,
            ct_params=ct_params,
            density=density,
        )
    else:
        raise ValueError(f"Unknown graph type: '{graph_type}'. Supported types are 'cycle' and 'random'.")
