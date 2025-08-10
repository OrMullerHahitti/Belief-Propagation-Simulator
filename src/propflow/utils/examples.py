# create simple factorgraph cycle domain 3 3 variables
from ..bp.factor_graph import FactorGraph
import os

from ..configs.global_config_mapping import CT_FACTORIES
from .create.create_factor_graphs_from_config import (
    FactorGraphBuilder,
    build_cycle_graph,
    build_random_graph,
)
from .path_utils import find_project_root
import sys


# create simple factorgraph cycle domain 3 3 variables
def create_simple_factor_graph_cycle() -> FactorGraph:
    project_root = find_project_root()
    sys.path.append(str(project_root))
    cfg = os.path.join(
        project_root, "configs", "factor_graph_configs", "simple_example.pkl"
    )

    fg = FactorGraphBuilder().build_and_return(cfg)

    return fg


def create_factor_graph(
    graph_type="cycle",
    num_vars=5,
    domain_size=3,
    ct_factory="random_int",
    ct_params=None,
    density=0.5,
):
    """
    Create a factor graph directly without going through the config and pickle process.

    Args:
        graph_type (str): Type of graph to create ("cycle" or "random")
        num_vars (int): Number of variables in the graph
        domain_size (int): Size of the domain for each variable
        ct_factory (str): Name of the cost table factory to use
        ct_params (dict): Parameters for the cost table factory
        density (float): Density of the graph (for random graphs)

    Returns:
        FactorGraph: The created factor graph
    """
    if ct_params is None:
        ct_params = {"low": 1, "high": 100}

    # Get the cost table factory function
    ct_factory_fn = CT_FACTORIES[ct_factory]
    if graph_type == "cycle":
        variables, factors, edges = build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory_fn,
            ct_params=ct_params,
            density=density,
        )

    if graph_type == "random":
        variables, factors, edges = build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory_fn,
            ct_params=ct_params,
            density=density,
        )
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Create the factor graph
    fg = FactorGraph(variable_li=variables, factor_li=factors, edges=edges)

    return fg
