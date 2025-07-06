# create simple factorgraph cycle domain 3 3 variables
from bp_base.factor_graph import FactorGraph
import os

from configs.global_config_mapping import CT_FACTORIES
from utils.create.create_factor_graphs_from_config import (
    FactorGraphBuilder,
    build_cycle_graph,
    build_random_graph,
)
from utils.path_utils import find_project_root
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
    ct_params = _get_default_ct_params(ct_params)
    ct_factory_fn = CT_FACTORIES[ct_factory]
    
    variables, factors, edges = _build_graph_by_type(
        graph_type, num_vars, domain_size, ct_factory_fn, ct_params, density
    )
    
    return FactorGraph(variable_li=variables, factor_li=factors, edges=edges)


def _get_default_ct_params(ct_params):
    """Get default cost table parameters if none provided."""
    return ct_params if ct_params is not None else {"low": 1, "high": 100}


def _build_graph_by_type(graph_type, num_vars, domain_size, ct_factory_fn, ct_params, density):
    """Build graph components based on specified type."""
    graph_builders = {
        "cycle": lambda: build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory_fn,
            ct_params=ct_params,
            density=density,
        ),
        "random": lambda: build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory_fn,
            ct_params=ct_params,
            density=density,
        )
    }
    
    if graph_type not in graph_builders:
        raise ValueError(f"Unknown graph type: {graph_type}")
        
    return graph_builders[graph_type]()
