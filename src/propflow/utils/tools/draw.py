"""A utility function for visualizing factor graphs using Matplotlib.

This module provides a convenient way to draw a `FactorGraph` object,
distinguishing between variable and factor nodes by color and shape.
"""
from typing import Any
import networkx as nx
import matplotlib.pyplot as plt


def draw_factor_graph(
    fg: Any,
    node_size: int = 300,
    var_color: str = "lightblue",
    factor_color: str = "lightgreen",
    with_labels: bool = True,
) -> None:
    """Visualizes a bipartite factor graph using a bipartite layout.

    This function attempts to access `fg.variables` and `fg.factors` to
    identify the two sets of nodes for the bipartite layout. If these
    attributes are not present, it falls back to using `nx.bipartite.sets`.

    Args:
        fg: A `FactorGraph` object or any object with a `G` attribute
            that is a `networkx.Graph`.
        node_size: The size of the nodes in the plot.
        var_color: The color to use for variable nodes.
        factor_color: The color to use for factor nodes.
        with_labels: If True, displays the names of the nodes as labels.
    """
    try:
        var_nodes = fg.variables
        factor_nodes = fg.factors
    except AttributeError:
        # Fallback for generic graphs with a 'bipartite' attribute
        var_nodes, factor_nodes = nx.bipartite.sets(fg.G)

    pos = nx.bipartite_layout(fg.G, var_nodes)

    nx.draw_networkx_nodes(
        fg.G, pos, nodelist=var_nodes, node_shape="o",
        node_color=var_color, node_size=node_size
    )
    nx.draw_networkx_nodes(
        fg.G, pos, nodelist=factor_nodes, node_shape="s",
        node_color=factor_color, node_size=node_size
    )

    nx.draw_networkx_edges(fg.G, pos)

    if with_labels:
        nx.draw_networkx_labels(fg.G, pos)

    plt.show()
