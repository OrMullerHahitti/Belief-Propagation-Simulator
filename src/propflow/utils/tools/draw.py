import networkx as nx
import matplotlib.pyplot as plt
from src.propflow.utils.create import crea


def draw_factor_graph(
    fg,
    node_size=300,
    var_color="lightblue",
    factor_color="lightgreen",
    with_labels=True,
):
    """Visualize a bipartite factor graph.

    :param fg: FactorGraph object with attributes G (nx.Graph), variables, and factors
    :param node_size: Size of the nodes in the plot
    :param var_color: Color for variable nodes
    :param factor_color: Color for factor nodes
    :param with_labels: Whether to display node labels
    """
    # Determine variable and factor nodes
    try:
        var_nodes = fg.variables
        factor_nodes = fg.factors
    except AttributeError:
        var_nodes, factor_nodes = nx.bipartite.sets(fg.G)

    # Compute layout
    pos = nx.bipartite_layout(fg.G, var_nodes)

    # Draw nodes
    nx.draw_networkx_nodes(
        fg.G,
        pos,
        nodelist=var_nodes,
        node_shape="o",
        node_color=var_color,
        node_size=node_size,
    )
    nx.draw_networkx_nodes(
        fg.G,
        pos,
        nodelist=factor_nodes,
        node_shape="s",
        node_color=factor_color,
        node_size=node_size,
    )

    # Draw edges
    nx.draw_networkx_edges(fg.G, pos)

    # Draw labels
    if with_labels:
        nx.draw_networkx_labels(fg.G, pos)

    # Show plot
    plt.show()
