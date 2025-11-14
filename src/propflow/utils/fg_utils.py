"""Utilities for creating, loading, and manipulating factor graphs.

This module provides a collection of helper functions and classes for common
tasks related to factor graphs, such as building graphs with specific
topologies (random, cycle), calculating bounds, and safely handling pickled
graph objects.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import pickle
import sys
from typing import Callable, Dict, Any, List, Tuple
from functools import lru_cache
import random

import networkx as nx
import numpy as np

from .path_utils import find_project_root
from ..bp.factor_graph import FactorGraph
from ..configs.global_config_mapping import get_ct_factory, CTFactories
from ..core.agents import VariableAgent, FactorAgent

project_root = find_project_root()
sys.path.append(str(project_root))

_MAX_SEED = 2**63 - 1


def _make_variable(idx: int, domain: int) -> VariableAgent:
    """Creates a single `VariableAgent` with a standardized name."""
    return VariableAgent(name=f"x{idx}", domain=domain)


def _make_factor(
    name: str, domain: int, ct_factory: Callable | str, ct_params: dict
) -> FactorAgent:
    """Creates a single `FactorAgent`, deferring cost table creation."""
    ct_fn = get_ct_factory(ct_factory)
    return FactorAgent(
        name=name, domain=domain, ct_creation_func=ct_fn, param=ct_params
    )


def _build_factor_edge_list(
    edges: List[Tuple[VariableAgent, VariableAgent]],
    domain_size: int,
    ct_factory: Any,
    ct_params: dict,
) -> Dict[FactorAgent, List[VariableAgent]]:
    """Creates factor nodes for binary constraints and maps them to variables."""
    edge_dict = {}
    for a, b in edges:
        fname = f"f{a.name[1:]}{b.name[1:]}"
        fnode = _make_factor(fname, domain_size, ct_factory, ct_params)
        edge_dict[fnode] = [a, b]
    return edge_dict


def _resolve_graph_seed(seed: int | None) -> int:
    """Resolve a deterministic seed, respecting the global numpy RNG."""

    if seed is not None:
        return int(seed) % _MAX_SEED

    # numpy's legacy global RNG honors np.random.seed calls from user scripts.
    return int(np.random.randint(0, _MAX_SEED, dtype=np.int64))


def _make_connections_density(
    variable_list: List[VariableAgent], density: float, *, seed: int | None = None
) -> List[Tuple[VariableAgent, VariableAgent]]:
    """Creates a random graph of variable connections based on a given density."""
    graph_seed = _resolve_graph_seed(seed)
    rng = random.Random(graph_seed)
    num_vars = len(variable_list)
    r_graph = nx.erdos_renyi_graph(num_vars, density, seed=graph_seed)
    if num_vars > 1 and not nx.is_connected(r_graph):
        components = list(nx.connected_components(r_graph))
        # Connect components sequentially to ensure a single connected component.
        for comp_a, comp_b in zip(components, components[1:]):
            u = rng.choice(tuple(comp_a))
            v = rng.choice(tuple(comp_b))
            r_graph.add_edge(u, v)
    variable_map = dict(enumerate(variable_list))
    full_graph = nx.relabel_nodes(r_graph, variable_map)
    return list(full_graph.edges())


class FGBuilder:
    """A builder class providing static methods to construct factor graphs."""

    @staticmethod
    def build_from_edges(
        variables: List[VariableAgent],
        factors: List[FactorAgent],
        edges: Dict[FactorAgent, List[VariableAgent]],
    ) -> FactorGraph:
        """Builds a factor graph from the provided variables, factors, and edges.

        Args:
            variables (List[VariableAgent]): The variable nodes in the graph.
            factors (List[FactorAgent]): The factor nodes in the graph.
            edges (Dict[FactorAgent, List[VariableAgent]]): The edges connecting factors to variables.

        Returns:
            FactorGraph: The constructed factor graph.
        """

        return FactorGraph(variables, factors, edges)

    @staticmethod
    def build_random_graph(
        num_vars: int,
        domain_size: int,
        ct_factory: Callable | str,
        ct_params: Dict[str, Any],
        density: float,
        *,
        seed: int | None = None,
    ) -> FactorGraph:
        """Builds a factor graph with random binary constraints.

        Args:
            num_vars: The number of variables in the graph.
            domain_size: The size of the domain for each variable.
            ct_factory: The factory for creating cost tables.
            ct_params: Parameters for the cost table factory.
            density: The density of the graph (probability of an edge).
            seed: Optional seed controlling the random topology. When omitted,
                randomness is derived from the globally-configured numpy and
                ``random`` RNGs so user-level seeding still produces
                deterministic graphs.

        Returns:
            A `FactorGraph` instance with a random topology.
        """
        variables = [_make_variable(i + 1, domain_size) for i in range(num_vars)]
        connections = _make_connections_density(variables, density, seed=seed)
        edges = _build_factor_edge_list(connections, domain_size, ct_factory, ct_params)
        factors = list(edges.keys())
        return FactorGraph(variables, factors, edges)

    @staticmethod
    def build_cycle_graph(
        num_vars: int,
        domain_size: int,
        ct_factory: Callable | str,
        ct_params: Dict[str, Any],
        **kwargs,
    ) -> FactorGraph:
        """Builds a factor graph with a simple cycle topology.

        The graph structure is `x1 – f12 – x2 – ... – xn – fn1 – x1`.

        Args:
            num_vars: The number of variables in the cycle.
            domain_size: The size of the domain for each variable.
            ct_factory: The factory for creating cost tables.
            ct_params: Parameters for the cost table factory.
            **kwargs: Catches unused arguments like `density` for API consistency.

        Returns:
            A `FactorGraph` instance with a cycle topology.
        """
        variables = [_make_variable(i + 1, domain_size) for i in range(num_vars)]
        edges = {}
        for j in range(num_vars):
            a, b = variables[j], variables[(j + 1) % num_vars]
            f_name = f"f{a.name[1:]}{b.name[1:]}"
            f_node = _make_factor(f_name, domain_size, ct_factory, ct_params)
            edges[f_node] = [a, b]
        factors = list(edges.keys())
        return FactorGraph(variables, factors, edges)

    @staticmethod
    def build_lemniscate_graph(
        num_vars: int,
        domain_size: int,
        ct_factory: Callable | str,
        ct_params: Dict[str, Any],
        **kwargs,
    ) -> FactorGraph:
        """Builds a factor graph with a lemniscate (∞) topology.

        The structure consists of two cycles that share a single central
        variable, producing a figure-eight shape. Each loop is guaranteed to
        contain at least two distinct variables in addition to the central node.

        Args:
            num_vars: Total number of variables in the graph. Must be >= 5.
            domain_size: The size of the domain for each variable.
            ct_factory: Factory used to create cost tables for the factors.
            ct_params: Parameters forwarded to the cost table factory.
            **kwargs: Captures unused parameters (e.g., density) for API parity.

        Returns:
            A `FactorGraph` instance shaped like a lemniscate.

        Raises:
            ValueError: If fewer than five variables are provided.
        """
        if num_vars < 5:
            raise ValueError(
                "Lemniscate graph requires at least 5 variables to form two loops."
            )

        variables = [_make_variable(i + 1, domain_size) for i in range(num_vars)]
        center = variables[0]
        remaining = variables[1:]

        left_size = max(2, len(remaining) // 2)
        right_size = len(remaining) - left_size
        if right_size < 2:
            shortage = 2 - right_size
            left_size -= shortage
            right_size += shortage

        if left_size < 2 or right_size < 2:
            raise ValueError(
                "Lemniscate graph requires at least 5 variables to form two loops."
            )

        left_loop_nodes = [center, *remaining[:left_size]]
        right_loop_nodes = [center, *remaining[left_size:]]

        edge_pairs: List[Tuple[VariableAgent, VariableAgent]] = []
        for loop in (left_loop_nodes, right_loop_nodes):
            edge_pairs.extend(
                (loop[idx], loop[idx + 1]) for idx in range(len(loop) - 1)
            )
            edge_pairs.append((loop[-1], loop[0]))

        params = ct_params or {}
        edges = _build_factor_edge_list(edge_pairs, domain_size, ct_factory, params)
        factors = list(edges.keys())
        return FactorGraph(variables, factors, edges)

    # Provide aliases for API compatibility/user preference.
    create_lemniscate_graph = build_lemniscate_graph
    create_leminscate_graph = build_lemniscate_graph


def get_message_shape(domain_size: int, connections: int = 2) -> tuple[int, ...]:
    """Calculates the shape of a cost table for a factor.

    Args:
        domain_size: The size of the domain for each connected variable.
        connections: The number of variables connected to the factor.

    Returns:
        A tuple representing the shape of the cost table.
    """
    return (domain_size,) * connections


@lru_cache(maxsize=128)
def get_broadcast_shape(ct_dims: int, domain_size: int, ax: int) -> tuple[int, ...]:
    """Calculates the shape for broadcasting a message into a cost table."""
    shape = [1] * ct_dims
    shape[ax] = domain_size
    return tuple(shape)


def generate_random_cost(fg: FactorGraph) -> float:
    """Calculates a total cost based on a random assignment for each factor.

    Args:
        fg: The factor graph to evaluate.

    Returns:
        The sum of costs from a random assignment in each factor's cost table.
    """
    cost = 0.0
    for fact in fg.factors:
        random_index = tuple(
            np.random.randint(0, fact.domain, size=fact.cost_table.ndim)
        )
        cost += fact.cost_table[random_index]
    return cost


class SafeUnpickler(pickle.Unpickler):
    """A custom unpickler to handle module path changes during deserialization.

    This class overrides `find_class` to intercept and correct module paths
    that may have changed between the time of pickling and unpickling,
    preventing `ImportError` or `AttributeError`.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Finds a class, handling potential module path changes."""
        module_mapping = {
            "bp.factor_graph": "propflow.bp.factor_graph",
            "bp.agents": "propflow.core.agents",
            "bp.components": "propflow.core.components",
        }
        module = module_mapping.get(module, module)
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import {module}.{name}: {e}")
            return type(name, (), {})


def load_pickle_safely(file_path: str) -> Any:
    """Loads a pickle file using the `SafeUnpickler` to prevent import errors.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The deserialized object, or `None` if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            return SafeUnpickler(f).load()
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None


def repair_factor_graph(fg: FactorGraph) -> FactorGraph:
    """Attempts to repair a loaded factor graph by ensuring essential attributes exist.

    This is useful when unpickling older versions of `FactorGraph` objects
    that may be missing attributes added in newer versions.

    Args:
        fg: The `FactorGraph` object to repair.

    Returns:
        The repaired `FactorGraph` object.
    """
    if not hasattr(fg, "G") or fg.G is None:
        print("Initializing missing NetworkX graph")
        fg.G = nx.Graph()
        if hasattr(fg, "variables") and hasattr(fg, "factors"):
            fg.G.add_nodes_from(fg.variables)
            fg.G.add_nodes_from(fg.factors)
            for factor in fg.factors:
                if hasattr(factor, "connection_number"):
                    for var, dim in factor.connection_number.items():
                        fg.G.add_edge(factor, var, dim=dim)
    for node in fg.G.nodes():
        if not hasattr(node, "mailbox"):
            node.mailbox = []
        if (
            hasattr(node, "type")
            and node.type == "factor"
            and (not hasattr(node, "cost_table") or node.cost_table is None)
        ):
            try:
                if hasattr(node, "initiate_cost_table"):
                    node.initiate_cost_table()
            except Exception as e:
                print(f"Could not initialize cost table for {node}: {e}")
    return fg


def get_bound(factor_graph: FactorGraph, reduce_func: Callable = np.min) -> float:
    """Calculates a simple bound on the total cost of the factor graph.

    This is typically used to get a lower bound by summing the minimum values
    from each factor's cost table.

    Args:
        factor_graph: The factor graph to analyze.
        reduce_func: The function to apply to each cost table to get a single
            value (e.g., `np.min` for a lower bound, `np.max` for an upper bound).
            Defaults to `np.min`.

    Returns:
        The calculated bound.
    """
    bound = 0.0
    for factor in factor_graph.factors:
        if hasattr(factor, "cost_table") and factor.cost_table is not None:
            bound += reduce_func(factor.cost_table)
    return bound


def pretty_print_array(
    A,
    *,
    cmap="Blues",
    fmt="{:.2g}",
    annotate=True,
    auto_min=True,
    auto_max=True,
    cell_highlights=None,  # list[(r,c)] or dict[(r,c)]="label"
    row_highlights=None,  # list[row] or dict[row]="label"
    label_colors=None,  # dict[label]->color
    title=None,
    figsize=(6, 4),
    cbar=True,
):
    """
    Render a NumPy array as a heatmap with optional highlighting.

    Parameters
    ----------
    A : array-like (2D)
        Numeric data.
    cmap : str
        Matplotlib colormap name.
    fmt : str
        Number format for annotations, e.g. "{:.2g}".
    annotate : bool
        If True, draw numbers on each cell.
    auto_min : bool
        If True, highlight all global minima (ties included) with label 'min'.
    auto_max : bool
        If True, highlight all global maxima (ties included) with label 'max'.
    cell_highlights : list[(r,c)] | dict[(r,c)] -> str
        Cells to highlight. If list, uses label 'selected'.
        If dict, value is the legend label for that cell group.
    row_highlights : list[int] | dict[int] -> str
        Rows to highlight. If list, labels become 'row {i}'.
        If dict, value is the legend label for that row.
    label_colors : dict[str] -> str
        Custom colors per label. Unknown labels get auto-assigned.
    title : str
        Figure title.
    figsize : (w,h)
        Figure size in inches.
    cbar : bool
        If True, show colorbar.

    Returns
    -------
    fig, ax : Matplotlib figure and axes.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2D")

    nrows, ncols = A.shape
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(A, cmap=cmap, aspect="equal")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Grid lines
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels([str(j) for j in range(ncols)])
    ax.set_yticklabels([str(i) for i in range(nrows)])
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Normalize inputs
    used_labels = []
    default_label_colors = {
        "selected": "#FFB000",
        "row": "#FFD166",
        "min": "#00C2A0",
        "max": "#E63946",
    }
    if label_colors:
        default_label_colors.update(label_colors)

    # Build highlight maps
    cell_map = {}
    if cell_highlights is not None:
        if isinstance(cell_highlights, dict):
            for (r, c), lab in cell_highlights.items():
                cell_map.setdefault(lab, []).append((int(r), int(c)))
        else:  # assume list of (r,c)
            cell_map.setdefault("selected", []).extend(
                (int(r), int(c)) for r, c in cell_highlights
            )

    row_map = {}
    if row_highlights is not None:
        if isinstance(row_highlights, dict):
            for r, lab in row_highlights.items():
                row_map.setdefault(lab, []).append(int(r))
        else:  # assume list of rows
            for r in row_highlights:
                row_map.setdefault(f"row {int(r)}", []).append(int(r))

    # Auto min/max
    if auto_min:
        mn = np.min(A)
        mins = np.argwhere(A == mn)
        if len(mins):
            cell_map.setdefault("min", []).extend((int(r), int(c)) for r, c in mins)
    if auto_max:
        mx = np.max(A)
        maxs = np.argwhere(A == mx)
        if len(maxs):
            cell_map.setdefault("max", []).extend((int(r), int(c)) for r, c in maxs)

    # Draw highlights (filled translucent rectangles), then annotate
    def color_for(
        label,
        fallback_cycle=(
            "#FFB000",
            "#6A4C93",
            "#2A9D8F",
            "#E76F51",
            "#118AB2",
            "#EF476F",
        ),
    ):
        if label in default_label_colors:
            return default_label_colors[label]
        # Assign a stable color based on label hash
        return fallback_cycle[abs(hash(label)) % len(fallback_cycle)]

    # Row highlights
    for lab, rows in row_map.items():
        col = color_for(lab)
        for r in rows:
            rect = Rectangle(
                (-0.5, r - 0.5),
                ncols,
                1,
                linewidth=2,
                edgecolor=col,
                facecolor=col,
                alpha=0.25,
            )
            ax.add_patch(rect)
        used_labels.append((lab, col))

    # Cell highlights
    for lab, cells in cell_map.items():
        col = color_for(lab)
        for r, c in cells:
            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor=col,
                facecolor=col,
                alpha=0.35,
            )
            ax.add_patch(rect)
        used_labels.append((lab, col))

    # Annotations
    if annotate:
        # Pick contrasting text color vs background
        norm = im.norm
        for i in range(nrows):
            for j in range(ncols):
                val = A[i, j]
                # heuristic contrast: dark text on light cells, white text on dark cells
                txt_color = "white" if norm(val) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=10,
                )

    # Legend describing highlight colors
    if used_labels:
        # Deduplicate while preserving order
        seen = set()
        handles = []
        for lab, col in used_labels:
            if (lab, col) in seen:
                continue
            seen.add((lab, col))
            handles.append(Patch(facecolor=col, edgecolor=col, alpha=0.6, label=lab))
        ax.legend(
            handles=handles,
            title="Highlights",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax
