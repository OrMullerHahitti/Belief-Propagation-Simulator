"""Tools for convex hull analysis of cost functions in belief propagation.

This module provides data structures and functions for computing and visualizing
the convex hull of cost functions, which is a technique used in hierarchical
models and for analyzing the behavior of certain message-passing algorithms.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Literal, Any
import math

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ...core.agents import VariableAgent, FactorAgent
from ...core.components import CostTable

EPSILON = 1e-9


@dataclass(frozen=True, eq=True)
class ConvexLine:
    """Represents a line y = slope * k + intercept for convex hull computation.

    This is used in the dual space where each cell of a cost table defines a line.

    Attributes:
        slope: The coefficient of k, derived from a cost table cell.
        intercept: The constant term, typically from a Q-message value.
        cell_i: The row index in the original cost table.
        cell_j: The column index in the original cost table.
    """
    slope: float
    intercept: float
    cell_i: int
    cell_j: int

    def evaluate(self, k: float) -> float:
        """Evaluates the line equation at a given point k."""
        if math.isnan(k) or math.isinf(k):
            return float("nan")
        val = self.slope * k + self.intercept
        return val if math.isfinite(val) else float("nan")


@dataclass(frozen=True, eq=True)
class InterceptPoint:
    """Represents an intersection point between lines or envelopes.

    Attributes:
        k: The k-value (x-coordinate) where the intersection occurs.
        intersection_value: The y-coordinate of the intersection.
        type: The type of intercept, indicating if it's a change within an
            envelope or between envelopes.
        envelope1_id: The ID of the first envelope involved.
        envelope2_id: The ID of the second envelope involved.
        line1: The first `ConvexLine` object at the intersection.
        line2: The second `ConvexLine` object at the intersection.
    """
    k: float
    intersection_value: float
    type: Literal["change_assignment", "partial_assignment_change"]
    envelope1_id: int
    envelope2_id: int
    line1: Optional[ConvexLine] = None
    line2: Optional[ConvexLine] = None


@dataclass
class ConvexHullResult:
    """Stores the results of a convex hull computation.

    Attributes:
        lines: All lines generated from the cost table.
        hull_lines: The subset of lines that form the convex hull.
        hull_vertices: The vertices (slope, intercept) of the convex hull.
        k_range: The range of k values used for the computation.
        envelope_id: A unique identifier for this specific envelope.
    """
    lines: List[ConvexLine]
    hull_lines: List[ConvexLine]
    hull_vertices: np.ndarray
    k_range: Tuple[float, float]
    envelope_id: int = 0


@dataclass
class HierarchicalEnvelopeResult:
    """Stores the results of a hierarchical convex hull computation.

    Attributes:
        individual_envelopes: A list of the individual convex hulls.
        meta_envelope: The convex hull of all the individual envelopes.
        all_intercepts: A sorted list of all found intercept points.
        change_assignment_points: Intercepts that occur between different envelopes.
        partial_change_points: Intercepts that occur within a single envelope.
        envelope_type: The type of envelope computed ('lower' or 'upper').
    """
    individual_envelopes: List[ConvexHullResult]
    meta_envelope: ConvexHullResult
    all_intercepts: List[InterceptPoint]
    change_assignment_points: List[InterceptPoint]
    partial_change_points: List[InterceptPoint]
    envelope_type: str


def create_lines_from_cost_table(
    cost_table: CostTable, q_values: np.ndarray, k_min: float = 0.0, k_max: float = 1.0
) -> List[ConvexLine]:
    """Creates a list of `ConvexLine` objects from a cost table and q-values.

    Each cell `(i, j)` in the cost table defines a line `y = c_ij * k + q_i`.

    Args:
        cost_table: A 2D numpy array of costs.
        q_values: A 1D numpy array of intercepts (q-values).
        k_min: The minimum k-value for the analysis range.
        k_max: The maximum k-value for the analysis range.

    Returns:
        A list of `ConvexLine` objects.
    """
    if not isinstance(cost_table, np.ndarray) or cost_table.ndim != 2:
        raise ValueError("cost_table must be a 2D numpy array")
    if not isinstance(q_values, np.ndarray) or q_values.ndim != 1:
        raise ValueError("q_values must be a 1D numpy array")
    if cost_table.shape[0] != len(q_values):
        raise ValueError("cost_table rows must match q_values length")
    if k_min >= k_max:
        raise ValueError("k_min must be less than k_max")

    lines = []
    for i, row in enumerate(cost_table):
        for j, slope in enumerate(row):
            intercept = q_values[i]
            if math.isfinite(slope) and math.isfinite(intercept):
                lines.append(ConvexLine(slope=slope, intercept=intercept, cell_i=i, cell_j=j))
    return lines


def compute_convex_hull_from_lines(
    lines: List[ConvexLine], hull_type: str = "lower"
) -> ConvexHullResult:
    """Computes the convex hull from a list of `ConvexLine` objects.

    Args:
        lines: The list of lines to compute the hull from.
        hull_type: The type of hull, either 'lower' or 'upper'.

    Returns:
        A `ConvexHullResult` containing the results.
    """
    if not lines: raise ValueError("Cannot compute convex hull from empty line list")
    if len(lines) == 1:
        return ConvexHullResult(lines=lines, hull_lines=lines, hull_vertices=np.array([[lines[0].slope, lines[0].intercept]]), k_range=(0.0, 1.0))

    points = np.array([[line.slope, -line.intercept if hull_type == "upper" else line.intercept] for line in lines])

    try:
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        hull_lines = sorted([lines[i] for i in hull_indices], key=lambda x: x.slope)
        hull_vertices = points[hull_indices]
        if hull_type == "upper":
            hull_vertices[:, 1] *= -1
        return ConvexHullResult(lines=lines, hull_lines=hull_lines, hull_vertices=hull_vertices, k_range=(0.0, 1.0))
    except QhullError as e:
        print(f"Warning: ConvexHull failed ({e}). Using all lines.")
        return ConvexHullResult(lines=lines, hull_lines=lines, hull_vertices=points, k_range=(0.0, 1.0))


def hierarchical_convex_hull_from_agents(
    variable_agents: List[VariableAgent], factor_agents: List[FactorAgent],
    k_min: float = 0.0, k_max: float = 1.0
) -> HierarchicalEnvelopeResult:
    """Creates a hierarchical convex hull from multiple variable and factor agents.

    This function automatically detects the envelope type (lower/upper) based
    on the agents' computators.

    Args:
        variable_agents: A list of variable agents.
        factor_agents: A list of factor agents.
        k_min: The minimum k-value for the analysis range.
        k_max: The maximum k-value for the analysis range.

    Returns:
        A `HierarchicalEnvelopeResult` containing the full analysis.
    """
    if not variable_agents or not factor_agents:
        raise ValueError("Need at least one variable agent and one factor agent")

    envelope_type = "lower"
    for agent in variable_agents + factor_agents:
        detected_type = determine_envelope_type(agent)
        if detected_type != "lower":
            envelope_type = detected_type
            break

    individual_hulls = []
    for i, var_agent in enumerate(variable_agents):
        for j, factor_agent in enumerate(factor_agents):
            if factor_agent.connection_number and var_agent.name in factor_agent.connection_number:
                hull = convex_hull_from_agents(var_agent, factor_agent, envelope_type, k_min, k_max)
                hull.envelope_id = i * len(factor_agents) + j
                individual_hulls.append(hull)

    return compute_hierarchical_envelopes(individual_hulls, envelope_type, k_min, k_max)


def convex_hull_from_agents(
    variable_agent: VariableAgent, factor_agent: FactorAgent,
    hull_type: Optional[str] = None, k_min: float = 0.0, k_max: float = 1.0
) -> ConvexHullResult:
    """Creates a convex hull for a single variable-factor pair.

    Args:
        variable_agent: The `VariableAgent` with domain information.
        factor_agent: The `FactorAgent` with the cost table.
        hull_type: The type of hull ('lower' or 'upper'). If None, it's auto-detected.
        k_min: The minimum k-value for the analysis range.
        k_max: The maximum k-value for the analysis range.

    Returns:
        A `ConvexHullResult` for the given pair.
    """
    if factor_agent.cost_table is None:
        raise ValueError("FactorAgent must have a cost table")
    if hull_type is None:
        hull_type = determine_envelope_type(variable_agent)

    q_values = np.zeros(variable_agent.domain)
    if hasattr(variable_agent, "belief") and variable_agent.belief is not None:
        try:
            if len(variable_agent.belief) == variable_agent.domain:
                q_values = variable_agent.belief
        except Exception:
            pass

    lines = create_lines_from_cost_table(factor_agent.cost_table, q_values, k_min, k_max)
    return compute_convex_hull_from_lines(lines, hull_type)


def convex_hull_from_cost_table(
    cost_table: CostTable, q_values: Optional[np.ndarray] = None,
    hull_type: str = "lower", k_min: float = 0.0, k_max: float = 1.0
) -> ConvexHullResult:
    """Creates a convex hull directly from a cost table and q-values.

    Args:
        cost_table: A 2D numpy array of costs.
        q_values: A 1D numpy array of intercepts. If None, zeros are used.
        hull_type: The type of hull ('lower' or 'upper').
        k_min: The minimum k-value for the analysis range.
        k_max: The maximum k-value for the analysis range.

    Returns:
        A `ConvexHullResult` object.
    """
    if q_values is None:
        q_values = np.zeros(cost_table.shape[0])
    lines = create_lines_from_cost_table(cost_table, q_values, k_min, k_max)
    return compute_convex_hull_from_lines(lines, hull_type)


def find_line_intersection(line1: ConvexLine, line2: ConvexLine) -> Optional[Tuple[float, float]]:
    """Finds the intersection point (k, value) between two lines.

    Args:
        line1: The first line.
        line2: The second line.

    Returns:
        A tuple (k, value) of the intersection, or `None` if parallel.
    """
    slope_diff = line1.slope - line2.slope
    if abs(slope_diff) < EPSILON:
        return None
    k_intersect = (line2.intercept - line1.intercept) / slope_diff
    if not math.isfinite(k_intersect):
        return None
    return k_intersect, line1.evaluate(k_intersect)


def find_all_envelope_intercepts(hull_result: ConvexHullResult) -> List[InterceptPoint]:
    """Finds all intersection points between adjacent lines on a single convex hull.

    Args:
        hull_result: The `ConvexHullResult` to analyze.

    Returns:
        A list of `InterceptPoint` objects, marked as 'partial_assignment_change'.
    """
    if len(hull_result.hull_lines) <= 1:
        return []
    intercepts = []
    sorted_lines = sorted(hull_result.hull_lines, key=lambda x: x.slope)
    for i in range(len(sorted_lines) - 1):
        line1, line2 = sorted_lines[i], sorted_lines[i + 1]
        intersection = find_line_intersection(line1, line2)
        if intersection:
            k_val, intersection_value = intersection
            k_min, k_max = hull_result.k_range
            if k_min <= k_val <= k_max:
                intercepts.append(InterceptPoint(
                    k=k_val, intersection_value=intersection_value, type="partial_assignment_change",
                    envelope1_id=hull_result.envelope_id, envelope2_id=hull_result.envelope_id,
                    line1=line1, line2=line2
                ))
    return intercepts


def determine_envelope_type(agent: Union[VariableAgent, FactorAgent]) -> str:
    """Determines the envelope type ('lower' or 'upper') from an agent's computator.

    Args:
        agent: The agent to inspect.

    Returns:
        'lower' for Min-Sum computators, 'upper' for Max-Sum, otherwise 'lower'.
    """
    computator = getattr(agent, "computator", None)
    if computator and hasattr(computator, "reduce_func"):
        reduce_func = computator.reduce_func
        if reduce_func == np.max:
            return "upper"
    return "lower"


def compute_hierarchical_envelopes(
    individual_hulls: List[ConvexHullResult], envelope_type: str = "lower",
    k_min: float = 0.0, k_max: float = 1.0
) -> HierarchicalEnvelopeResult:
    """Computes the meta-envelope of a set of individual envelopes and classifies all intercepts.

    Args:
        individual_hulls: A list of pre-computed `ConvexHullResult` objects.
        envelope_type: The type of envelope to compute ('lower' or 'upper').
        k_min: The minimum k-value for the analysis.
        k_max: The maximum k-value for the analysis.

    Returns:
        A `HierarchicalEnvelopeResult` containing the full analysis.
    """
    if not individual_hulls:
        raise ValueError("Cannot compute hierarchical envelopes from empty hull list")

    for i, hull in enumerate(individual_hulls):
        hull.envelope_id = i

    partial_change_points = [icept for hull in individual_hulls for icept in find_all_envelope_intercepts(hull)]

    meta_lines, envelope_line_mapping = [], {}
    for hull in individual_hulls:
        for line in hull.hull_lines:
            meta_line = ConvexLine(slope=line.slope, intercept=line.intercept, cell_i=hull.envelope_id, cell_j=len(meta_lines))
            envelope_line_mapping[len(meta_lines)] = (hull.envelope_id, line)
            meta_lines.append(meta_line)

    meta_hull = compute_convex_hull_from_lines(meta_lines, envelope_type)
    meta_hull.envelope_id = -1

    change_assignment_points = []
    meta_sorted_lines = sorted(meta_hull.hull_lines, key=lambda x: x.slope)
    for i in range(len(meta_sorted_lines) - 1):
        line1, line2 = meta_sorted_lines[i], meta_sorted_lines[i + 1]
        if line1.cell_i != line2.cell_i:
            intersection = find_line_intersection(line1, line2)
            if intersection:
                k_val, intersection_value = intersection
                if k_min <= k_val <= k_max:
                    orig_line1 = envelope_line_mapping.get(line1.cell_j, (None, None))[1]
                    orig_line2 = envelope_line_mapping.get(line2.cell_j, (None, None))[1]
                    change_assignment_points.append(InterceptPoint(
                        k=k_val, intersection_value=intersection_value, type="change_assignment",
                        envelope1_id=line1.cell_i, envelope2_id=line2.cell_i,
                        line1=orig_line1, line2=orig_line2
                    ))

    all_intercepts = sorted(partial_change_points + change_assignment_points, key=lambda x: x.k)
    return HierarchicalEnvelopeResult(
        individual_envelopes=individual_hulls, meta_envelope=meta_hull, all_intercepts=all_intercepts,
        change_assignment_points=change_assignment_points, partial_change_points=partial_change_points,
        envelope_type=envelope_type,
    )


def evaluate_hull_at_k(hull_result: ConvexHullResult, k: float) -> Tuple[float, ConvexLine]:
    """DEPRECATED: Evaluates the convex hull at a specific k-value.

    Args:
        hull_result: The result from a convex hull computation.
        k: The k-value to evaluate at.

    Returns:
        A tuple of (value, line) where line is the active line at k.
    """
    if not hull_result.hull_lines: raise ValueError("No hull lines available")
    if len(hull_result.hull_lines) == 1:
        line = hull_result.hull_lines[0]
        return line.evaluate(k), line

    values = [line.evaluate(k) for line in hull_result.hull_lines]
    valid_values = [v for v in values if math.isfinite(v)]
    if not valid_values: return float('inf'), hull_result.hull_lines[0]

    best_value = min(valid_values)
    best_line = hull_result.hull_lines[values.index(best_value)]
    return best_value, best_line


def plot_convex_hull(
    hull_result: ConvexHullResult, k_min: float = 0.0, k_max: float = 1.0,
    title: str = "Convex Hull", show_all_lines: bool = True,
    show_intercepts: bool = True, ax: Optional["plt.Axes"] = None
) -> "plt.Axes":
    """Plots a single convex hull, its constituent lines, and intercept points.

    Args:
        hull_result: The `ConvexHullResult` to plot.
        k_min: The minimum k-value for the plot's x-axis.
        k_max: The maximum k-value for the plot's x-axis.
        title: The title of the plot.
        show_all_lines: If True, plots all lines, not just those on the hull.
        show_intercepts: If True, marks the intersection points on the hull.
        ax: An existing matplotlib Axes object to plot on. If None, creates a new figure.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if not HAS_MATPLOTLIB: raise ImportError("Matplotlib is required for visualization.")
    if ax is None: _, ax = plt.subplots(1, 1, figsize=(10, 6))
    k_vals = np.linspace(k_min, k_max, 100)

    if show_all_lines:
        for i, line in enumerate(hull_result.lines):
            ax.plot(k_vals, [line.evaluate(k) for k in k_vals], "--", alpha=0.3, color="gray", label=f"Line {line.cell_i}{line.cell_j}" if i < 5 else "")

    colors = plt.cm.Set1(np.linspace(0, 1, len(hull_result.hull_lines)))
    for i, line in enumerate(hull_result.hull_lines):
        ax.plot(k_vals, [line.evaluate(k) for k in k_vals], "-", linewidth=2, color=colors[i], label=f"Hull Line {line.cell_i}{line.cell_j}")

    if show_intercepts:
        intercepts = find_all_envelope_intercepts(hull_result)
        for i, intercept in enumerate(intercepts):
            if k_min <= intercept.k <= k_max:
                ax.plot(intercept.k, intercept.intersection_value, "ro", markersize=8, label="Intercept" if i == 0 else "")
                ax.annotate(f"k={intercept.k:.3f}", (intercept.k, intercept.intersection_value), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("k"); ax.set_ylabel("Value"); ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    return ax


def plot_hierarchical_envelopes(
    hierarchical_result: HierarchicalEnvelopeResult, k_min: float = 0.0,
    k_max: float = 1.0, figsize: Tuple[int, int] = (15, 10)
) -> "plt.Figure":
    """Plots a comprehensive visualization of a hierarchical envelope system.

    This creates a multi-panel figure showing each individual envelope, the
    meta-envelope, and a summary of all intercept points.

    Args:
        hierarchical_result: The `HierarchicalEnvelopeResult` to plot.
        k_min: The minimum k-value for the plot's x-axis.
        k_max: The maximum k-value for the plot's x-axis.
        figsize: The size of the figure.

    Returns:
        The matplotlib Figure object containing the plots.
    """
    if not HAS_MATPLOTLIB: raise ImportError("Matplotlib is required for visualization.")
    n_individual = len(hierarchical_result.individual_envelopes)
    fig = plt.figure(figsize=figsize)
    n_cols = min(3, n_individual)
    n_rows = (n_individual + n_cols - 1) // n_cols

    for i, hull in enumerate(hierarchical_result.individual_envelopes):
        ax = fig.add_subplot(n_rows + 2, n_cols, i + 1)
        plot_convex_hull(hull, k_min, k_max, f"Envelope {hull.envelope_id}", show_all_lines=False, show_intercepts=True, ax=ax)

    ax_meta = fig.add_subplot(n_rows + 2, 1, n_rows + 1)
    plot_convex_hull(hierarchical_result.meta_envelope, k_min, k_max, "Meta-Envelope (Envelope of Envelopes)", show_all_lines=True, show_intercepts=False, ax=ax_meta)

    ax_intercepts = fig.add_subplot(n_rows + 2, 1, n_rows + 2)
    plot_intercept_summary(hierarchical_result, k_min, k_max, ax=ax_intercepts)

    plt.tight_layout()
    return fig


def plot_intercept_summary(
    hierarchical_result: HierarchicalEnvelopeResult, k_min: float = 0.0,
    k_max: float = 1.0, ax: Optional["plt.Axes"] = None
) -> "plt.Axes":
    """Plots a summary of all intercept points, classified by type.

    Args:
        hierarchical_result: The `HierarchicalEnvelopeResult` containing the intercepts.
        k_min: The minimum k-value for the plot's x-axis.
        k_max: The maximum k-value for the plot's x-axis.
        ax: An existing matplotlib Axes object to plot on. If None, creates a new figure.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if not HAS_MATPLOTLIB: raise ImportError("Matplotlib is required for visualization.")
    if ax is None: _, ax = plt.subplots(1, 1, figsize=(12, 6))
    k_vals = np.linspace(k_min, k_max, 100)

    for i, line in enumerate(hierarchical_result.meta_envelope.hull_lines):
        ax.plot(k_vals, [line.evaluate(k) for k in k_vals], "-", linewidth=2, color=plt.cm.tab10(i), label=f"Meta-Hull Line (Env {line.cell_i})")

    change_points = [p for p in hierarchical_result.all_intercepts if p.type == "change_assignment" and k_min <= p.k <= k_max]
    partial_points = [p for p in hierarchical_result.all_intercepts if p.type == "partial_assignment_change" and k_min <= p.k <= k_max]

    if change_points:
        ax.scatter([p.k for p in change_points], [p.intersection_value for p in change_points], c="red", s=100, marker="o", label="Change Assignment", zorder=5, edgecolors="black", linewidth=1)
        for p in change_points:
            ax.annotate(f"k={p.k:.3f}\nEnv {p.envelope1_id}→{p.envelope2_id}", (p.k, p.intersection_value), xytext=(5, 10), textcoords="offset points", fontsize=8, ha="left", bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    if partial_points:
        ax.scatter([p.k for p in partial_points], [p.intersection_value for p in partial_points], c="blue", s=80, marker="^", label="Partial Assignment Change", zorder=5, edgecolors="black", linewidth=1)
        for p in partial_points:
            ax.annotate(f"k={p.k:.3f}\nWithin Env {p.envelope1_id}", (p.k, p.intersection_value), xytext=(5, -15), textcoords="offset points", fontsize=8, ha="left", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    ax.set_xlabel("k"); ax.set_ylabel("Value"); ax.set_title("Intercept Classification Summary"); ax.grid(True, alpha=0.3); ax.legend()
    return ax


def plot_envelope_comparison(
    hull_results: List[ConvexHullResult], labels: Optional[List[str]] = None,
    k_min: float = 0.0, k_max: float = 1.0, title: str = "Envelope Comparison"
) -> "plt.Figure":
    """Plots multiple convex hulls on the same axes for comparison.

    Args:
        hull_results: A list of `ConvexHullResult` objects to plot.
        labels: Optional labels for each hull.
        k_min: The minimum k-value for the plot's x-axis.
        k_max: The maximum k-value for the plot's x-axis.
        title: The title for the plot.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    if not HAS_MATPLOTLIB: raise ImportError("Matplotlib is required for visualization.")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    k_vals = np.linspace(k_min, k_max, 100)
    colors = plt.cm.Set1(np.linspace(0, 1, len(hull_results)))

    for i, hull in enumerate(hull_results):
        label_prefix = labels[i] if labels else f"Envelope {hull.envelope_id}"
        for j, line in enumerate(hull.hull_lines):
            ax.plot(k_vals, [line.evaluate(k) for k in k_vals], "-" if j == 0 else "--", color=colors[i], linewidth=2, label=f"{label_prefix} Line {j+1}" if j < 2 else "")
        for intercept in find_all_envelope_intercepts(hull):
            if k_min <= intercept.k <= k_max:
                ax.plot(intercept.k, intercept.intersection_value, "o", color=colors[i], markersize=6, markeredgecolor="black")

    ax.set_xlabel("k"); ax.set_ylabel("Value"); ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    return fig


def save_visualization(
    hierarchical_result: HierarchicalEnvelopeResult, filename: str,
    k_min: float = 0.0, k_max: float = 1.0, dpi: int = 300
) -> None:
    """Saves a hierarchical envelope visualization to a file.

    Args:
        hierarchical_result: The `HierarchicalEnvelopeResult` to visualize.
        filename: The output filename (e.g., 'my_plot.png').
        k_min: The minimum k-value for the plot.
        k_max: The maximum k-value for the plot.
        dpi: The resolution of the saved image.
    """
    if not HAS_MATPLOTLIB: raise ImportError("Matplotlib is required for visualization.")
    fig = plot_hierarchical_envelopes(hierarchical_result, k_min, k_max)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {filename}")


if __name__ == "__main__":
    def example_usage():
        """Runs an example demonstrating the hierarchical convex hull computation and visualization."""
        print("=== Hierarchical Convex Hull Example ===")
        cost_table1 = np.array([[1.0, 3.0], [2.0, 1.5]]); q_values1 = np.array([0.5, 1.0])
        cost_table2 = np.array([[0.5, 2.5], [3.0, 1.0]]); q_values2 = np.array([1.0, 0.5])
        hull1 = convex_hull_from_cost_table(cost_table1, q_values1, hull_type="lower")
        hull2 = convex_hull_from_cost_table(cost_table2, q_values2, hull_type="lower")
        hierarchical_result = compute_hierarchical_envelopes([hull1, hull2], envelope_type="lower")
        print(f"Meta-envelope has {len(hierarchical_result.meta_envelope.hull_lines)} lines on hull")
        if HAS_MATPLOTLIB:
            save_visualization(hierarchical_result, "hierarchical_envelopes.png")
            plt.show()
    example_usage()
