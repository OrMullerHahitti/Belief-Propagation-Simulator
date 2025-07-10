import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import math

from src.propflow.base_models.agents import VariableAgent, FactorAgent
from src.propflow.base_models.components import CostTable

# Global constant for numerical stability
EPSILON = 1e-9


@dataclass(frozen=True, eq=True)
class ConvexLine:
    """Represents a line y = slope * k + intercept for convex hull computation."""

    slope: float          # Coefficient of k (from cost table cell)
    intercept: float      # Constant term q_i
    cell_i: int          # Row index in cost table
    cell_j: int          # Column index in cost table

    def evaluate(self, k: float) -> float:
        """Evaluates the line equation at a given k."""
        if math.isnan(k) or math.isinf(k):
            return float("nan")
        val = self.slope * k + self.intercept
        if math.isnan(val) or math.isinf(val):
            return float("nan")
        return val


@dataclass
class ConvexHullResult:
    """Result of convex hull computation."""

    lines: List[ConvexLine]              # All generated lines
    hull_lines: List[ConvexLine]         # Lines on the convex hull
    hull_vertices: np.ndarray            # Hull vertices as (slope, intercept) points
    k_range: Tuple[float, float]         # Range of k values used


def create_lines_from_cost_table(
    cost_table: CostTable,
    q_values: np.ndarray,
    k_min: float = 0.0,
    k_max: float = 1.0
) -> List[ConvexLine]:
    """
    Create lines l_ij = k * c_ij + q_i for each cell in the cost table.

    Args:
        cost_table: 2D array where cost_table[i,j] represents cost from state i to state j
        q_values: 1D array where q_values[i] represents the intercept for state i
        k_min: Minimum k value (default 0.0)
        k_max: Maximum k value (default 1.0)

    Returns:
        List of ConvexLine objects, one for each cell in the cost table
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
    rows, cols = cost_table.shape

    for i in range(rows):
        for j in range(cols):
            slope = cost_table[i, j]
            intercept = q_values[i]

            # Skip lines with invalid slope or intercept
            if not (math.isfinite(slope) and math.isfinite(intercept)):
                continue

            lines.append(ConvexLine(
                slope=slope,
                intercept=intercept,
                cell_i=i,
                cell_j=j
            ))

    return lines


def compute_convex_hull_from_lines(
    lines: List[ConvexLine],
    hull_type: str = "lower"
) -> ConvexHullResult:
    """
    Compute convex hull from a list of lines.

    Args:
        lines: List of ConvexLine objects
        hull_type: "lower" for lower convex hull, "upper" for upper convex hull

    Returns:
        ConvexHullResult containing hull information
    """
    if not lines:
        raise ValueError("Cannot compute convex hull from empty line list")

    if len(lines) == 1:
        return ConvexHullResult(
            lines=lines,
            hull_lines=lines,
            hull_vertices=np.array([[lines[0].slope, lines[0].intercept]]),
            k_range=(0.0, 1.0)
        )

    # Create points for hull computation: (slope, intercept)
    # For upper hull, negate intercept
    points = []
    for line in lines:
        if hull_type == "upper":
            points.append([line.slope, -line.intercept])
        else:
            points.append([line.slope, line.intercept])

    points = np.array(points)

    try:
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        hull_lines = [lines[i] for i in hull_indices]

        # Sort hull lines by slope
        hull_lines.sort(key=lambda x: x.slope)

        # Get hull vertices, restoring original intercept for upper hull
        hull_vertices = points[hull_indices]
        if hull_type == "upper":
            hull_vertices[:, 1] = -hull_vertices[:, 1]
            # Also fix the hull_lines intercepts
            hull_lines = [
                ConvexLine(line.slope, line.intercept, line.cell_i, line.cell_j)
                for line in hull_lines
            ]

        return ConvexHullResult(
            lines=lines,
            hull_lines=hull_lines,
            hull_vertices=hull_vertices,
            k_range=(0.0, 1.0)
        )

    except QhullError as e:
        # Handle collinear points or other precision issues
        print(f"Warning: ConvexHull failed ({e}). Using all lines.")
        return ConvexHullResult(
            lines=lines,
            hull_lines=lines,
            hull_vertices=points,
            k_range=(0.0, 1.0)
        )


def convex_hull_from_agents(
    variable_agent: VariableAgent,
    factor_agent: FactorAgent,
    hull_type: str = "lower",
    k_min: float = 0.0,
    k_max: float = 1.0
) -> ConvexHullResult:
    """
    Create convex hull from variable and factor agents.

    Args:
        variable_agent: VariableAgent with domain information
        factor_agent: FactorAgent with cost table
        hull_type: "lower" for lower convex hull, "upper" for upper convex hull
        k_min: Minimum k value (default 0.0)
        k_max: Maximum k value (default 1.0)

    Returns:
        ConvexHullResult containing hull information
    """
    if factor_agent.cost_table is None:
        raise ValueError("FactorAgent must have a cost table")

    # Check if variable agent is connected to this factor agent
    var_dimension = None
    if factor_agent.connection_number and variable_agent.name in factor_agent.connection_number:
        var_dimension = factor_agent.connection_number[variable_agent.name]
        print(f"Variable {variable_agent.name} connected to factor {factor_agent.name} at dimension {var_dimension}")

    # Create q_values based on variable agent domain
    # For now, use uniform q values (can be extended based on beliefs or other criteria)
    q_values = np.zeros(variable_agent.domain)

    # If variable agent has beliefs, use them as q values
    if hasattr(variable_agent, 'belief') and variable_agent.belief is not None:
        try:
            belief = variable_agent.belief
            if len(belief) == variable_agent.domain:
                q_values = belief
        except:
            # Fallback to zeros if belief computation fails
            pass

    # If we have connection information, we might want to slice the cost table
    # appropriately based on the variable's dimension in the factor
    cost_table_to_use = factor_agent.cost_table

    # For multi-dimensional cost tables, we might need to marginalize or slice
    # This is a simplified approach - in practice, you might want more sophisticated handling
    if var_dimension is not None and len(cost_table_to_use.shape) > 2:
        print(f"Warning: Multi-dimensional cost table detected. Using full table for convex hull.")

    lines = create_lines_from_cost_table(
        cost_table_to_use,
        q_values,
        k_min,
        k_max
    )

    return compute_convex_hull_from_lines(lines, hull_type)


def convex_hull_from_cost_table(
    cost_table: CostTable,
    q_values: Optional[np.ndarray] = None,
    hull_type: str = "lower",
    k_min: float = 0.0,
    k_max: float = 1.0
) -> ConvexHullResult:
    """
    Create convex hull directly from cost table and q values.

    Args:
        cost_table: 2D cost table
        q_values: 1D array of intercept values (if None, uses zeros)
        hull_type: "lower" for lower convex hull, "upper" for upper convex hull
        k_min: Minimum k value (default 0.0)
        k_max: Maximum k value (default 1.0)

    Returns:
        ConvexHullResult containing hull information
    """
    if q_values is None:
        q_values = np.zeros(cost_table.shape[0])

    lines = create_lines_from_cost_table(cost_table, q_values, k_min, k_max)
    return compute_convex_hull_from_lines(lines, hull_type)


def evaluate_hull_at_k(hull_result: ConvexHullResult, k: float) -> Tuple[float, ConvexLine]:
    """
    Evaluate the convex hull at a specific k value.

    Args:
        hull_result: Result from convex hull computation
        k: K value to evaluate at

    Returns:
        Tuple of (value, line) where line is the active line at k
    """
    if not hull_result.hull_lines:
        raise ValueError("No hull lines available")

    if len(hull_result.hull_lines) == 1:
        line = hull_result.hull_lines[0]
        return line.evaluate(k), line

    # Find the line that gives the minimum (or maximum) value at k
    best_value = float('inf')
    best_line = hull_result.hull_lines[0]

    for line in hull_result.hull_lines:
        value = line.evaluate(k)
        if math.isfinite(value) and value < best_value:
            best_value = value
            best_line = line

    return best_value, best_line


# Example usage and testing
if __name__ == "__main__":
    print("=== Convex Hull Example ===")

    # Example 1: Simple 2x2 cost table
    cost_table = np.array([
        [1.0, 3.0],
        [2.0, 1.5]
    ])
    q_values = np.array([0.5, 1.0])

    print("Cost table:")
    print(cost_table)
    print("Q values:", q_values)

    hull_result = convex_hull_from_cost_table(cost_table, q_values)

    print(f"\nGenerated {len(hull_result.lines)} lines:")
    for line in hull_result.lines:
        print(f"  l_{line.cell_i}{line.cell_j} = {line.slope:.2f}*k + {line.intercept:.2f}")

    print(f"\nConvex hull has {len(hull_result.hull_lines)} lines:")
    for line in hull_result.hull_lines:
        print(f"  l_{line.cell_i}{line.cell_j} = {line.slope:.2f}*k + {line.intercept:.2f}")

    # Evaluate at specific k values
    test_k_values = [0.0, 0.5, 1.0]
    print(f"\nEvaluating hull:")
    for k in test_k_values:
        value, active_line = evaluate_hull_at_k(hull_result, k)
        print(f"  k={k}: value={value:.3f}, active_line=l_{active_line.cell_i}{active_line.cell_j}")
