import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Literal
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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


@dataclass(frozen=True, eq=True)
class InterceptPoint:
    """Represents an intercept point between lines or envelopes."""

    k: float                                                    # K value where intersection occurs
    intersection_value: float                                   # Value at intersection point
    type: Literal["change_assignment", "partial_assignment_change"]  # Type of intercept
    envelope1_id: int                                          # ID of first envelope/line
    envelope2_id: int                                          # ID of second envelope/line
    line1: Optional[ConvexLine] = None                         # First line (if partial change)
    line2: Optional[ConvexLine] = None                         # Second line (if partial change)


@dataclass
class ConvexHullResult:
    """Result of convex hull computation."""

    lines: List[ConvexLine]              # All generated lines
    hull_lines: List[ConvexLine]         # Lines on the convex hull
    hull_vertices: np.ndarray            # Hull vertices as (slope, intercept) points
    k_range: Tuple[float, float]         # Range of k values used
    envelope_id: int = 0                 # Unique ID for this envelope


@dataclass
class HierarchicalEnvelopeResult:
    """Result of hierarchical envelope computation."""

    individual_envelopes: List[ConvexHullResult]     # Individual convex hulls
    meta_envelope: ConvexHullResult                  # Envelope of envelopes
    all_intercepts: List[InterceptPoint]             # All intercept points
    change_assignment_points: List[InterceptPoint]   # Intercepts between different envelopes
    partial_change_points: List[InterceptPoint]      # Intercepts within same envelope
    envelope_type: str                               # "lower" or "upper"


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


def hierarchical_convex_hull_from_agents(
    variable_agents: List[VariableAgent],
    factor_agents: List[FactorAgent],
    k_min: float = 0.0,
    k_max: float = 1.0
) -> HierarchicalEnvelopeResult:
    """
    Create hierarchical convex hull from multiple variable and factor agents.
    Automatically detects envelope type from agent computators.

    Args:
        variable_agents: List of variable agents
        factor_agents: List of factor agents
        k_min: Minimum k value (default 0.0)
        k_max: Maximum k value (default 1.0)

    Returns:
        HierarchicalEnvelopeResult with classified intercepts
    """
    if not variable_agents or not factor_agents:
        raise ValueError("Need at least one variable agent and one factor agent")

    # Determine envelope type from the first agent with a computator
    envelope_type = "lower"  # Default
    for agent in variable_agents + factor_agents:
        detected_type = determine_envelope_type(agent)
        if detected_type in ["lower", "upper"]:
            envelope_type = detected_type
            break

    # Create individual convex hulls for each variable-factor pair
    individual_hulls = []

    for var_agent in variable_agents:
        for factor_agent in factor_agents:
            # Check if they are connected
            if (factor_agent.connection_number and
                var_agent.name in factor_agent.connection_number):

                hull = convex_hull_from_agents(
                    var_agent, factor_agent, envelope_type, k_min, k_max
                )
                individual_hulls.append(hull)

    if not individual_hulls:
        # If no connections found, create hulls for all combinations
        for i, var_agent in enumerate(variable_agents):
            for j, factor_agent in enumerate(factor_agents):
                hull = convex_hull_from_agents(
                    var_agent, factor_agent, envelope_type, k_min, k_max
                )
                hull.envelope_id = i * len(factor_agents) + j
                individual_hulls.append(hull)

    # Compute hierarchical envelopes
    return compute_hierarchical_envelopes(individual_hulls, envelope_type, k_min, k_max)


def convex_hull_from_agents(
    variable_agent: VariableAgent,
    factor_agent: FactorAgent,
    hull_type: Optional[str] = None,
    k_min: float = 0.0,
    k_max: float = 1.0
) -> ConvexHullResult:
    """
    Create convex hull from variable and factor agents.

    Args:
        variable_agent: VariableAgent with domain information
        factor_agent: FactorAgent with cost table
        hull_type: "lower" or "upper" (if None, auto-detect from computator)
        k_min: Minimum k value (default 0.0)
        k_max: Maximum k value (default 1.0)

    Returns:
        ConvexHullResult containing hull information
    """
    if factor_agent.cost_table is None:
        raise ValueError("FactorAgent must have a cost table")

    # Auto-detect hull type if not specified
    if hull_type is None:
        hull_type = determine_envelope_type(variable_agent)
        # If variable agent doesn't have computator, try factor agent
        if hull_type == "lower":  # Default, try factor agent
            factor_type = determine_envelope_type(factor_agent)
            if factor_type != "lower":
                hull_type = factor_type

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


def find_line_intersection(line1: ConvexLine, line2: ConvexLine) -> Optional[Tuple[float, float]]:
    """
    Find intersection point between two lines.

    Args:
        line1: First line
        line2: Second line

    Returns:
        Tuple of (k, value) at intersection, or None if lines are parallel
    """
    slope_diff = line1.slope - line2.slope
    if abs(slope_diff) < EPSILON:
        return None  # Lines are parallel

    # Solve: line1.slope * k + line1.intercept = line2.slope * k + line2.intercept
    # (line1.slope - line2.slope) * k = line2.intercept - line1.intercept
    k_intersect = (line2.intercept - line1.intercept) / slope_diff

    if not math.isfinite(k_intersect):
        return None

    # Calculate value at intersection using either line (should be the same)
    value = line1.evaluate(k_intersect)
    return k_intersect, value


def find_all_envelope_intercepts(hull_result: ConvexHullResult) -> List[InterceptPoint]:
    """
    Find all intercepts within a single envelope between adjacent hull lines.

    Args:
        hull_result: Convex hull result to analyze

    Returns:
        List of intercept points marked as "partial_assignment_change"
    """
    if len(hull_result.hull_lines) <= 1:
        return []  # No intercepts possible with 0 or 1 line

    intercepts = []

    # Sort hull lines by slope to ensure proper ordering
    sorted_lines = sorted(hull_result.hull_lines, key=lambda x: x.slope)

    # Find intersections between adjacent lines
    for i in range(len(sorted_lines) - 1):
        line1 = sorted_lines[i]
        line2 = sorted_lines[i + 1]

        intersection = find_line_intersection(line1, line2)
        if intersection is not None:
            k_val, intersection_value = intersection

            # Only include intersections within the k range and that are meaningful
            k_min, k_max = hull_result.k_range
            if k_min <= k_val <= k_max:
                intercept = InterceptPoint(
                    k=k_val,
                    intersection_value=intersection_value,
                    type="partial_assignment_change",
                    envelope1_id=hull_result.envelope_id,
                    envelope2_id=hull_result.envelope_id,  # Same envelope
                    line1=line1,
                    line2=line2
                )
                intercepts.append(intercept)

    return intercepts


def determine_envelope_type(agent: Union[VariableAgent, FactorAgent]) -> str:
    """
    Determine envelope type based on agent's computator.

    Args:
        agent: Variable or Factor agent with computator

    Returns:
        "lower" for MinSum computators, "upper" for MaxSum computators
    """
    if not hasattr(agent, 'computator') or agent.computator is None:
        return "lower"  # Default to lower envelope

    # Check the reduce function used by the computator
    if hasattr(agent.computator, 'reduce_func'):
        reduce_func = agent.computator.reduce_func
        # Import numpy here to avoid circular imports
        import numpy as np

        if reduce_func == np.min:
            return "lower"
        elif reduce_func == np.max:
            return "upper"
        else:
            # For other reduce functions, default to lower
            return "lower"

    return "lower"  # Default fallback


def compute_hierarchical_envelopes(
    individual_hulls: List[ConvexHullResult],
    envelope_type: str = "lower",
    k_min: float = 0.0,
    k_max: float = 1.0
) -> HierarchicalEnvelopeResult:
    """
    Compute hierarchical envelopes: create envelope of envelopes and classify all intercepts.

    Args:
        individual_hulls: List of individual convex hull results
        envelope_type: "lower" or "upper" envelope type
        k_min: Minimum k value
        k_max: Maximum k value

    Returns:
        Hierarchical envelope result with classified intercepts
    """
    if not individual_hulls:
        raise ValueError("Cannot compute hierarchical envelopes from empty hull list")

    # Assign unique envelope IDs
    for i, hull in enumerate(individual_hulls):
        hull.envelope_id = i

    # Collect all partial assignment change points (within individual envelopes)
    partial_change_points = []
    for hull in individual_hulls:
        partial_intercepts = find_all_envelope_intercepts(hull)
        partial_change_points.extend(partial_intercepts)

    # Create meta-lines representing each envelope
    # For each envelope, we need to create a representative line or set of lines
    # Here we'll use a simplified approach: create lines from the hull lines of each envelope
    meta_lines = []
    envelope_line_mapping = {}  # Maps meta-line index to (envelope_id, original_line)

    for hull in individual_hulls:
        for line in hull.hull_lines:
            # Create a meta-line that represents this line in the hierarchical system
            meta_line = ConvexLine(
                slope=line.slope,
                intercept=line.intercept,
                cell_i=hull.envelope_id,  # Use envelope ID as cell_i
                cell_j=len(meta_lines)    # Unique identifier within this envelope
            )
            envelope_line_mapping[len(meta_lines)] = (hull.envelope_id, line)
            meta_lines.append(meta_line)

    # Compute convex hull of all meta-lines
    meta_hull = compute_convex_hull_from_lines(meta_lines, envelope_type)
    meta_hull.envelope_id = -1  # Special ID for meta-envelope

    # Find intercepts between different envelopes (change_assignment points)
    change_assignment_points = []

    # Check intersections between lines from different envelopes that are on the meta-hull
    meta_sorted_lines = sorted(meta_hull.hull_lines, key=lambda x: x.slope)

    for i in range(len(meta_sorted_lines) - 1):
        line1 = meta_sorted_lines[i]
        line2 = meta_sorted_lines[i + 1]

        # Get original envelope information
        env1_id = line1.cell_i
        env2_id = line2.cell_i

        # Only consider as change_assignment if lines are from different envelopes
        if env1_id != env2_id:
            intersection = find_line_intersection(line1, line2)
            if intersection is not None:
                k_val, intersection_value = intersection

                if k_min <= k_val <= k_max:
                    # Get original lines for reference
                    orig_line1 = envelope_line_mapping.get(line1.cell_j, (None, None))[1]
                    orig_line2 = envelope_line_mapping.get(line2.cell_j, (None, None))[1]

                    intercept = InterceptPoint(
                        k=k_val,
                        intersection_value=intersection_value,
                        type="change_assignment",
                        envelope1_id=env1_id,
                        envelope2_id=env2_id,
                        line1=orig_line1,
                        line2=orig_line2
                    )
                    change_assignment_points.append(intercept)

    # Combine all intercepts
    all_intercepts = partial_change_points + change_assignment_points

    # Sort all intercepts by k value
    all_intercepts.sort(key=lambda x: x.k)

    return HierarchicalEnvelopeResult(
        individual_envelopes=individual_hulls,
        meta_envelope=meta_hull,
        all_intercepts=all_intercepts,
        change_assignment_points=change_assignment_points,
        partial_change_points=partial_change_points,
        envelope_type=envelope_type
    )


def evaluate_hull_at_k(hull_result: ConvexHullResult, k: float) -> Tuple[float, ConvexLine]:
    """
    DEPRECATED: Use find_all_envelope_intercepts() and hierarchical envelopes instead.

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
    print("=== Hierarchical Convex Hull Example ===")

    # Example 1: Simple 2x2 cost table
    cost_table1 = np.array([
        [1.0, 3.0],
        [2.0, 1.5]
    ])
    q_values1 = np.array([0.5, 1.0])

    # Example 2: Another cost table
    cost_table2 = np.array([
        [0.5, 2.5],
        [3.0, 1.0]
    ])
    q_values2 = np.array([1.0, 0.5])

    print("Cost table 1:")
    print(cost_table1)
    print("Q values 1:", q_values1)

    print("\nCost table 2:")
    print(cost_table2)
    print("Q values 2:", q_values2)

    # Create individual hulls
    hull1 = convex_hull_from_cost_table(cost_table1, q_values1, hull_type="lower")
    hull2 = convex_hull_from_cost_table(cost_table2, q_values2, hull_type="lower")

    print(f"\nHull 1 - Generated {len(hull1.lines)} lines:")
    for line in hull1.lines:
        print(f"  l_{line.cell_i}{line.cell_j} = {line.slope:.2f}*k + {line.intercept:.2f}")

    print(f"\nHull 1 - Convex hull has {len(hull1.hull_lines)} lines:")
    for line in hull1.hull_lines:
        print(f"  l_{line.cell_i}{line.cell_j} = {line.slope:.2f}*k + {line.intercept:.2f}")

    # Find intercepts within hull 1
    intercepts1 = find_all_envelope_intercepts(hull1)
    print(f"\nHull 1 - Found {len(intercepts1)} partial assignment change points:")
    for intercept in intercepts1:
        print(f"  k={intercept.k:.3f}, value={intercept.intersection_value:.3f}, type={intercept.type}")

    # Create hierarchical envelope
    hierarchical_result = compute_hierarchical_envelopes([hull1, hull2], envelope_type="lower")

    print(f"\n=== Hierarchical Results ===")
    print(f"Total intercepts: {len(hierarchical_result.all_intercepts)}")
    print(f"Change assignment points: {len(hierarchical_result.change_assignment_points)}")
    print(f"Partial assignment change points: {len(hierarchical_result.partial_change_points)}")

    print(f"\nAll intercepts (sorted by k):")
    for intercept in hierarchical_result.all_intercepts:
        print(f"  k={intercept.k:.3f}, value={intercept.intersection_value:.3f}, type={intercept.type}")
        if intercept.type == "change_assignment":
            print(f"    Between envelopes {intercept.envelope1_id} and {intercept.envelope2_id}")
        else:
            print(f"    Within envelope {intercept.envelope1_id}")

    print(f"\nMeta-envelope has {len(hierarchical_result.meta_envelope.hull_lines)} lines on hull")
