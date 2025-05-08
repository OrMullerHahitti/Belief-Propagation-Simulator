import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError  # Import specific error
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
import math  # For checking isnan/isinf

# Global constant for numerical stability
EPSILON = 1e-9


@dataclass(frozen=True, eq=True)
class Line:
    """Represents a line y = slope * k + intercept."""

    slope: float
    intercept: float
    source_j: int  # Original source index (row in cost_table)
    target_i: int  # Original target index (column in cost_table)

    def evaluate(self, k: float) -> float:
        """Evaluates the line equation at a given k."""
        # Handle potential NaN/Inf from calculations if needed, though inputs should be valid
        if math.isnan(k) or math.isinf(k):
            return float("nan")
        val = self.slope * k + self.intercept
        if math.isnan(val) or math.isinf(val):
            # Fallback or error based on how invalid inputs should be treated
            # For now, return NaN if result is invalid
            return float("nan")
        return val


@dataclass
class EnvelopeSegment:
    """Represents a segment of the piecewise envelope."""

    line: Line
    k_start: float
    k_end: float


@dataclass
class IntersectionPoint:
    """Represents an intersection point between lines in an envelope."""

    k: float
    type: Literal["weak", "strong"]
    line1_indices: Optional[Tuple[int, int]] = None  # (source_j, target_i)
    line2_indices: Optional[Tuple[int, int]] = None  # (source_j, target_i)


def _find_intersection_k(line1: Line, line2: Line) -> Optional[float]:
    """Calculates the k-value where two lines intersect."""
    delta_slope = line1.slope - line2.slope
    if abs(delta_slope) < EPSILON:
        return None
    delta_intercept = line2.intercept - line1.intercept
    k = delta_intercept / delta_slope
    # Check if k is finite (avoid inf/nan if slopes are extremely close but not identical)
    if not math.isfinite(k):
        return None
    return k


def _are_points_collinear(points: np.ndarray) -> bool:
    """Checks if 3 or more 2D points are approximately collinear."""
    n_points = points.shape[0]
    if n_points < 3:
        return False  # Collinearity requires at least 3 points

    # Sort points by x-coordinate to check slopes between adjacent points
    try:
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
    except IndexError:
        print("Warning: IndexError during sorting points for collinearity check.")
        return False  # Cannot determine

    initial_slope = None
    for i in range(n_points - 1):
        p1 = sorted_points[i]
        p2 = sorted_points[i + 1]
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]

        current_slope = None
        if abs(delta_x) < EPSILON:
            # Vertical line segment
            if abs(delta_y) < EPSILON:  # Points identical, skip
                continue
            current_slope = float("inf")
        else:
            current_slope = delta_y / delta_x

        if i == 0 or initial_slope is None:  # Find first valid slope
            # If first segment was identical points, keep searching for initial slope
            if current_slope is not None:
                initial_slope = current_slope
        elif current_slope is not None:
            # Compare with initial slope
            if initial_slope == float("inf"):
                if current_slope != float("inf"):
                    return False  # Not collinear
            elif abs(current_slope - initial_slope) > EPSILON * max(
                1.0, abs(initial_slope)
            ):
                return False  # Slopes differ significantly

    # If loop completes and we found a consistent slope (or all points were identical)
    return (
        True if initial_slope is not None else False
    )  # Need at least one non-identical pair


def _compute_single_envelope(
    lines: List[Line], op: Literal["min", "max"], k_min: float, k_max: float
) -> Tuple[List[EnvelopeSegment], List[IntersectionPoint]]:
    """
    Computes the lower ('min') or upper ('max') envelope for a set of lines
    over the interval [k_min, k_max]. Handles cases with 1, 2, or >=3 effective lines,
    including collinear points for the >=3 case.
    """
    if not lines:
        return [], []

    if len(lines) == 1:
        return [EnvelopeSegment(lines[0], k_min, k_max)], []

    # --- Filter lines with duplicate slopes ---
    unique_slopes = {}
    for i, line in enumerate(lines):
        slope = line.slope
        intercept = line.intercept
        # Handle non-finite slopes/intercepts if necessary (e.g., skip or error)
        if not (math.isfinite(slope) and math.isfinite(intercept)):
            print(f"Warning: Skipping line with non-finite slope/intercept: {line}")
            continue

        rounded_slope = round(slope, 9)  # Round slope for robust duplicate check
        if rounded_slope not in unique_slopes:
            unique_slopes[rounded_slope] = i
        else:
            existing_idx = unique_slopes[rounded_slope]
            existing_intercept = lines[existing_idx].intercept
            if op == "min":
                if intercept < existing_intercept - EPSILON:
                    unique_slopes[rounded_slope] = i
            else:  # op == 'max'
                if intercept > existing_intercept + EPSILON:
                    unique_slopes[rounded_slope] = i

    filtered_lines = [lines[idx] for idx in unique_slopes.values()]
    num_effective_lines = len(filtered_lines)

    if num_effective_lines == 0:
        return [], []
    if num_effective_lines == 1:
        return [EnvelopeSegment(filtered_lines[0], k_min, k_max)], []

    # --- Prepare common calculation for 2 lines or collinear >=3 lines ---
    def build_envelope_from_two_lines(lineA, lineB, op, k_min, k_max):
        envelope_segments = []
        intersection_points = []
        k_intersect = _find_intersection_k(lineA, lineB)

        if k_intersect is None:
            # Parallel lines, pick the dominant one over the whole interval
            valA_mid = lineA.evaluate((k_min + k_max) / 2.0)
            valB_mid = lineB.evaluate((k_min + k_max) / 2.0)
            if op == "min":
                dominant_line = lineA if valA_mid <= valB_mid else lineB
            else:  # max
                dominant_line = lineA if valA_mid >= valB_mid else lineB
            envelope_segments.append(EnvelopeSegment(dominant_line, k_min, k_max))
        else:
            # Sort by slope for consistent handling
            if lineA.slope > lineB.slope:
                lineA, lineB = lineB, lineA  # lineA now has smaller slope

            line_lower_slope = lineA
            line_higher_slope = lineB

            if op == "min":
                line_dominant_first = line_lower_slope
                line_dominant_second = line_higher_slope
            else:  # op == 'max'
                line_dominant_first = line_higher_slope
                line_dominant_second = line_lower_slope

            if k_intersect <= k_min + EPSILON:
                envelope_segments.append(
                    EnvelopeSegment(line_dominant_second, k_min, k_max)
                )
            elif k_intersect >= k_max - EPSILON:
                envelope_segments.append(
                    EnvelopeSegment(line_dominant_first, k_min, k_max)
                )
            else:
                envelope_segments.append(
                    EnvelopeSegment(line_dominant_first, k_min, k_intersect)
                )
                envelope_segments.append(
                    EnvelopeSegment(line_dominant_second, k_intersect, k_max)
                )
                intersection_points.append(
                    IntersectionPoint(
                        k=k_intersect,
                        type="weak",
                        line1_indices=(lineA.source_j, lineA.target_i),
                        line2_indices=(lineB.source_j, lineB.target_i),
                    )
                )
        return envelope_segments, intersection_points

    # --- Handle Case: 2 Effective Lines ---
    if num_effective_lines == 2:
        return build_envelope_from_two_lines(
            filtered_lines[0], filtered_lines[1], op, k_min, k_max
        )

    # --- Case: >= 3 Effective Lines (Use Convex Hull) ---
    assert num_effective_lines >= 3

    points = np.array(
        [
            [line.slope, line.intercept if op == "min" else -line.intercept]
            for line in filtered_lines
        ]
    )
    lines_for_hull = (
        filtered_lines  # Keep track of which line corresponds to which point
    )

    try:
        hull = ConvexHull(
            points, qhull_options="Qt"
        )  # Add 'Qt' for triangulation, sometimes helps precision
    except QhullError as e:
        # Check if the error is due to precision/flatness (collinear points)
        if (
            "QH6154" in str(e)
            or "QH6214" in str(e)
            or "precision error" in str(e).lower()
        ):
            print(
                f"Warning: ConvexHull failed ({e}). Checking for collinearity among {num_effective_lines} points."
            )
            if _are_points_collinear(points):
                print("-> Points are collinear. Building envelope from endpoints.")
                # Sort the original filtered_lines by slope
                sorted_collinear_lines = sorted(
                    lines_for_hull, key=lambda line: line.slope
                )
                line_min_slope = sorted_collinear_lines[0]
                line_max_slope = sorted_collinear_lines[-1]
                # Build envelope using the two extreme lines
                segments, weak_points = build_envelope_from_two_lines(
                    line_min_slope, line_max_slope, op, k_min, k_max
                )
                # Clip segments and filter points (redundant if build_envelope handles it, but safe)
                final_segments = []
                for seg in segments:
                    clip_start = max(k_min, seg.k_start)
                    clip_end = min(k_max, seg.k_end)
                    if clip_end > clip_start + EPSILON:
                        final_segments.append(
                            EnvelopeSegment(seg.line, clip_start, clip_end)
                        )

                final_intersection_points = [
                    p for p in weak_points if k_min + EPSILON < p.k < k_max - EPSILON
                ]
                return final_segments, final_intersection_points
            else:
                # Error occurred, but points were not detected as collinear. Re-raise.
                print(
                    f"Error: ConvexHull precision error ({e}), but points not detected as collinear."
                )
                raise e
        else:
            # Different Qhull error, re-raise
            print(f"Error: Unhandled QhullError during ConvexHull: {e}")
            raise e
    except Exception as e:
        print(f"Error: Unexpected error during ConvexHull: {e}")
        raise e  # Re-raise other unexpected errors

    # --- Convex Hull Succeeded: Proceed with Vertex Extraction ---
    hull_indices = hull.vertices  # Indices relative to the 'points' array
    hull_lines_unsorted = [lines_for_hull[i] for i in hull_indices]
    hull_points_unsorted = points[hull_indices]

    sort_order = np.argsort(hull_points_unsorted[:, 0])
    hull_lines_sorted = [hull_lines_unsorted[i] for i in sort_order]

    if op == "max":
        hull_lines_final = []
        for line in hull_lines_sorted:
            # Make sure to create a NEW Line object with the original indices
            hull_lines_final.append(
                Line(line.slope, -line.intercept, line.source_j, line.target_i)
            )
        hull_lines_sorted = hull_lines_final

    envelope_segments: List[EnvelopeSegment] = []
    intersection_points: List[IntersectionPoint] = []
    last_k = k_min

    if not hull_lines_sorted:
        print(
            "Warning: Hull lines sorted list is empty after successful hull computation."
        )
        # Fallback might be needed here similar to other error cases
        return [], []  # Or implement fallback

    for i in range(len(hull_lines_sorted) - 1):
        line1 = hull_lines_sorted[i]
        line2 = hull_lines_sorted[i + 1]
        active_line_for_interval = line1
        intersect_k = _find_intersection_k(line1, line2)

        if intersect_k is not None:
            # Ensure intersection is strictly between last_k and k_max for segment generation
            # Clip intersection point to be within [k_min, k_max] before using it
            # intersect_k_clipped = max(k_min, min(k_max, intersect_k)) # Alternative clipping approach

            if intersect_k > last_k + EPSILON and intersect_k < k_max - EPSILON:
                # Segment ends at intersection
                segment = EnvelopeSegment(active_line_for_interval, last_k, intersect_k)
                envelope_segments.append(segment)
                intersection_points.append(
                    IntersectionPoint(
                        k=intersect_k,
                        type="weak",
                        line1_indices=(line1.source_j, line1.target_i),
                        line2_indices=(line2.source_j, line2.target_i),
                    )
                )
                last_k = intersect_k
            # elif intersect_k <= last_k + EPSILON:
            # Intersection is in the past or too close, line1 is superseded immediately
            # last_k remains the same, the next iteration will use line2 (via active_line = line1 logic)
            # else: # intersect_k >= k_max - EPSILON
            # Intersection is outside or at the end, line1 dominates until k_max
            # Loop will terminate or continue, last_k determines start of final segment

    # Add the final segment extending to k_max
    if last_k < k_max - EPSILON and hull_lines_sorted:
        final_line = hull_lines_sorted[-1]
        envelope_segments.append(EnvelopeSegment(final_line, last_k, k_max))
    elif (
        not envelope_segments and hull_lines_sorted
    ):  # Only one line formed the hull relevant to the range
        envelope_segments.append(EnvelopeSegment(hull_lines_sorted[0], k_min, k_max))

    # --- Final Clipping and Filtering ---
    final_segments = []
    if not envelope_segments and filtered_lines:
        # Fallback if hull processing yielded nothing (e.g., all intersections outside range)
        print(
            "Warning: Hull processing yielded no segments, using fallback evaluation."
        )
        k_mid = (k_min + k_max) / 2.0
        best_val = float("inf") if op == "min" else float("-inf")
        chosen_line = filtered_lines[0]  # Use filtered lines
        for line in filtered_lines:
            val = line.evaluate(k_mid)
            if op == "min":
                if val < best_val:
                    best_val = val
                    chosen_line = line
            else:  # max
                if val > best_val:
                    best_val = val
                    chosen_line = line
        final_segments = [EnvelopeSegment(chosen_line, k_min, k_max)]
    else:
        # Clip segments generated
        for seg in envelope_segments:
            clip_start = max(k_min, seg.k_start)
            clip_end = min(k_max, seg.k_end)
            if clip_end > clip_start + EPSILON:
                final_segments.append(EnvelopeSegment(seg.line, clip_start, clip_end))

    # Ensure full range coverage by segments (addressing potential small gaps)
    if final_segments:
        if final_segments[0].k_start > k_min + EPSILON:
            final_segments[0] = EnvelopeSegment(
                final_segments[0].line, k_min, final_segments[0].k_end
            )
        if final_segments[-1].k_end < k_max - EPSILON:
            final_segments[-1] = EnvelopeSegment(
                final_segments[-1].line, final_segments[-1].k_start, k_max
            )
    elif filtered_lines:  # If still no segments, use the fallback again
        print("Warning: No segments after clipping, using fallback evaluation.")
        k_mid = (k_min + k_max) / 2.0
        # ... (same fallback logic as above) ...
        # Find best line among filtered_lines at k_mid
        best_val = float("inf") if op == "min" else float("-inf")
        chosen_line = filtered_lines[0]
        for line in filtered_lines:
            val = line.evaluate(k_mid)
            if op == "min":
                if val < best_val - EPSILON:
                    best_val = val
                    chosen_line = line
                elif abs(val - best_val) < EPSILON:  # Tie break
                    if line.slope < chosen_line.slope:  # Prefer lower slope for min tie
                        chosen_line = line
            else:  # max
                if val > best_val + EPSILON:
                    best_val = val
                    chosen_line = line
                elif abs(val - best_val) < EPSILON:  # Tie break
                    if (
                        line.slope > chosen_line.slope
                    ):  # Prefer higher slope for max tie
                        chosen_line = line
        final_segments = [EnvelopeSegment(chosen_line, k_min, k_max)]

    final_intersection_points = [
        p for p in intersection_points if k_min + EPSILON < p.k < k_max - EPSILON
    ]

    return final_segments, final_intersection_points


# --- Main Function (compute_piecewise_envelopes) remains largely the same ---
# ... (Keep the previous version of compute_piecewise_envelopes) ...
def compute_piecewise_envelopes(
    cost_table: np.ndarray,
    q: np.ndarray,
    op: Literal["min", "max"] = "min",
    k_min: float = 0.0,
    k_max: float = 1.0,
) -> Tuple[List[List[EnvelopeSegment]], List[EnvelopeSegment], List[IntersectionPoint]]:
    """
    Computes individual and final piecewise envelopes for the cost function:
    cost(j -> i, k) = k * C[j, i] + q[j]

    Args:
        cost_table (np.ndarray): Shape (d, d). C[j, i] is the slope for j -> i.
        q (np.ndarray): Shape (d,). q[j] is the intercept offset for source j.
        op (Literal['min', 'max']): Whether to compute the lower ('min') or upper ('max') envelope.
        k_min (float): The minimum value of k for the interval.
        k_max (float): The maximum value of k for the interval.

    Returns:
        Tuple containing:
        - individual_envelopes: List[List[EnvelopeSegment]]. Outer list has length d (for each target i).
                                Inner list contains segments for E_i(k).
        - final_envelope: List[EnvelopeSegment]. Segments for E_final(k) = op_i { E_i(k) }.
        - intersection_points: List[IntersectionPoint]. All intersection points found,
                               marked as 'weak' or 'strong'.
    """
    if not isinstance(cost_table, np.ndarray) or cost_table.ndim != 2:
        raise ValueError("cost_table must be a 2D numpy array.")
    if not isinstance(q, np.ndarray) or q.ndim != 1:
        raise ValueError("q must be a 1D numpy array.")
    if cost_table.shape[0] != cost_table.shape[1] or cost_table.shape[0] != q.shape[0]:
        raise ValueError(
            "Shapes mismatch: cost_table must be (d, d) and q must be (d,)."
        )
    if k_min >= k_max:
        raise ValueError("k_min must be strictly less than k_max.")

    d = cost_table.shape[0]
    individual_envelopes: List[List[EnvelopeSegment]] = [[] for _ in range(d)]
    all_intersection_points: List[IntersectionPoint] = []

    # --- Step 1: Compute Individual Envelopes E_i ---
    print(f"--- Computing {d} individual envelopes ({op}) ---")
    all_final_segments_for_sweep = []  # Store segments from all E_i for the final step
    all_breakpoints = set([k_min, k_max])  # Collect all unique breakpoints

    for i in range(d):  # Target state i
        lines_for_i: List[Line] = []
        for j in range(d):  # Source state j
            slope = cost_table[j, i]
            intercept = q[j]
            lines_for_i.append(
                Line(slope=slope, intercept=intercept, source_j=j, target_i=i)
            )

        # print(f"  Target i={i}: Input lines:")
        # for line in lines_for_i: print(f"    {line}")

        segments, weak_points = _compute_single_envelope(lines_for_i, op, k_min, k_max)
        if not segments:
            print(
                f"Warning: No segments returned for individual envelope E_{i}. This might indicate an issue."
            )
            # If no segments, create a dummy line with NaN value? Or ensure fallback works.
            # For now, just skip adding breakpoints if empty
            continue

        individual_envelopes[i] = segments
        all_intersection_points.extend(weak_points)  # These are inherently weak

        # print(f"  Target i={i}: Envelope segments:")
        # for seg in segments: print(f"    {seg}")
        # print(f"  Target i={i}: Weak points:")
        # for p in weak_points: print(f"    {p}")

        # Collect segments and breakpoints for the final sweep
        all_final_segments_for_sweep.extend(segments)
        for seg in segments:
            if k_min + EPSILON < seg.k_start < k_max - EPSILON:
                all_breakpoints.add(seg.k_start)
            if k_min + EPSILON < seg.k_end < k_max - EPSILON:
                all_breakpoints.add(seg.k_end)
        for p in weak_points:
            all_breakpoints.add(p.k)

    # --- Step 2: Compute Final Envelope E_final using Plane Sweep ---
    print(f"\n--- Computing final envelope ({op}) ---")
    final_envelope: List[EnvelopeSegment] = []
    sorted_breakpoints = sorted(
        list(b for b in all_breakpoints if math.isfinite(b))
    )  # Ensure finite

    # Filter out breakpoints too close together
    filtered_breakpoints = []
    if sorted_breakpoints:
        filtered_breakpoints.append(sorted_breakpoints[0])
        for k in sorted_breakpoints[1:]:
            if k > filtered_breakpoints[-1] + EPSILON:
                filtered_breakpoints.append(k)

    # Ensure k_min and k_max are the absolute boundaries
    if not filtered_breakpoints or filtered_breakpoints[0] > k_min + EPSILON:
        filtered_breakpoints.insert(0, k_min)
    if not filtered_breakpoints or filtered_breakpoints[-1] < k_max - EPSILON:
        filtered_breakpoints.append(k_max)
    # Refilter after adding boundaries
    temp_filtered = []
    if filtered_breakpoints:
        temp_filtered.append(filtered_breakpoints[0])
        for k in filtered_breakpoints[1:]:
            if k > temp_filtered[-1] + EPSILON:
                temp_filtered.append(k)
    filtered_breakpoints = temp_filtered

    last_winning_line_info: Optional[Tuple[int, int]] = None
    last_winning_target_i: Optional[int] = None

    for idx in range(len(filtered_breakpoints) - 1):
        k_start = filtered_breakpoints[idx]
        k_end = filtered_breakpoints[idx + 1]

        if k_end <= k_start + EPSILON:
            continue

        k_mid = (k_start + k_end) / 2.0
        # Ensure k_mid is within bounds strictly
        if k_mid < k_min:
            k_mid = k_min + EPSILON * (k_max - k_min)
        if k_mid > k_max:
            k_mid = k_max - EPSILON * (k_max - k_min)
        # If interval is tiny, k_mid might equal k_start or k_end, adjust slightly
        if abs(k_mid - k_start) < EPSILON:
            k_mid = k_start + EPSILON * (k_end - k_start)
        if abs(k_mid - k_end) < EPSILON:
            k_mid = k_end - EPSILON * (k_end - k_start)
        # Clamp again after adjustment
        k_mid = max(k_min, min(k_max, k_mid))

        best_val = float("inf") if op == "min" else float("-inf")
        winning_segment: Optional[EnvelopeSegment] = None
        current_winning_target_i_for_interval: Optional[int] = None

        for i in range(d):
            if not individual_envelopes[i]:
                continue

            active_segment_for_i: Optional[EnvelopeSegment] = None
            for seg in individual_envelopes[i]:
                # Be inclusive at start, exclusive at end for check? No, check if k_mid is within [start, end]
                if seg.k_start <= k_mid + EPSILON and seg.k_end >= k_mid - EPSILON:
                    active_segment_for_i = seg
                    break

            if active_segment_for_i:
                current_val = active_segment_for_i.line.evaluate(k_mid)
                # Skip if evaluation failed
                if math.isnan(current_val):
                    continue

                is_better = (op == "min" and current_val < best_val - EPSILON) or (
                    op == "max" and current_val > best_val + EPSILON
                )
                is_effectively_equal = abs(current_val - best_val) < EPSILON

                if is_better:
                    best_val = current_val
                    winning_segment = active_segment_for_i
                    current_winning_target_i_for_interval = (
                        active_segment_for_i.line.target_i
                    )
                elif is_effectively_equal:
                    if winning_segment is None:
                        winning_segment = active_segment_for_i
                        current_winning_target_i_for_interval = (
                            active_segment_for_i.line.target_i
                        )
                    else:
                        # Tie-breaking: prefer lower source_j, then lower target_i
                        current_line = active_segment_for_i.line
                        existing_line = winning_segment.line
                        if current_line.source_j < existing_line.source_j:
                            winning_segment = active_segment_for_i
                            current_winning_target_i_for_interval = (
                                active_segment_for_i.line.target_i
                            )
                        elif (
                            current_line.source_j == existing_line.source_j
                            and current_line.target_i < existing_line.target_i
                        ):
                            winning_segment = active_segment_for_i
                            current_winning_target_i_for_interval = (
                                active_segment_for_i.line.target_i
                            )

        if winning_segment:
            current_winning_line_info = (
                winning_segment.line.source_j,
                winning_segment.line.target_i,
            )

            if (
                last_winning_target_i is not None
                and current_winning_target_i_for_interval is not None
                and current_winning_target_i_for_interval != last_winning_target_i
                and k_start > k_min + EPSILON
            ):
                point_found = False
                for p in all_intersection_points:
                    if abs(p.k - k_start) < EPSILON:
                        p.type = "strong"
                        point_found = True
                        break
                if not point_found:
                    all_intersection_points.append(
                        IntersectionPoint(k=k_start, type="strong")
                    )

            # Merge or add segment
            if (
                final_envelope
                and final_envelope[-1].line == winning_segment.line
                and abs(final_envelope[-1].k_end - k_start) < EPSILON
            ):
                final_envelope[-1] = EnvelopeSegment(
                    final_envelope[-1].line, final_envelope[-1].k_start, k_end
                )
            else:
                final_envelope.append(
                    EnvelopeSegment(
                        winning_segment.line, k_start, k_end  # Start at the breakpoint
                    )
                )

            last_winning_line_info = current_winning_line_info
            last_winning_target_i = current_winning_target_i_for_interval
        else:
            # Handle case where no valid segment was found in any E_i for this interval
            if final_envelope:  # Attempt to extend the previous segment if possible
                print(
                    f"Warning: No winning segment found for interval [{k_start:.4f}, {k_end:.4f}]. Extending previous segment if possible."
                )
                # Check if previous segment's line is valid at k_mid
                prev_seg = final_envelope[-1]
                prev_val = prev_seg.line.evaluate(k_mid)
                if math.isfinite(
                    prev_val
                ):  # If previous line is valid, tentatively extend it
                    final_envelope[-1] = EnvelopeSegment(
                        prev_seg.line, prev_seg.k_start, k_end
                    )
                    # Keep last_winning info the same
                else:  # Cannot extend, discontinuity or error
                    print(
                        f"Error: Cannot extend previous segment in gap interval [{k_start:.4f}, {k_end:.4f}]."
                    )
                    last_winning_line_info = None
                    last_winning_target_i = None

            else:  # No previous segment to extend, serious issue
                print(
                    f"Error: No winning segment found for initial interval [{k_start:.4f}, {k_end:.4f}]. Cannot build final envelope."
                )
                # Possibly return empty or raise error depending on desired behavior
                return (
                    individual_envelopes,
                    [],
                    all_intersection_points,
                )  # Return partial results
            # Reset tracking if gap occurs?
            # last_winning_line_info = None
            # last_winning_target_i = None

    # --- Final Cleanup of Intersection Points ---
    all_intersection_points.sort(key=lambda p: p.k)
    unique_points = []
    processed_ks = set()  # Keep track of k values already added

    for p in all_intersection_points:
        # Skip points outside the open interval (k_min, k_max)
        if p.k <= k_min + EPSILON or p.k >= k_max - EPSILON:
            continue

        # Check if a point with this k (within tolerance) is already added
        found_nearby = False
        for uk in processed_ks:
            if abs(p.k - uk) < EPSILON:
                found_nearby = True
                # If new point is strong, upgrade the existing one
                for up in unique_points:
                    if abs(p.k - up.k) < EPSILON:
                        if p.type == "strong" and up.type == "weak":
                            up.type = "strong"
                        break  # Stop searching unique_points
                break  # Stop searching processed_ks

        if not found_nearby:
            unique_points.append(p)
            processed_ks.add(p.k)

    # Ensure final envelope covers k_min to k_max if possible
    if final_envelope:
        if final_envelope[0].k_start > k_min + EPSILON:
            final_envelope[0] = EnvelopeSegment(
                final_envelope[0].line, k_min, final_envelope[0].k_end
            )
        if final_envelope[-1].k_end < k_max - EPSILON:
            final_envelope[-1] = EnvelopeSegment(
                final_envelope[-1].line, final_envelope[-1].k_start, k_max
            )
    elif individual_envelopes:  # If final failed but individual exist, try fallback
        print("Warning: Final envelope is empty, attempting final fallback.")
        # Perform fallback evaluation using individual envelopes at midpoint
        k_mid = (k_min + k_max) / 2.0
        best_val = float("inf") if op == "min" else float("-inf")
        winning_line: Optional[Line] = None
        # ... (logic similar to single envelope fallback) ...
        # Find the best line across all individual envelopes at k_mid
        for i in range(d):
            if not individual_envelopes[i]:
                continue
            active_segment_for_i: Optional[EnvelopeSegment] = None
            for seg in individual_envelopes[i]:
                if seg.k_start <= k_mid + EPSILON and seg.k_end >= k_mid - EPSILON:
                    active_segment_for_i = seg
                    break
            if active_segment_for_i:
                current_val = active_segment_for_i.line.evaluate(k_mid)
                if math.isnan(current_val):
                    continue
                # ... (comparison and tie-breaking logic as in sweep) ...
                is_better = (op == "min" and current_val < best_val - EPSILON) or (
                    op == "max" and current_val > best_val + EPSILON
                )
                is_effectively_equal = abs(current_val - best_val) < EPSILON

                if is_better:
                    best_val = current_val
                    winning_line = active_segment_for_i.line
                elif is_effectively_equal:
                    if winning_line is None:
                        winning_line = active_segment_for_i.line
                    else:
                        # Tie-breaking
                        current_line = active_segment_for_i.line
                        existing_line = winning_line
                        if current_line.source_j < existing_line.source_j:
                            winning_line = current_line
                        elif (
                            current_line.source_j == existing_line.source_j
                            and current_line.target_i < existing_line.target_i
                        ):
                            winning_line = current_line

        if winning_line:
            final_envelope = [EnvelopeSegment(winning_line, k_min, k_max)]

    print("\n--- Final Results ---")
    return individual_envelopes, final_envelope, unique_points


# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    print("--- Example 1 (d=2, min, k in [0, 1]) ---")
    # C = [[C00, C01], [C10, C11]] -> C[j, i]
    cost_table_ex1 = np.array(
        [
            [1.0, 5.0],  # j=0 -> C[0,0]=1, C[0,1]=5
            [3.0, 2.0],  # j=1 -> C[1,0]=3, C[1,1]=2
        ]
    )
    q_ex1 = np.array([4.0, 1.0])  # q[0]=4, q[1]=1

    ind_env1, final_env1, points1 = compute_piecewise_envelopes(
        cost_table_ex1, q_ex1, op="min", k_min=0.0, k_max=1.0
    )

    print("\n--- Detailed Results Example 1 ---")
    print("Individual Envelopes:")
    for i, env in enumerate(ind_env1):
        print(f" E_{i}:")
        for seg in env:
            print(f"  {seg}")
    print("\nFinal Envelope:")
    for seg in final_env1:
        print(f"  {seg}")
    print("\nIntersection Points:")
    if not points1:
        print("  (None)")
    for p in points1:
        print(f"  {p}")

    print("\n\n--- Example 2 (d=3, max, k in [0, 1]) ---")
    cost_table_ex2 = np.array(
        [[2.0, 1.0, 0.5], [0.0, 3.0, 1.0], [1.0, -1.0, 4.0]]  # j=0  # j=1  # j=2
    )
    q_ex2 = np.array([0.0, 1.0, -1.0])  # q[0]=0, q[1]=1, q[2]=-1

    ind_env2, final_env2, points2 = compute_piecewise_envelopes(
        cost_table_ex2, q_ex2, op="max", k_min=0.0, k_max=1.0
    )

    print("\n--- Detailed Results Example 2 ---")
    print("Individual Envelopes:")
    for i, env in enumerate(ind_env2):
        print(f" E_{i}:")
        for seg in env:
            print(f"  {seg}")
    print("\nFinal Envelope:")
    for seg in final_env2:
        print(f"  {seg}")
    print("\nIntersection Points:")
    if not points2:
        print("  (None)")
    for p in points2:
        print(f"  {p}")

    print("\n\n--- Example 3 (d=2, min, k in [0, 5] - potential intersections) ---")
    cost_table_ex3 = np.array([[1.0, 5.0], [3.0, 2.0]])
    q_ex3 = np.array([4.0, 1.0])
    ind_env3, final_env3, points3 = compute_piecewise_envelopes(
        cost_table_ex3, q_ex3, op="min", k_min=0.0, k_max=5.0  # Extended k range
    )
    print("\n--- Detailed Results Example 3 ---")
    print("Individual Envelopes:")
    for i, env in enumerate(ind_env3):
        print(f" E_{i}:")
        for seg in env:
            print(f"  {seg}")
    print("\nFinal Envelope:")
    for seg in final_env3:
        print(f"  {seg}")
    print("\nIntersection Points:")
    if not points3:
        print("  (None)")
    for p in points3:
        print(f"  {p}")

    print("\n\n--- Example 4 (Collinear/Parallel Lines Test, d=3, min) ---")
    cost_table_ex4 = np.array(
        [
            [2.0, 1.0, 3.0],  # j=0
            [2.0, 3.0, 3.0],  # j=1 (duplicate slope for i=0, i=2)
            [1.0, -1.0, 3.0],  # j=2 (duplicate slope for i=2)
        ]
    )
    q_ex4 = np.array([1.0, 2.0, 0.0])  # q[0]=1, q[1]=2, q[2]=0

    ind_env4, final_env4, points4 = compute_piecewise_envelopes(
        cost_table_ex4, q_ex4, op="min", k_min=0.0, k_max=1.0
    )

    print("\n--- Detailed Results Example 4 ---")
    print("Individual Envelopes:")
    for i, env in enumerate(ind_env4):
        print(f" E_{i}:")
        for seg in env:
            print(f"  {seg}")
    print("\nFinal Envelope:")
    for seg in final_env4:
        print(f"  {seg}")
    print("\nIntersection Points:")
    if not points4:
        print("  (None)")
    for p in points4:
        print(f"  {p}")
