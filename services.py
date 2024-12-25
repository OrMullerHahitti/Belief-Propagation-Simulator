from models import VariableNode, FactorNode, Q, R
import pandas as pd

class Line:
    """
    Represents L(k) = slope*k + intercept.
    slope = costTable[i][j]
    intercept = Q[i]
    """

    def __init__(self, slope: float, intercept: float, name: str = ""):
        self.slope = slope
        self.intercept = intercept
        self.name = name  # Optional: store something like f"Line i=..., j=..."

    def value_at(self, k: float) -> float:
        """Compute the line value at k."""
        return self.slope * k + self.intercept

    def intersect(self, other: "Line") -> float or None:
        """
        Find k where this line == other line.
        slope1*k + intercept1 = slope2*k + intercept2
        => k * (slope1 - slope2) = intercept2 - intercept1
        => k = (intercept2 - intercept1) / (slope1 - slope2)

        Returns None if lines are parallel or effectively parallel.
        """
        denom = (self.slope - other.slope)
        if abs(denom) < 1e-12:
            return None  # parallel or nearly parallel
        return (other.intercept - self.intercept) / denom

    def __repr__(self):
        return f"Line(name={self.name}, slope={self.slope:.3f}, intercept={self.intercept:.3f})"

class LinesGroup:
    """
    Holds multiple lines, e.g., from a single column of cost table.
    """
    def __init__(self, lines):
        self.lines = lines  # list of Line objects

    def build_curve(self) -> "Curve":
        """
        Build the piecewise-minimum envelope (Curve) from these lines.
        """
        # 1) Find all intersection points among lines.
        intersection_points = [0.0, 1.0]  # start with 0 and 1
        n = len(self.lines)

        for i in range(n):
            for j in range(i+1, n):
                k_int = self.lines[i].intersect(self.lines[j])
                if k_int is not None:
                    # Only consider intersections within [0,1]
                    if 0.0 <= k_int <= 1.0:
                        intersection_points.append(k_int)

        # 2) Sort unique intersection points
        intersection_points = sorted(set(intersection_points))

        # 3) For each sub-interval, figure out which line is minimal
        segments = []
        for idx in range(len(intersection_points) - 1):
            start_k = intersection_points[idx]
            end_k = intersection_points[idx + 1]
            mid_k = (start_k + end_k) / 2.0  # test the midpoint

            # Find which line is minimal at mid_k
            min_line = min(self.lines, key=lambda line: line.value_at(mid_k))

            segments.append(Segment(start_k, end_k, min_line))

        # Build the final Curve
        return Curve(segments)

    def __repr__(self):
        return f"LinesGroup with {len(self.lines)} lines"
class Segment:
    """One piece of the piecewise-minimum envelope."""
    def __init__(self, start_k: float, end_k: float, line: Line):
        self.start_k = start_k
        self.end_k = end_k
        self.line = line  # The line that is minimal on [start_k, end_k)

    def __repr__(self):
        return (f"Segment(k in [{self.start_k:.2f}, {self.end_k:.2f}), "
                f"line={self.line})")


class Curve:
    """A piecewise linear function that is the minimum of multiple lines."""
    def __init__(self, segments):
        self.segments = segments  # list of Segment objects

    def evaluate(self, k: float) -> float:
        """Evaluate the piecewise-minimum at a given k in [0,1]."""
        # If k == 1 exactly, we might want to handle carefully
        for seg in self.segments:
            if seg.start_k <= k < seg.end_k:
                return seg.line.value_at(k)
        # If k = 1.0 exactly, let's pick the last segment
        if abs(k - 1.0) < 1e-12:
            return self.segments[-1].line.value_at(k)
        raise ValueError(f"k={k} is outside [0,1] or no segment found.")

    def __repr__(self):
        text = "Curve:\n"
        for seg in self.segments:
            text += f"  {seg}\n"
        return text


class FactorService:
    """
    Builds lines from a given FactorNode (cost table) + Q
    Then constructs piecewise-minimum curves (one per column or row, etc.).
    """

    def __init__(self, factor_node:FactorNode, q:Q, use_columns=True):
        """
        :param factor_node: FactorNode that has a cost_table of size n x n
        :param q: Q object, domain_size = n
        :param use_columns: If True, interpret domain as columns.
                           Otherwise, we'd do rows.
        """
        self.factor_node = factor_node
        self.q = q
        self.use_columns = use_columns
        # We assume domain_size is the same as Q's domain_size
        self.domain_size = q.domain_size

    def build_lines_for_column(self, col_idx: int):
        """
        Builds a LinesGroup for a single column (col_idx).
        We'll produce n lines, one for each domain index i in [1..n].

        L_i(k) = k * cost_table[i][col_idx] + Q[i]
        """
        lines = []
        for i in range(1, self.q.domain_size+1 ):
            slope = self.factor_node.cost_table[i,col_idx]  # cost_table[i][j]
            intercept = self.q[i]  # Q[i]
            line_name = f"Line(i={i}, col={col_idx})"
            lines.append(Line(slope, intercept, line_name))
        return LinesGroup(lines)

    def build_lines_for_row(self, row_idx: int):
        """
        Similar, if we wanted row-based approach:
        L_j(k) = k * cost_table[row_idx][j] + Q[j]
        """
        lines = []
        for j in range(1, self.domain_size + 1):
            slope = self.factor_node.cost_table[row_idx][j]
            intercept = self.q[j]
            line_name = f"Line(row={row_idx}, j={j})"
            lines.append(Line(slope, intercept, line_name))
        return LinesGroup(lines)

    def build_all_curves(self):
        """
        For each column (if use_columns=True) or each row (otherwise),
        build the piecewise-minimum curve. Return a list of Curves.
        """
        curves = []
        if self.use_columns:
            # domain_size columns
            for col in range(1, self.domain_size):
                group = self.build_lines_for_column(col)
                curve = group.build_curve()
                curves.append(curve)
        else:
            # domain_size rows
            for row in range(1, self.domain_size):
                group = self.build_lines_for_row(row)
                curve = group.build_curve()
                curves.append(curve)
        return curves

# 1) Create VariableNodes, FactorNode, Q, etc. (just as an example)
X1 = VariableNode(name="X1", domain_size=3)
X2 = VariableNode(name="X2", domain_size=3)
F12 = FactorNode(name="F12", var_nodes=(X1, X2))
Q_X1_F12 = Q(domain_size=X1.domain_size)  # random Q of size 3

# 2) Instantiate FactorService
service = FactorService(factor_node=F12, q=Q_X1_F12, use_columns=True)

# 3) Build piecewise-minimum curves for each column
curves = service.build_all_curves()  # returns a list of Curve objects

# 4) Print them out
for idx, curve in enumerate(curves, start=1):
    print(f"=== Curve for Column {idx} ===")
    print(curve)
    # Evaluate the curve at a few points
    for test_k in [0.0, 0.3, 0.7, 1.0]:
        val = curve.evaluate(test_k)
        print(f"  k={test_k} => curve value = {val:.4f}")
    print()

#q_instance = Q(df)
#print(q_instance)
#q_instance[1] = 5
#q_instance['a'] = 7
#print(q_instance)


# #df = create_random_df()
# add_index_labels(df)
# Q = create_Q(df)
# print(df)
# print(Q)


# class recieved:
#     def __init__(self, ):
#         self.R = R
#         self.recieved_message = [min(R[i][1:]) for i in range(1, len(R))]
#         self.recieved_message_index = [R[i][1:].index(min(R[i][1:])) + 1 for i in range(1, len(R))]
#








# # Example usage
# print(C('a', 'b'))
# print(R('a'))
#
# k_min, k_max = 0, 1
# k_values = np.linspace(k_min, k_max, 500)
# #TODO - also implement it so the "lines" are either column or rows depending on whats the input
# def L_x(k): return a*k + Qx
# def L_y(k): return d*k + Qy
# def L_z(k): return g*k + Qz
#
# # Helper to find intersection k of two lines:
# def intersection(slope1, intercept1, slope2, intercept2):
#     # (slope1*k+intercept1) = (slope2*k+intercept2)
#     denom = slope1 - slope2
#     if abs(denom) < 1e-14:
#         return None
#     return (intercept2 - intercept1)/denom
#
# # 1. Plot original 3 lines
# fig, ax = plt.subplots()
# Lx_vals = L_x(k_values)
# Ly_vals = L_y(k_values)
# Lz_vals = L_z(k_values)
# ax.plot(k_values, Lx_vals, label='L_x')
# ax.plot(k_values, Ly_vals, label='L_y')
# ax.plot(k_values, Lz_vals, label='L_z')
#
# ax.set_title("Original 3 Lines")
# ax.set_xlabel("k")
# ax.set_ylabel("Cost")
# ax.grid(True)
# ax.legend()
#
# # 2. Find breakpoints (intersections) between pairs (x,y), (x,z), (y,z)
# line_params = {
#     'x': (a,Qx),
#     'y': (d,Qy),
#     'z': (g,Qz)
# }
# pairs = [('x','y'), ('x','z'), ('y','z')]
# breakpoints = []
# for p1,p2 in pairs:
#     s1,i1 = line_params[p1]
#     s2,i2 = line_params[p2]
#     k_int = intersection(s1,i1,s2,i2)
#     if k_int is not None and k_min <= k_int <= k_max:
#         breakpoints.append(k_int)
# breakpoints = sorted(breakpoints)
#
# # 3. Determine minimal line in each interval
# interval_starts = [k_min] + breakpoints
# interval_ends = breakpoints + [k_max]
#
# def val_at(k):
#     return {'x':L_x(k), 'y':L_y(k), 'z':L_z(k)}
#
# interval_min_lines = []
# for start, end in zip(interval_starts, interval_ends):
#     test_k = (start+end)/2.0
#     values = val_at(test_k)
#     min_line = min(values, key=values.get)
#     interval_min_lines.append((start, end, min_line))
#
# # Plot just the minimal line piecewise on a new figure
# fig2, ax2 = plt.subplots()
#
# for (start,end,min_line) in interval_min_lines:
#     # For k in [start,end], min_line is minimal
#     ks = np.linspace(start,end,100)
#     if min_line == 'x':
#         vals = L_x(ks)
#     elif min_line == 'y':
#         vals = L_y(ks)
#     else:
#         vals = L_z(ks)
#     ax2.plot(ks, vals, label=f'{min_line} [{start:.2f}, {end:.2f}]')
#
# # Mark the breakpoints as dashed vertical lines
# for bp in breakpoints:
#     ax2.axvline(x=bp, color='gray', linestyle='--')
#
# ax2.set_title("Piecewise Minimal Line")
# ax2.set_xlabel("k")
# ax2.set_ylabel("Cost")
# ax2.grid(True)
# ax2.legend()
#
# # 4. Now consider a second set of lines that might cause belief equality.
# # Let's say we have two candidate lines L_a'(k) and L_b'(k):
# # For simplicity, define them arbitrarily:
# slope_a_prime = 0.5
# intercept_a_prime = 1.5
# slope_b_prime = 1.5
# intercept_b_prime = 0.0
#
# def L_a_prime(k): return slope_a_prime*k + intercept_a_prime
# def L_b_prime(k): return slope_b_prime*k + intercept_b_prime
#
# # We want to check where these lines intersect the minimal piecewise line.
#
# # We'll create another figure:
# fig3, ax3 = plt.subplots()
#
# # Plot the candidate lines:
# ax3.plot(k_values, L_a_prime(k_values), label="L_a'")
# ax3.plot(k_values, L_b_prime(k_values), label="L_b'")
#
# # Now, for each interval of the minimal line, we will:
# # - Identify which line (x,y,z) is minimal there
# # - Find where L_a' or L_b' equals that minimal line
# # - These intersections are candidates for belief equality breakpoints.
#
# candidate_k_for_equality = []
#
# def line_func_by_name(name):
#     if name == 'x': return (a,Qx)
#     elif name == 'y': return (d,Qy)
#     else: return (g,Qz)
#
# other_lines = [
#     ("L_a'", (slope_a_prime, intercept_a_prime)),
#     ("L_b'", (slope_b_prime, intercept_b_prime))
# ]
#
# for (start,end,min_line) in interval_min_lines:
#     s_min,i_min = line_func_by_name(min_line)
#     # minimal line: s_min*k + i_min
#     # For each other line:
#     for (oname, (s_other, i_other)) in other_lines:
#         k_int = intersection(s_min, i_min, s_other, i_other)
#         # Check if intersection is within this interval:
#         if k_int is not None and start <= k_int <= end:
#             # At this k_int, minimal line = other line => candidate for equality
#             # We must verify it is actually minimal at that k_int.
#             # Check minimal at k_int:
#             vals = val_at(k_int)
#             actual_min = min(vals.values())
#             min_names = [n for n,v in vals.items() if np.isclose(v,actual_min)]
#             # If min_line is indeed minimal there:
#             if min_line in min_names:
#                 candidate_k_for_equality.append(k_int)
#                 # Mark on the plot
#                 y_val = s_other*k_int + i_other
#                 ax3.plot(k_int, y_val, 'ro')
#                 ax3.text(k_int, y_val, f"{min_line}={oname}@{k_int:.2f}",
#                          rotation=45, ha='left', va='bottom', fontsize=8)
#
# # After collecting candidates, pick the highest k:
# if candidate_k_for_equality:
#     max_k = max(candidate_k_for_equality)
#     print("Candidate k-values for equality:", candidate_k_for_equality)
#     print("Highest k:", max_k)
#     # Mark highest k:
#     ax3.axvline(x=max_k, color='red', linestyle=':', linewidth=2)
# else:
#     print("No candidate k found that forces equality.")
#
# # Also plot the minimal line again on this figure to visualize equality points
# for (start,end,min_line) in interval_min_lines:
#     ks = np.linspace(start,end,100)
#     if min_line == 'x':
#         vals = L_x(ks)
#     elif min_line == 'y':
#         vals = L_y(ks)
#     else:
#         vals = L_z(ks)
#     ax3.plot(ks, vals, label=f'Min line: {min_line}')
#
# ax3.set_title("Belief Equality with Additional Lines")
# ax3.set_xlabel("k")
# ax3.set_ylabel("Cost")
# ax3.grid(True)
# ax3.legend()
#
# plt.tight_layout()
# plt.show()
