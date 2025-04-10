from collections import namedtuple
from typing import Any, List, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
############################################################
# Existing Line class
############################################################
class Line:
    def __init__(self, belief: int|float, constraint: int|float):
        # A, B are two distinct points: (x1, y1) and (x2, y2).
        # We compute line coefficients a1, b1, c1 for the equation:
        #     a1*x + b1*y = c1
        #     self.a = constraint
        self.b = belief

    @staticmethod
    def intersection(line:'Line', other: 'Line') -> Tuple[float,int|float] | None:
        """
        Returns the intersection Point of this line with `other`,
        or None if they are parallel.
        """
        matrix = np.array([[self.a1, self.b1],
                           [other.a1, other.b1]])
        constants = np.array([self.c1, other.c1])

        det = np.linalg.det(matrix)
        if abs(det) < 1e-12:
            # Lines are parallel or numerically unstable
            return None

        x, y = np.linalg.solve(matrix, constants)
        return Point(x, y)

    def y(self, x: float) -> float:
        """
        For a given x, return y on this line:  y = (c1 - a1*x) / b1
        """
        return (self.c1 - self.a1 * x) / self.b1

    @property
    def mid_point(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0,
                     (self.y1 + self.y2) / 2.0)

    def __gt__(self, other: 'Line'):
        """
        Example comparison by the y-coord of the mid_point.
        Only used if you sort lines in some manner.
        """
        return self.mid_point.y > other.mid_point.y


############################################################
# New/updated Envelope class
############################################################
class Envelope:
    def __init__(self, ct: np.ndarray, Q: np.ndarray, k: float):
        """
        Initializes the Envelope class.

        Args:
            ct (np.ndarray): Cost table data (n x n).
            Q (np.ndarray): Message data (dimension n, or n x n, etc.).
            k (float): A scaling factor (though for the lines approach,
                       we'll treat 'k' as the variable).
        """
        self.ct = ct
        self.Q = Q
        self.k = k

    def calculate_line_values(self, k: float) -> np.ndarray:
        """
        Returns an n x n array where each entry is:
              ct[i,j] * k + Q[i or j or i,j] (depending on how Q is interpreted).
        For illustration, here we assume row-based offset Q[i].
        """
        # If your Q is 1D of length n, interpret as Q[i].
        # If your Q is n x n, you might do Q[i, j].
        # If column-based is desired, do Q[j].
        # Make it 'modular' by passing an offset_func if needed.
        return self.ct * k + self.Q.reshape(-1, 1)  # Example: row-based if Q.shape == (n,)

    def calculate_envelope(self, num_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Uses sampling-based approach:
          - Varies k in [1.0 ... 0.0].
          - For each k, compute minimum over all n x n lines.
        This yields a piecewise *looking* curve, but internally it's just sampling.
        """
        k_values = np.linspace(1.0, 0.0, num_points)
        envelope_values = np.zeros_like(k_values)

        for i, kv in enumerate(k_values):
            line_values = self.calculate_line_values(kv)  # shape = (n,n)
            envelope_values[i] = np.min(line_values)
        return k_values, envelope_values

    def calculate_minimum_envelope_segments(self, num_points: int = 200) -> list[tuple[float, float, int, int]]:
        """
        Also sampling-based, but tracks *which* entry in ct is minimal
        for a given k. This yields segments where (row_index, col_index) is
        the argmin region.

        Returns:
            list of (k_start, k_end, row_index, col_index)
        """
        k_values = np.linspace(1.0, 0.0, num_points)
        segments = []
        current_indices = None
        k_start = k_values[0]

        for i, kv in enumerate(k_values):
            line_values = self.calculate_line_values(kv)
            min_indices = np.unravel_index(np.argmin(line_values), line_values.shape)

            if min_indices != current_indices:
                # If new minima appear, that ends the old segment
                if current_indices is not None:
                    segments.append((k_start, k_values[i], *current_indices))
                current_indices = min_indices
                k_start = kv

        # Append the last segment
        segments.append((k_start, k_values[-1], *current_indices))
        return segments

    def create_lines(self, offset_func: Callable[[int, int], float]) -> List[Line]:
        """
        Creates and returns n x n = (number_of_lines) lines.

        Each line is defined by two points:
           (0, offset_func(i,j)) and (1, offset_func(i,j) + ct[i,j])
        which corresponds to the function:
           L_{i,j}(k) = ct[i,j] * k + offset_func(i,j).

        The offset_func is how you keep it 'modular':
        - If you want row-based offset, pass lambda i, j: Q[i].
        - If you want column-based offset, pass lambda i, j: Q[j].
        - If Q is n x n, pass lambda i, j: Q[i, j].
        """
        n = self.ct.shape[0]
        lines = []
        for i in range(n):
            for j in range(n):
                off = offset_func(i, j)
                # Points for k=0 and k=1
                p1 = Point(0.0, off)
                p2 = Point(1.0, off + self.ct[i, j])
                lines.append(Line(p1, p2))
        return lines


############################################################
# Example usage
############################################################
if __name__ == "__main__":
    # Example inputs
    cost_table_data = np.array([
        [2, 3, 5],
        [6, 5, 9],
        [7, 8, 10]
    ])
    # Suppose Q is length 3 for row-based offsets
    message_data = np.array([4, 2, 1])

    # Create Envelope instance
    envelope = Envelope(cost_table_data, message_data, k=1.0)

    # (A) Create lines with row-based offset: Q[i]
    lines_row = envelope.create_lines(offset_func=lambda i, j: envelope.Q[i])

    # (B) Or create lines with column-based offset: Q[j]
    # lines_col = envelope.create_lines(offset_func=lambda i, j: envelope.Q[j])

    # (C) If Q is 2D and you want Q[i, j] as the offset, do:
    # lines_2d = envelope.create_lines(offset_func=lambda i, j: envelope.Q[i, j])

    # Plot a sampling-based envelope
    k_vals, env_vals = envelope.calculate_envelope(num_points=50)
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, env_vals, label='Minimum Envelope (sampled)')
    plt.xlabel('k')
    plt.ylabel('Min over all lines')
    plt.title('Piecewise Linear Envelope (via sampling)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Show how we can see the "segments" of who is minimal
    segments = envelope.calculate_minimum_envelope_segments(num_points=50)
    for seg in segments:
        k_start, k_end, i_row, j_col = seg
        print(f"Segment from k in [{k_start:.2f}, {k_end:.2f}] minimal at ct[{i_row},{j_col}]")

