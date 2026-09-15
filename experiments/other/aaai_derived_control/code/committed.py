"""Pattern-specific fixed-point intervals from common active factor rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _gauge(array: np.ndarray) -> np.ndarray:
    return array - array[..., :1]


@dataclass(frozen=True)
class CommittedWeightInterval:
    """Strict weight interval for one fixed, fully row-committed pattern.

    ``q`` and ``r`` coefficients use interleaved clones and recipient/sender
    endpoint axes ``(2E, 2, d)``. Actual hypothetical fixed messages are
    ``intercept + weight * slope``. These expressions remain defined outside
    the feasible interval, but then at least one specified row is not active.
    """

    lower: float
    upper: float
    feasible: bool
    constraint_intercept: np.ndarray
    constraint_slope: np.ndarray
    q_intercept: np.ndarray
    q_slope: np.ndarray
    r_intercept: np.ndarray
    r_slope: np.ndarray
    weight_intercept: np.ndarray
    weight_slope: np.ndarray

    def margin(self, weight: float) -> float:
        """Return the smallest competitor-minus-selected-row cost gap."""
        return float(np.min(self.constraint_intercept + weight * self.constraint_slope))

    def contains(self, weight: float) -> bool:
        """Whether every row is uniquely selected at this admissible weight."""
        return bool(0 < weight < 1 and self.margin(weight) > 0)


def committed_weight_interval(
    costs: np.ndarray,
    unary: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    pattern: np.ndarray,
    controlled_edge: int | None = None,
) -> CommittedWeightInterval:
    """Derive the strict existence interval of a fixed committed row pattern.

    ``pattern[e,c,s]`` is the sending label chosen by original factor ``e``,
    clone ``c``, sender endpoint ``s`` for *every* receiving label. Different
    clones and different incident edges may choose different sender labels.
    ``controlled_edge=None`` varies every original factor to the same split
    ``w``. Otherwise only that edge varies and the other weights stay fixed.

    Proof: commitment makes normalized factor responses
    ``R(v)=alpha(w)*(C(u*,v)-C(u*,0))`` (transpose at the other endpoint),
    affine in ``w``. The candidate fixed Q is the gauged unary plus incoming
    R sum excluding its own factor, hence also affine. Each selected-row
    inequality is then ``A+B*w>0``. Their intersection with ``(0,1)`` is
    exactly the strict existence interval of this pattern. The candidate Q
    does not depend on old-Q damping ``0<=lambda<1``. In a neighborhood that
    retains the pattern, normalized R is constant and Q errors contract by
    ``lambda``. This proves local attraction of this fixed pattern, not its
    reachability, original-cost quality, or convergence after leaving it.

    This is exact-arithmetic algebra implemented in float64. Boundary ties,
    incomplete commitment, and partial active-row patterns require separate
    analysis. It does not use current Q values to substitute for fixed Q.
    """
    costs = np.asarray(costs, dtype=float)
    unary = np.asarray(unary, dtype=float)
    edges = np.asarray(edges)
    weights = np.asarray(weights, dtype=float)
    pattern = np.asarray(pattern)
    if unary.ndim != 2 or unary.shape[1] < 2:
        raise ValueError("unary must have shape (n,d), with d >= 2")
    n, domain = unary.shape
    count = len(costs)
    valid_shapes = (
        costs.shape == (count, domain, domain),
        edges.shape == (count, 2),
        weights.shape == (count,),
        pattern.shape == (count, 2, 2),
    )
    if not all(valid_shapes) or count < 1:
        raise ValueError("costs, edges, weights and pattern have inconsistent shapes")
    if not all(np.isfinite(value).all() for value in (costs, unary, weights)):
        raise ValueError("all objective entries and weights must be finite")
    if np.any((weights <= 0) | (weights >= 1)):
        raise ValueError("base split weights must be inside (0,1)")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge endpoints must be integers")
    if np.any((edges < 0) | (edges >= n)) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("edge endpoints must be distinct and inside the unary array")
    if not np.issubdtype(pattern.dtype, np.integer):
        raise ValueError("pattern sender labels must be integers")
    if np.any((pattern < 0) | (pattern >= domain)):
        raise ValueError("pattern sender labels are outside the domain")
    if controlled_edge is not None and not 0 <= controlled_edge < count:
        raise ValueError("controlled_edge is outside the original factor array")

    weight_intercept = weights.copy()
    weight_slope = np.zeros(count)
    if controlled_edge is None:
        weight_intercept[:] = 0
        weight_slope[:] = 1
    else:
        weight_intercept[controlled_edge] = 0
        weight_slope[controlled_edge] = 1
    alpha_intercept = np.stack((weight_intercept, 1 - weight_intercept), axis=1)
    alpha_slope = np.stack((weight_slope, -weight_slope), axis=1)
    shape = (2 * count, 2, domain)
    r_intercept = np.zeros(shape)
    r_slope = np.zeros(shape)
    for edge, cost in enumerate(costs):
        for clone in range(2):
            # Q endpoint labels identify the sender; R endpoints are recipients.
            sender_left, sender_right = pattern[edge, clone]
            rows = _gauge(np.stack((cost[:, sender_right], cost[sender_left, :])))
            r_intercept[2 * edge + clone] = alpha_intercept[edge, clone] * rows
            r_slope[2 * edge + clone] = alpha_slope[edge, clone] * rows
    ends = np.repeat(edges, 2, axis=0)
    belief_intercept = unary.copy()
    belief_slope = np.zeros_like(unary)
    np.add.at(belief_intercept, ends.reshape(-1), r_intercept.reshape(-1, domain))
    np.add.at(belief_slope, ends.reshape(-1), r_slope.reshape(-1, domain))
    q_intercept = _gauge(belief_intercept[ends] - r_intercept)
    q_slope = _gauge(belief_slope[ends] - r_slope)
    constants, slopes = [], []
    for edge, cost in enumerate(costs):
        for clone in range(2):
            for sender_side, table in enumerate((cost, cost.T)):
                sender = pattern[edge, clone, sender_side]
                for other in range(domain):
                    if other == sender:
                        continue
                    differences = table[other] - table[sender]
                    current_q = q_intercept[2 * edge + clone, sender_side]
                    current_slope = q_slope[2 * edge + clone, sender_side]
                    cost_contrast = alpha_intercept[edge, clone] * differences
                    slope_contrast = alpha_slope[edge, clone] * differences
                    constants.extend(
                        current_q[other] - current_q[sender] + cost_contrast
                    )
                    slopes.extend(
                        current_slope[other] - current_slope[sender] + slope_contrast
                    )
    constants = np.asarray(constants)
    slopes = np.asarray(slopes)
    positive, negative = slopes > 0, slopes < 0
    lower = max(0.0, float(np.max(-constants[positive] / slopes[positive], initial=0)))
    upper = min(1.0, float(np.min(-constants[negative] / slopes[negative], initial=1)))
    feasible = bool(lower < upper and np.all(constants[slopes == 0] > 0))
    return CommittedWeightInterval(
        lower,
        upper,
        feasible,
        constants,
        slopes,
        q_intercept,
        q_slope,
        r_intercept,
        r_slope,
        weight_intercept,
        weight_slope,
    )
