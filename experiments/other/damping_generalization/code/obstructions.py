"""Exact fixed-point feasibility checks for complementary pairwise splitting.

These finite-domain certificates use the original objective, not simulated
message residuals. Rational arithmetic makes strict branch boundaries explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Sequence

import numpy as np

from experiments.other.aaai_derived_control.code.kernel import PairwiseProblem


def _rational(value: float | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


@dataclass(frozen=True)
class CommitmentCertificate:
    """Candidate messages and exact minima of every required strict inequality."""

    assignment: tuple[int, ...]
    local_margin: Fraction
    commitment_margin: Fraction
    q: np.ndarray
    r: np.ndarray

    @property
    def strict(self) -> bool:
        """Whether the candidate is a consistent, fully committed fixed point."""
        return self.local_margin > 0 and self.commitment_margin > 0


def certify_assignment(
    problem: PairwiseProblem,
    assignment: Sequence[int],
    weights: Fraction | Sequence[Fraction] = Fraction(1, 2),
) -> CommitmentCertificate:
    """Certify an assignment by exact local gaps and interaction cross-differences.

    Costs stored as floating point in ``PairwiseProblem`` are interpreted as
    their displayed decimal rationals. Q/R arrays follow PairwiseKernel clone
    order and endpoint order, with label zero normalized to zero.
    """
    x = tuple(assignment)
    if len(x) != problem.n or any(
        not isinstance(a, (int, np.integer)) or not 0 <= a < problem.d for a in x
    ):
        raise ValueError("assignment requires one valid integer label per variable")
    if problem.d < 2 or len(problem.edges) < 1:
        raise ValueError("certificate requires pairwise edges and at least two labels")
    if isinstance(weights, (int, float, Fraction, np.number)):
        weights = [_rational(weights)] * len(problem.edges)
    weights = tuple(map(_rational, weights))
    if len(weights) != len(problem.edges) or any(not 0 < w < 1 for w in weights):
        raise ValueError("one complementary weight in (0,1) is required per edge")
    costs = np.vectorize(_rational)(problem.costs)
    local = np.vectorize(_rational)(problem.unary)
    for edge, (u, v) in enumerate(problem.edges):
        local[u] += costs[edge, :, x[v]]
        local[v] += costs[edge, x[u], :]
    q = np.empty((2 * len(problem.edges), 2, problem.d), dtype=object)
    r = np.empty_like(q)
    local_margin = min(
        local[i, a] - local[i, x[i]]
        for i in range(problem.n)
        for a in range(problem.d)
        if a != x[i]
    )
    margins = []
    for edge, (u, v) in enumerate(problem.edges):
        for clone, weight in enumerate((weights[edge], 1 - weights[edge])):
            index = 2 * edge + clone
            for axis, (i, j, table) in enumerate(
                ((u, v, costs[edge]), (v, u, costs[edge].T))
            ):
                r[index, axis] = weight * table[:, x[j]]
                q[index, axis] = local[i] - r[index, axis]
                for a in range(problem.d):
                    if a == x[i]:
                        continue
                    gap = local[i, a] - local[i, x[i]]
                    for b in range(problem.d):
                        cross = table[a, b] - table[x[i], b]
                        cross -= table[a, x[j]] - table[x[i], x[j]]
                        margins.append(gap + weight * cross)
    q -= q[..., :1].copy()
    r -= r[..., :1].copy()
    return CommitmentCertificate(x, local_margin, min(margins), q, r)


def anti_equality_cycle(
    size: int, unary_gaps: Sequence[int] | None = None
) -> PairwiseProblem:
    """Construct an integer-cost cycle with diagonal four and off-diagonal zero."""
    if not isinstance(size, int) or size < 3:
        raise ValueError("a simple cycle requires at least three vertices")
    gaps = np.zeros(size, dtype=int) if unary_gaps is None else np.asarray(unary_gaps)
    if gaps.shape != (size,):
        raise ValueError("one unary gap per variable is required")
    unary = np.stack((np.maximum(-gaps, 0), np.maximum(gaps, 0)), axis=1)
    return PairwiseProblem(
        np.array([(i, (i + 1) % size) for i in range(size)]),
        np.tile([[4, 0], [0, 4]], (size, 1, 1)),
        unary,
        "damping_obstruction",
        0,
        f"cycle{size}",
    )


def enumerate_certificates(
    problem: PairwiseProblem, weights: Fraction = Fraction(1, 2)
) -> list[CommitmentCertificate]:
    """Enumerate only bounded tiny fixtures; this is not an online algorithm."""
    if problem.d**problem.n > 4096:
        raise ValueError("exhaustive certificate check exceeds the tiny fixture limit")
    return [
        certify_assignment(problem, assignment, weights)
        for assignment in product(range(problem.d), repeat=problem.n)
    ]


def biased_cycle_bad_fixed_point(
    weight: Fraction,
    diagonal_cost: Fraction = Fraction(4),
    unary_bias: Fraction = Fraction(1),
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact original-gauge Q/R gaps at a suboptimal four-cycle fixed point.

    The parameter family has original diagonal cost C, off-diagonal zero and
    unary (b,0) at vertex zero, with 0 < b < C/2. The continuation is valid for
    1/2 <= weight <= 1-b/(8C), including active boundaries. Endpoint and clone
    ordering agree with ``anti_equality_cycle(4, [-1, 0, 0, 0])``.
    """
    weight = Fraction(weight)
    diagonal_cost, unary_bias = Fraction(diagonal_cost), Fraction(unary_bias)
    if not 0 < unary_bias < diagonal_cost / 2:
        raise ValueError("the parameter family requires 0 < unary bias < C/2")
    threshold = 1 - unary_bias / (8 * diagonal_cost)
    if not Fraction(1, 2) <= weight <= threshold:
        raise ValueError("the bad continuation requires 1/2 <= weight <= 1-b/(8C)")
    strong, weak = diagonal_cost * weight, diagonal_cost * (1 - weight)
    messages = [min(strong, strong - unary_bias + 2 * j * weak) for j in range(1, 4)]
    messages.append(strong)
    ends = np.repeat(np.array([(i, (i + 1) % 4) for i in range(4)]), 2, axis=0)
    r = np.full((8, 2), weak, dtype=object)
    for edge in range(4):
        r[2 * edge, 0] = messages[3 - edge]
        r[2 * edge, 1] = messages[edge]
    beliefs = np.array([-unary_bias, Fraction(0), Fraction(0), Fraction(0)])
    for clone, (u, v) in enumerate(ends):
        beliefs[u] += r[clone, 0]
        beliefs[v] += r[clone, 1]
    q = beliefs[ends] - r
    signs = np.array([1, -1, 1, -1], dtype=object)
    return q * signs[ends], r * signs[ends]


def biased_cycle_positive_subsolution(
    diagonal_cost: Fraction = Fraction(4), unary_bias: Fraction = Fraction(1)
) -> tuple[np.ndarray, np.ndarray]:
    """Return a common positive subsolution for every 1/2 <= weight <= 1-b/(8C).

    Any Q/R state above this subsolution in the alternating vertex-sign gauge
    stays above it, even under changing weights and damping in the stated range.
    Return message differences in the original anti-equality gauge.
    """
    diagonal_cost, unary_bias = Fraction(diagonal_cost), Fraction(unary_bias)
    if not 0 < unary_bias < diagonal_cost / 2:
        raise ValueError("the parameter family requires 0 < unary bias < C/2")
    profile = [unary_bias * Fraction(j, 4) for j in range(1, 5)]
    q = np.full((8, 2), unary_bias / 8, dtype=object)
    for edge in range(4):
        q[2 * edge, 0] = profile[edge]
        q[2 * edge, 1] = profile[3 - edge]
    r = q[:, ::-1].copy()
    ends = np.repeat(np.array([(i, (i + 1) % 4) for i in range(4)]), 2, axis=0)
    signs = np.array([1, -1, 1, -1], dtype=object)
    return q * signs[ends], r * signs[ends]
