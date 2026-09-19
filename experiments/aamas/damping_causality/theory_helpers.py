"""Exact rational recurrences for the damping-causality witnesses.

These are algebraic reference models, not replacement BP engines. Their state
contains label-1 minus label-0 Q differences after clone synchronization. The
native zero-message initialization reaches reference state zero after its first
completed update, once the fixed unary R messages have been sent.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def clipped(value: Fraction, half_cost: Fraction) -> Fraction:
    """Clip one exact difference to the split table's transition interval."""
    return max(-half_cost, min(half_cost, value))


def path_raw_q(
    q: Sequence[Fraction],
    diagonal_cost: Fraction = Fraction(16),
    unary_gaps: Sequence[Fraction] = (Fraction(-12), Fraction(-13), Fraction(-4)),
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Compute the undamped equal-split update on the three-variable path.

    Q order is 0->01, 1->01, 1->12, 2->12, with one representative of each
    synchronized clone pair. Each original table has ``diagonal_cost`` on the
    diagonal and zero off the diagonal. A unary gap is phi(1)-phi(0).
    """
    if len(q) != 4 or len(unary_gaps) != 3 or diagonal_cost <= 0:
        raise ValueError(
            "the path requires four Q values, three gaps and positive cost"
        )
    half_cost = Fraction(diagonal_cost) / 2
    a, b, c, d = (clipped(Fraction(value), half_cost) for value in q)
    h0, h1, h2 = map(Fraction, unary_gaps)
    return h0 - b, h1 - a - 2 * d, h1 - 2 * a - d, h2 - c


def path_step(
    q: Sequence[Fraction],
    damping: Fraction,
    diagonal_cost: Fraction = Fraction(16),
    unary_gaps: Sequence[Fraction] = (Fraction(-12), Fraction(-13), Fraction(-4)),
) -> tuple[Fraction, ...]:
    """Apply exact old-Q damping to the reference path update."""
    damping = Fraction(damping)
    if not 0 <= damping < 1:
        raise ValueError("old-Q damping must be in [0,1)")
    raw = path_raw_q(q, diagonal_cost, unary_gaps)
    return tuple(
        damping * Fraction(old) + (1 - damping) * new for old, new in zip(q, raw)
    )


def path_belief_gaps(
    q: Sequence[Fraction],
    diagonal_cost: Fraction = Fraction(16),
    unary_gaps: Sequence[Fraction] = (Fraction(-12), Fraction(-13), Fraction(-4)),
) -> tuple[Fraction, Fraction, Fraction]:
    """Return decoded label-1 minus label-0 belief differences."""
    if len(q) != 4 or len(unary_gaps) != 3 or diagonal_cost <= 0:
        raise ValueError(
            "the path requires four Q values, three gaps and positive cost"
        )
    half_cost = Fraction(diagonal_cost) / 2
    a, b, c, d = (clipped(Fraction(value), half_cost) for value in q)
    h0, h1, h2 = map(Fraction, unary_gaps)
    return h0 - 2 * b, h1 - 2 * a - 2 * d, h2 - 2 * c


def path_cycle_candidate(
    damping: Fraction,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return the affine candidate for the robust path's initial 2-cycle pattern.

    It is an actual strict orbit only while the returned states retain the
    specified clipping pattern: A has a,d interior and b,c above 8; B has all
    four values below -8. Its first boundary is
    ``19*lambda**2 + 66*lambda - 1 == 0``. Beyond that boundary these algebraic
    candidates must not be interpreted as an orbit of the clipped update.
    """
    value = Fraction(damping)
    if not 0 <= value < 1:
        raise ValueError("old-Q damping must be in [0,1)")
    denominator = 1 + value
    first = (
        -4 * (5 * value + 1) / denominator,
        (31 * value**2 - 6 * value + 11) / denominator**2,
        (39 * value**2 + 2 * value + 11) / denominator**2,
        4 * (1 - 3 * value) / denominator,
    )
    second = (
        -4 * (value + 5) / denominator,
        (11 * value**2 + 42 * value - 17) / denominator**2,
        (11 * value**2 + 50 * value - 9) / denominator**2,
        4 * (value - 3) / denominator,
    )
    return first, second


def path_second_cycle_candidate(
    damping: Fraction,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return the second path pattern's affine cycle candidate.

    Relative to the first pattern, B's c component becomes interior to (-8,8).
    The pair is a true orbit only while A has a,d interior and b,c above 8,
    and B has a,b,d below -8 with c interior. Damping 2/125 is an exact witness.
    """
    value = Fraction(damping)
    if not 0 < value < 1:
        raise ValueError("this candidate requires old-Q damping strictly in (0,1)")
    squared_denominator = (1 + value) ** 2 * (2 + value)
    cubic_denominator = value * (1 + value) * (2 + value)
    quartic_numerator = 11 * value**4 + 64 * value**3 + 105 * value**2 + 98 * value - 2
    first = (
        -4 * (5 * value + 1) / (1 + value),
        (31 * value**3 + 94 * value**2 + 131 * value + 20) / squared_denominator,
        3 * (13 * value + 7) / (2 + value),
        -(12 * value**3 + 39 * value**2 + 58 * value - 1) / cubic_denominator,
    )
    second = (
        -4 * (value + 5) / (1 + value),
        quartic_numerator / (value * squared_denominator),
        (11 * value**2 + 50 * value - 1) / (value * (2 + value)),
        -(15 * value**2 + 70 * value + 23) / ((1 + value) * (2 + value)),
    )
    return first, second


def regular_scalar_step(
    q: Fraction,
    damping: Fraction,
    degree: int,
    half_cost: Fraction,
    unary_gap: Fraction,
) -> Fraction:
    """Advance the uniform invariant subspace of an equal-split regular graph.

    This is not a reduction for arbitrary nonuniform perturbations. Its fixed
    point has tied beliefs, so scalar message convergence need not imply
    convergence of the discrete assignment.
    """
    damping, half_cost = Fraction(damping), Fraction(half_cost)
    if not isinstance(degree, int) or degree < 1 or half_cost <= 0:
        raise ValueError("degree and half-cost must be positive")
    if not 0 <= damping < 1:
        raise ValueError("old-Q damping must be in [0,1)")
    q, unary_gap = Fraction(q), Fraction(unary_gap)
    raw = unary_gap - (2 * degree - 1) * clipped(q, half_cost)
    return damping * q + (1 - damping) * raw


def committed_cycle_threshold(
    degree: int, half_cost: Fraction, unary_gap: Fraction
) -> Fraction:
    """Return the upper damping boundary for the uniform fully clipped 2-cycle.

    Strict persistence requires 0 <= damping < the returned value. A nonpositive
    result means that no strict uniform fully clipped alternating orbit exists.
    This boundary does not rule out other, partly clipped oscillations above it.
    """
    if not isinstance(degree, int) or degree < 1 or half_cost <= 0:
        raise ValueError("degree and half-cost must be positive")
    a, h = Fraction(half_cost), abs(Fraction(unary_gap))
    return (2 * a * (degree - 1) - h) / (2 * a * degree + h)
