"""Exact scalar split intervals for the next decoded original-cost outcome."""

from __future__ import annotations

from typing import Any

import numpy as np


def decoded_region_candidates(
    kernel: Any,
    q: np.ndarray,
    edge: int,
    effective_boundaries: np.ndarray,
) -> list[float]:
    """Cover every open one-step decoding region for one complementary split.

    ``q`` is the actual prospective Q state, held fixed during the weight
    search, in either ``(2E, 2, d)`` or ``(E, 2, 2, d)`` form. The supplied
    boundaries must partition all changes of the active factor minimizers.
    On each resulting open interval the winning sender labels are fixed.
    Consequently every next R entry, and each endpoint's next belief, is
    affine in the split weight. A decoded label can then change only where
    two of these affine belief entries intersect. Partitioning at the active
    intersections and taking an interior representative therefore covers
    every open-region decoded assignment, and hence every original cost
    attained there. This is a one-step result, not a terminal-cost guarantee.

    Separate factor-active intervals are retained even when they decode to
    the same assignment, because subsequent dynamics can differ. The equal
    split and current split are also included. Isolated exact-tie assignments
    are not covered by the open-region statement. Arithmetic is float64;
    nearly coincident breakpoints require the same care as min-sum ties.
    """
    costs = np.asarray(kernel.problem.costs)
    domain = int(kernel.problem.d)
    if not 0 <= edge < len(costs):
        raise ValueError("edge is outside the original factor array")
    q = np.asarray(q, dtype=float)
    expected = (2 * len(costs), 2, domain)
    if q.shape == (len(costs), 2, 2, domain):
        q = q.reshape(expected)
    elif q.shape != expected:
        raise ValueError(f"q must have shape {expected} or its per-edge equivalent")
    boundaries = np.asarray(effective_boundaries, dtype=float)
    if any(
        (
            boundaries.ndim != 1,
            not np.isfinite(boundaries).all(),
            np.any((boundaries <= 0) | (boundaries >= 1)),
        )
    ):
        raise ValueError("effective boundaries must be finite and inside (0, 1)")

    # compute the unchanged contribution from every other original factor.
    tables = np.asarray(kernel.tables)
    responses = np.stack(
        (
            (tables + q[:, 1, None, :]).min(axis=2),
            (tables + q[:, 0, :, None]).min(axis=1),
        ),
        axis=1,
    )
    section = slice(2 * edge, 2 * edge + 2)
    responses[section] = 0
    base = np.array(kernel.problem.unary, dtype=float, copy=True)
    ends = np.asarray(kernel.ends)
    np.add.at(base, ends.reshape(-1), responses.reshape(-1, domain))
    left, right = ends[2 * edge]
    cost = costs[edge]
    local_q = q[section]
    factor_bounds = np.unique(np.r_[0.0, boundaries, 1.0])
    result = [0.5]
    current = float(kernel.weights[edge])
    if 0 < current < 1:
        result.append(current)

    for low, high in zip(factor_bounds[:-1], factor_bounds[1:]):
        midpoint = low + (high - low) / 2
        table = np.array([midpoint, 1 - midpoint])[:, None, None] * cost
        winner_left = np.argmin(table + local_q[:, 1, None, :], axis=2)
        winner_right = np.argmin(table + local_q[:, 0, :, None], axis=1)
        intercept = base.copy()
        slope = np.zeros_like(base)
        for clone in range(2):
            sign = 1 if clone == 0 else -1
            for receiver in range(domain):
                sender = winner_left[clone, receiver]
                entry = cost[receiver, sender]
                slope[left, receiver] += sign * entry
                intercept[left, receiver] += local_q[clone, 1, sender] + clone * entry
                sender = winner_right[clone, receiver]
                entry = cost[sender, receiver]
                slope[right, receiver] += sign * entry
                intercept[right, receiver] += local_q[clone, 0, sender] + clone * entry

        points = [low, high]
        for variable in np.unique([left, right]):
            for first in range(domain):
                for second in range(first + 1, domain):
                    denominator = slope[variable, first] - slope[variable, second]
                    if denominator == 0:
                        continue
                    crossing = (
                        intercept[variable, second] - intercept[variable, first]
                    ) / denominator
                    if low < crossing < high:
                        points.append(float(crossing))
        ordered = np.unique(points)
        samples = ordered[:-1] + np.diff(ordered) / 2
        signatures = [
            tuple(np.argmin(intercept + value * slope, axis=1)) for value in samples
        ]
        # inactive line crossings do not delimit a new decoded region.
        active = [low]
        active.extend(
            float(ordered[index + 1])
            for index in range(len(signatures) - 1)
            if signatures[index] != signatures[index + 1]
        )
        active.append(high)
        result.extend(
            first + (second - first) / 2
            for first, second in zip(active[:-1], active[1:])
        )
    return [float(value) for value in sorted(set(result))]
