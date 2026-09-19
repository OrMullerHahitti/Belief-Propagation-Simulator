"""Active-region derivatives for the genuine cost-splitting Q state map."""

from __future__ import annotations

import numpy as np

from .lab import gauge


def q_map(q, tables, ends, damping=0.0):
    """Compute the autonomous Q-to-next-Q map for a fixed decomposition."""
    r0 = (tables + q[:, 1, None, :]).min(axis=2)
    r1 = (tables + q[:, 0, :, None]).min(axis=1)
    r = gauge(np.stack((r0, r1), axis=1))
    beliefs = np.zeros((int(ends.max()) + 1, q.shape[-1]))
    np.add.at(beliefs, ends.reshape(-1), r.reshape(-1, q.shape[-1]))
    return gauge(damping * q + (1 - damping) * (beliefs[ends] - r))


def active_jacobian(q, tables, ends):
    """Exact derivative on the current strict active region, excluding gauge zeros."""
    factors, _, domain = q.shape
    width = domain - 1
    size = factors * 2 * width
    response = np.zeros((size, size))
    selectors = np.empty((factors, 2, domain), dtype=int)
    margin = float("inf")

    def coord(factor, axis, label):
        return (factor * 2 + axis) * width + label - 1

    for factor in range(factors):
        for axis in range(2):
            values = tables[factor] if axis == 0 else tables[factor].T
            values = values + q[factor, 1 - axis][None]
            order = np.sort(values, axis=1)
            margin = min(margin, float(np.min(order[:, 1] - order[:, 0])))
            active = values.argmin(axis=1)
            selectors[factor, axis] = active
            for label in range(1, domain):
                for selected, sign in ((active[label], 1), (active[0], -1)):
                    if selected:
                        response[
                            coord(factor, axis, label),
                            coord(factor, 1 - axis, selected),
                        ] += sign
    if margin <= 1e-10:
        raise ValueError("the active region contains a tie or insufficient margin")
    jacobian = np.zeros_like(response)
    for factor in range(factors):
        for axis in range(2):
            variable = ends[factor, axis]
            for other, other_axis in np.argwhere(ends == variable):
                if other != factor:
                    for label in range(1, domain):
                        jacobian[coord(factor, axis, label)] += response[
                            coord(other, other_axis, label)
                        ]
    return jacobian, selectors, margin
