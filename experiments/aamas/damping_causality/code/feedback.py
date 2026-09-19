"""Controlled full-scale feedback map for equal-clone binary Min-sum.

This diagnostic separates external-field multiplication from sibling return.
Only gains (1, 0) and (2, 1) represent native unsplit and equally split BP.
All other combinations are causal interventions, not proposed solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.aamas.aaai_derived_control.code.kernel import PairwiseProblem


def factor_response(problem: PairwiseProblem, q: np.ndarray) -> np.ndarray:
    """Compute full-table R differences from directed binary Q differences."""
    if problem.d != 2 or q.shape != (len(problem.edges), 2):
        raise ValueError("binary domains and one Q difference per endpoint required")
    fields = np.stack((np.zeros_like(q), q), axis=-1)
    result = np.empty_like(q)
    to_first = (problem.costs + fields[:, 1, None, :]).min(axis=2)
    to_second = (problem.costs + fields[:, 0, :, None]).min(axis=1)
    result[:, 0] = to_first[:, 1] - to_first[:, 0]
    result[:, 1] = to_second[:, 1] - to_second[:, 0]
    return result


def belief_differences(problem: PairwiseProblem, r: np.ndarray) -> np.ndarray:
    """Return original-variable belief differences, including unary factors."""
    result = problem.unary[:, 1] - problem.unary[:, 0]
    np.add.at(result, problem.edges.ravel(), r.ravel())
    return result


@dataclass
class FeedbackState:
    """Full-scale Q/R differences after a completed native-style update."""

    q: np.ndarray
    r: np.ndarray
    step: int = 0

    @classmethod
    def zeros(cls, problem: PairwiseProblem) -> FeedbackState:
        return cls(np.zeros((len(problem.edges), 2)), np.zeros((len(problem.edges), 2)))

    def copy(self) -> FeedbackState:
        return FeedbackState(self.q.copy(), self.r.copy(), self.step)


def advance(
    problem: PairwiseProblem,
    state: FeedbackState,
    damping: float = 0.0,
    external_gain: float = 2.0,
    sibling_gain: float = 1.0,
) -> FeedbackState:
    """Advance from the same message state under the specified intervention.

    Negative sibling gain is the binary label-swapped return: it preserves the
    absolute message difference and reverses its evidence direction. Unary evidence
    starts in the first factor phase, matching the native zero initialization.
    """
    controls = (damping, external_gain, sibling_gain)
    if not all(np.isfinite(controls)) or not 0 <= damping < 1:
        raise ValueError("finite gains and old-Q damping in [0,1) required")
    beliefs = belief_differences(problem, state.r)
    if state.step == 0:
        beliefs -= problem.unary[:, 1] - problem.unary[:, 0]
    external = beliefs[problem.edges] - state.r
    raw_q = external_gain * external + sibling_gain * state.r
    q = damping * state.q + (1 - damping) * raw_q
    return FeedbackState(q, factor_response(problem, q), state.step + 1)


def active_jacobian(
    problem: PairwiseProblem,
    q: np.ndarray,
    damping: float = 0.0,
    external_gain: float = 2.0,
    sibling_gain: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Return the exact affine-branch Q Jacobian and minimum selector margin.

    At zero margin the map has a kink; the returned selected-branch derivative is
    not a unique derivative and must not be used as a smooth stability certificate.
    The state must already satisfy R=response(Q), i.e. follow a factor phase.
    """
    m = len(problem.edges)
    if problem.d != 2 or q.shape != (m, 2):
        raise ValueError("binary Q shape required")
    fields = np.stack((np.zeros_like(q), q), axis=-1)
    response_derivative = np.zeros((2 * m, 2 * m))
    margin = float("inf")
    for edge, table in enumerate(problem.costs):
        for target in (0, 1):
            scores = table + (
                fields[edge, 1][None, :] if target == 0 else fields[edge, 0][:, None]
            )
            if target == 1:
                scores = scores.T
            selectors = scores.argmin(axis=1)
            margin = min(margin, float(np.abs(scores[:, 1] - scores[:, 0]).min()))
            response_derivative[2 * edge + target, 2 * edge + 1 - target] = (
                selectors[1] - selectors[0]
            )
    aggregate_derivative = np.zeros((2 * m, 2 * m))
    endpoints = problem.edges.ravel()
    for i, node in enumerate(endpoints):
        aggregate_derivative[i, endpoints == node] = external_gain
        aggregate_derivative[i, i] = sibling_gain
    scaled_aggregate = (1 - damping) * aggregate_derivative
    jacobian = damping * np.eye(2 * m) + scaled_aggregate @ response_derivative
    return jacobian, margin
