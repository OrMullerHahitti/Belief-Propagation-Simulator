"""Cost-directed split interventions at exact active-minimizer boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .intervals import decoded_region_candidates


@dataclass(frozen=True)
class SplitAction:
    """Change one original factor's complementary scalar split."""

    edge: int
    weight: float


def prospective_q(kernel: Any) -> np.ndarray:
    """Compute the next damped Q independently of the prospective split."""
    raw = kernel.beliefs()[kernel.ends] - kernel.r
    return kernel.damping * kernel.q + (1 - kernel.damping) * raw


def selectors(cost: np.ndarray, q: np.ndarray, weight: float) -> np.ndarray:
    """Return minimizing sender labels for both clones and both directions."""
    table = np.array([weight, 1 - weight])[:, None, None] * cost
    left = np.argmin(table + q[:, 0, :, None], axis=1)
    right = np.argmin(table + q[:, 1, None, :], axis=2)
    return np.stack((left, right), axis=1)


def effective_boundaries(cost: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Find all strict lower-envelope crossings in the open split interval.

    Enumerating pair intersections is inexpensive for the deliberately small
    domain. A change in the actual minimizer on the adjacent open intervals
    filters out intersections of inactive lines and coincident inactive ties.
    """
    domain = cost.shape[0]
    points = [0.0, 1.0]
    for clone in range(2):
        for direction, matrix in enumerate((cost, cost.T)):
            for u in range(domain):
                for z in range(u + 1, domain):
                    denominators = matrix[u] - matrix[z]
                    mask = denominators != 0
                    alpha = (q[clone, direction, z] - q[clone, direction, u]) / (
                        denominators[mask]
                    )
                    weight = alpha if clone == 0 else 1 - alpha
                    points.extend(weight[(weight > 0) & (weight < 1)].tolist())
    ordered = np.unique(points)
    if len(ordered) <= 2:
        return np.empty(0)
    midpoints = (ordered[1:] + ordered[:-1]) / 2
    signatures = [selectors(cost, q, float(w)) for w in midpoints]
    return np.array(
        [
            float(ordered[i + 1])
            for i in range(len(signatures) - 1)
            if not np.array_equal(signatures[i], signatures[i + 1])
        ]
    )


def branch_candidates(kernel: Any) -> list[SplitAction]:
    """Cover active-row and decoded-belief regions for each scalar split."""
    result = []
    q = prospective_q(kernel).reshape(-1, 2, 2, kernel.problem.d)
    for edge, cost in enumerate(kernel.problem.costs):
        boundaries = effective_boundaries(cost, q[edge])
        for weight in decoded_region_candidates(kernel, q, edge, boundaries):
            if abs(weight - kernel.weights[edge]) > 1e-12:
                result.append(SplitAction(edge, float(weight)))
    return result


def immediate_cost(kernel: Any, action: SplitAction) -> float:
    """Evaluate one action with the native update arithmetic and decoder."""
    trial = kernel.clone()
    trial.weights[action.edge] = action.weight
    trial.step()
    return float(trial.cost)


def rank_candidates(
    kernel: Any, actions: list[SplitAction], budget: int
) -> list[SplitAction]:
    """Attend to actions with the lowest predicted immediate original cost."""
    scored = sorted(
        actions,
        key=lambda action: (
            immediate_cost(kernel, action),
            abs(action.weight - kernel.weights[action.edge]),
            action.edge,
            action.weight,
        ),
    )
    # spread the trial budget across edges before trying further regions.
    selected = []
    seen = set()
    for action in scored:
        if action.edge not in seen:
            selected.append(action)
            seen.add(action.edge)
        if len(selected) == budget:
            return selected
    for action in scored:
        if action not in selected:
            selected.append(action)
        if len(selected) == budget:
            break
    return selected


def choose_split(
    kernel: Any,
    mode: str = "boundary",
    budget: int = 8,
    lookahead: int = 64,
    seed: int = 0,
    guarded: bool = False,
) -> tuple[SplitAction | None, dict]:
    """Select using short simulated continuations; leave live state untouched."""
    if budget < 1 or lookahead < 1:
        raise ValueError("budget and lookahead must be positive")
    if mode == "boundary":
        actions = rank_candidates(kernel, branch_candidates(kernel), budget)
    elif mode == "grid":
        actions = rank_candidates(
            kernel,
            [
                SplitAction(edge, weight)
                for edge in range(len(kernel.weights))
                for weight in (0.05, 0.2, 0.5, 0.8, 0.95)
                if abs(weight - kernel.weights[edge]) > 1e-10
            ],
            budget,
        )
    elif mode == "random":
        rng = np.random.default_rng(seed)
        actions = [
            SplitAction(
                int(rng.integers(len(kernel.weights))), float(rng.uniform(0.01, 0.99))
            )
            for _ in range(budget)
        ]
    else:
        raise ValueError(f"unknown candidate mode: {mode}")
    records = []
    for action in [None, *actions]:
        trial = kernel.clone()
        if action is not None:
            trial.weights[action.edge] = action.weight
        values = []
        assignment_tail = []
        defects = []
        for _ in range(lookahead):
            trial.step()
            values.append(float(trial.cost))
            assignment_tail.append(trial.assignment.copy())
            defects.append(fixed_defect(trial))
        tail = min(16, lookahead)
        score = float(np.mean(values[-tail:]))
        flips = (
            float(np.mean(np.diff(np.array(assignment_tail[-tail:]), axis=0) != 0))
            if tail > 1
            else 0.0
        )
        records.append(
            {
                "action": None if action is None else vars(action),
                "score": score,
                "flips": flips,
                "terminal": values[-1],
                "fixed_defect": max(defects[-tail:]),
            }
        )
    eligible = list(range(len(records)))
    if guarded:
        hold = records[0]
        epsilon = 1e-10 * kernel.problem.scale
        eligible = [0] + [
            i
            for i, record in enumerate(records[1:], start=1)
            if record["flips"] <= hold["flips"] + 1e-12
            if record["fixed_defect"] <= max(hold["fixed_defect"], 1e-7)
            if record["terminal"] <= hold["terminal"] + epsilon
            if record["score"] < hold["score"] - epsilon
        ]
    best = min(eligible, key=lambda i: (records[i]["score"], records[i]["flips"], i))
    return (
        [None, *actions][best],
        {
            "mode": mode,
            "trials": records,
            "chosen": best,
            "simulated_steps": len(records) * lookahead,
        },
    )


def fixed_defect(kernel: Any) -> float:
    """Measure the undamped Q-map defect, including unary-directed messages."""
    beliefs = kernel.beliefs()
    defects = [
        beliefs[kernel.ends] - kernel.r - kernel.q,
        beliefs[:, None, :] - kernel.unary_r - kernel.unary_q,
    ]
    magnitude = max(float(np.max(np.abs(d - d[..., :1]))) for d in defects)
    return magnitude / kernel.problem.scale
