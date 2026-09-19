"""Observable-state controls; no solver, exact optimum, or future labels inside policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


ARMS = ("hold", "split", "split_undamped", "damping_only", "local_split")
FEATURES = (
    "bias",
    "commitment",
    "row_churn",
    "crossing_fraction",
    "row_margin",
    "undamped_defect",
    "cost_progress",
    "assignment_flips",
    "elapsed",
)


def gauge(value: np.ndarray) -> np.ndarray:
    return value - value[..., :1]


def fixed_defect(kernel: Any) -> float:
    """Damping-independent Q fixed-point defect, including unary messages."""
    beliefs = kernel.beliefs()
    pair = beliefs[kernel.ends] - kernel.r - kernel.q
    unary = beliefs[:, None, :] - kernel.unary_r - kernel.unary_q
    magnitude = max(float(np.max(np.abs(gauge(x)))) for x in (pair, unary))
    return magnitude / kernel.problem.scale


def row_state(
    kernel: Any, weight: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizing sender labels and margins at the next actual damped Q."""
    beliefs = kernel.beliefs()
    q = kernel.damping * kernel.q + (1 - kernel.damping) * (
        beliefs[kernel.ends] - kernel.r
    )
    if weight is None:
        tables = kernel.tables
    else:
        alpha = np.tile([weight, 1 - weight], len(kernel.weights))
        tables = np.repeat(kernel.problem.costs, 2, axis=0) * alpha[:, None, None]
    # axes are clone, sender, receiver; each direction uses its own sender Q.
    left = tables + q[:, 0, :, None]
    right = tables.transpose(0, 2, 1) + q[:, 1, :, None]
    values = np.stack((left, right), axis=1)
    winners = np.argmin(values, axis=2)
    first_two = np.partition(values, 1, axis=2)[:, :, :2]
    margins = first_two[:, :, 1] - first_two[:, :, 0]
    return winners, margins


def observe(
    kernel: Any,
    previous_rows: np.ndarray | None,
    recent_costs: list,
    recent_assignments: list,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nine bounded features from current messages and already observed history."""
    rows, margins = row_state(kernel)
    changed = row_state(kernel, 0.95)[0] != rows
    per_edge = changed.reshape(len(kernel.weights), -1).mean(axis=1)
    commitment = float(np.mean(np.all(rows == rows[..., :1], axis=-1)))
    churn = 0.0 if previous_rows is None else float(np.mean(rows != previous_rows))
    local_scale = np.repeat(
        np.maximum(np.ptp(kernel.problem.costs, axis=(1, 2)), 1e-12), 2
    )
    margin = float(np.median(margins / local_scale[:, None, None]))
    progress = (
        0.0
        if len(recent_costs) < 2
        else (recent_costs[-1] - recent_costs[0]) / kernel.problem.scale
    )
    flips = (
        0.0
        if len(recent_assignments) < 2
        else float(np.mean(np.diff(np.array(recent_assignments), axis=0) != 0))
    )
    feature = np.array(
        [
            1.0,
            commitment,
            churn,
            float(changed.mean()),
            margin,
            fixed_defect(kernel),
            progress,
            flips,
            min(kernel.t / 1536, 1.0),
        ]
    )
    return np.clip(feature, -2, 2), rows, {"edge_crossing": per_edge.tolist()}


@dataclass(frozen=True)
class Schedule:
    """A preserved-message split pulse and independently specified damping phases."""

    name: str
    start: int = 64
    duration: int = 192
    weight: float = 0.95
    damping_before: float = 0.9
    damping_during: float = 0.9
    damping_after: float = 0.9
    split: bool = True

    def __post_init__(self) -> None:
        if self.start < 0 or self.duration < 1 or not 0 < self.weight < 1:
            raise ValueError("invalid split schedule")
        if any(
            not 0 <= d < 1
            for d in (self.damping_before, self.damping_during, self.damping_after)
        ):
            raise ValueError("damping must be in [0,1)")

    def apply(self, kernel: Any) -> None:
        during = self.start <= kernel.t < self.start + self.duration
        kernel.weights[:] = self.weight if self.split and during else 0.5
        if kernel.t < self.start:
            kernel.damping = self.damping_before
        elif during:
            kernel.damping = self.damping_during
        else:
            kernel.damping = self.damping_after


def schedules() -> list[Schedule]:
    """Predetermined factorial damping controls and one-axis pulse variations."""
    result = [
        Schedule("baseline", split=False),
        Schedule("pulse"),
        Schedule(
            "undamped", split=False, damping_before=0, damping_during=0, damping_after=0
        ),
        Schedule("pulse_undamped", damping_before=0, damping_during=0, damping_after=0),
        Schedule("pulse_no_damping_during", damping_during=0),
        Schedule("damping_only_during", split=False, damping_during=0),
        Schedule("pulse_damping_after", damping_before=0, damping_during=0),
        Schedule("damping_only_after", split=False, damping_before=0, damping_during=0),
    ]
    result += [Schedule(f"start_{n}", start=n) for n in (16, 32, 128, 256)]
    result += [Schedule(f"duration_{n}", duration=n) for n in (32, 64, 384)]
    result += [Schedule(f"weight_{w}", weight=w) for w in (0.51, 0.65, 0.8)]
    return result


def action_edges(kernel: Any, arm: int) -> np.ndarray:
    if arm != 4:
        return np.arange(len(kernel.weights), dtype=int)
    current = row_state(kernel)[0]
    crossed = row_state(kernel, 0.95)[0] != current
    score = crossed.reshape(len(kernel.weights), -1).mean(axis=1)
    order = np.argsort(-score, kind="stable")
    selected = order[: max(1, int(np.ceil(len(score) / 4)))]
    return selected[score[selected] > 0]


def apply_arm(kernel: Any, arm: int, elapsed: int, edges: np.ndarray) -> None:
    """Execute one observed action for192 steps followed by64 settling steps."""
    if not 0 <= arm < len(ARMS):
        raise ValueError("unknown action")
    kernel.weights[:] = 0.5
    kernel.damping = 0.9
    if elapsed < 192:
        if arm in (1, 2, 4):
            kernel.weights[edges] = 0.95
        if arm in (2, 3):
            kernel.damping = 0


class LinearSelector:
    """Two linear outcome heads per arm, updated with observed action outcomes.

    Five arms times nine features times two heads gives90 prediction weights.
    Ridge sufficient statistics retain the training prior during online updates.
    No counterfactual simulator or true optimum is available to this class.
    """

    def __init__(self, ridge: float = 0.1, penalty: float = 0.05) -> None:
        self.a = np.tile(np.eye(len(FEATURES)) * ridge, (len(ARMS), 1, 1))
        self.b = np.zeros((len(ARMS), len(FEATURES), 2))
        self.penalty = penalty

    def predict(self, feature: np.ndarray) -> np.ndarray:
        weights = np.linalg.solve(self.a, self.b)
        return np.einsum("f,afh->ah", feature, weights)

    def update(
        self, arm: int, feature: np.ndarray, target: np.ndarray, weight: float = 1.0
    ) -> None:
        if feature.shape != (len(FEATURES),) or target.shape != (2,):
            raise ValueError("incorrect training observation shape")
        if not np.isfinite(feature).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite training observation")
        self.a[arm] += weight * np.outer(feature, feature)
        self.b[arm] += weight * np.outer(feature, target)

    def choose(
        self, feature: np.ndarray, explore: bool, draw: float, random_arm: int
    ) -> tuple[int, list]:
        prediction = self.predict(feature)
        if explore and draw < 0.2:
            return random_arm, prediction.tolist()
        cost = prediction[:, 0]
        bad = np.maximum(prediction[:, 1], 0)
        eligible = bad <= bad[0] + 0.05
        score = np.where(eligible, cost + self.penalty * bad, np.inf)
        return int(np.argmin(score)), prediction.tolist()

    def save(self, path: Any) -> None:
        np.savez_compressed(path, a=self.a, b=self.b, penalty=self.penalty)

    @classmethod
    def load(cls, path: Any) -> LinearSelector:
        with np.load(path) as data:
            result = cls(penalty=float(data["penalty"]))
            result.a, result.b = data["a"].copy(), data["b"].copy()
        return result


def block_target(
    start_cost: float, costs: list, assignments: list, defects: list, scale: float
) -> np.ndarray:
    """Cost progress and instability observed after the chosen256-update block."""
    tail = np.array(assignments[-32:])
    flips = float(np.mean(np.diff(tail, axis=0) != 0))
    bad = flips + min(max(defects[-32:]) * 1000, 1)
    return np.array([(costs[-1] - start_cost) / scale, bad])


def state_trigger(name: str, feature: np.ndarray) -> bool:
    """Predetermined rules; their thresholds are hypotheses tested in validation."""
    _, commitment, churn, crossing, _, _, progress, flips, _ = feature
    if name in ("state_cross10", "state_restore"):
        return bool(crossing >= 0.1 and churn <= 0.05)
    if name == "state_commit50":
        return bool(commitment >= 0.5 and crossing >= 0.1)
    if name == "state_plateau":
        return bool(abs(progress) < 1e-10 and flips == 0 and crossing >= 0.05)
    raise ValueError(f"unknown state trigger {name}")


def schedule_dict(schedule: Schedule) -> dict:
    return asdict(schedule)
