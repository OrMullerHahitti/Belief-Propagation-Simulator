"""Small, gauge-fixed pairwise laboratory, cross-checked against PropFlow."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import itertools

import numpy as np


TOPOLOGIES = {
    "edge": [(0, 1)],
    "triangle": [(0, 1), (1, 2), (2, 0)],
    "bowtie": [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
    "k4": list(itertools.combinations(range(4), 2)),
}


def gauge(message: np.ndarray) -> np.ndarray:
    """Remove the additive constant using a fixed reference label."""
    return message - message[..., :1]


def decode(beliefs: np.ndarray, scale: float) -> np.ndarray:
    """Choose the first label within 1e-12 objective scales of the minimum."""
    near_minimum = beliefs <= beliefs.min(axis=-1, keepdims=True) + 1e-12 * scale
    return np.argmax(near_minimum, axis=-1)


@dataclass(frozen=True)
class Problem:
    """Original pairwise objective with ordered cost-table axes."""

    edges: np.ndarray
    costs: np.ndarray
    family: str = "custom"
    seed: int = 0
    topology: str = "custom"

    def __post_init__(self) -> None:
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape (m, 2)")
        if self.costs.ndim != 3 or self.costs.shape[0] != len(self.edges):
            raise ValueError("costs must have shape (m, d, d)")
        if self.costs.shape[1] != self.costs.shape[2] or not len(self.edges):
            raise ValueError("use nonempty pairwise tables with a common domain")
        if not np.isfinite(self.costs).all() or np.any(self.edges < 0):
            raise ValueError("costs must be finite and variable indices nonnegative")
        if np.any(self.edges[:, 0] == self.edges[:, 1]):
            raise ValueError("self edges are unsupported")

    @property
    def n(self) -> int:
        return int(self.edges.max()) + 1

    @property
    def d(self) -> int:
        return self.costs.shape[1]

    @property
    def scale(self) -> float:
        return max(float(np.ptp(self.costs, axis=(1, 2)).sum()), 1e-12)

    def cost(self, assignment: np.ndarray) -> float:
        return float(
            self.costs[
                np.arange(len(self.edges)),
                assignment[self.edges[:, 0]],
                assignment[self.edges[:, 1]],
            ].sum()
        )

    def exact(self) -> tuple[float, np.ndarray]:
        """Enumerate the deliberately tiny validation problem, never training labels."""
        if self.d**self.n > 100000:
            raise ValueError("exact validation is restricted to tiny graphs")
        assignments = np.array(list(itertools.product(range(self.d), repeat=self.n)))
        values = self.costs[
            np.arange(len(self.edges))[None, :],
            assignments[:, self.edges[:, 0]],
            assignments[:, self.edges[:, 1]],
        ].sum(axis=1)
        index = int(np.argmin(values))
        return float(values[index]), assignments[index]


def make_problem(topology: str, family: str, seed: int) -> Problem:
    """Generate a deterministic small objective; all methods reuse these tables."""
    rng = np.random.default_rng(seed)
    edges = np.array(TOPOLOGIES[topology], dtype=int)
    if family == "random":
        costs = rng.uniform(0, 10, (len(edges), 3, 3))
    elif family == "frustrated":
        costs = rng.uniform(1, 10, (len(edges), 1, 1)) * np.eye(2)[None]
        costs += rng.uniform(0, 0.05, costs.shape)
    else:
        raise ValueError(f"unknown family: {family}")
    return Problem(edges, costs, family, seed, topology)


@dataclass(frozen=True)
class Action:
    """An edge-specific split intervention and a global old-Q damping weight."""

    edge: int
    weight: float
    damping: float


class PairwiseLab:
    """Experimental vectorized Q/R kernel; not a public PropFlow engine."""

    def __init__(
        self, problem: Problem, split: float | None = 0.5, damping: float = 0.9
    ) -> None:
        self.problem = problem
        self.clones = 1 if split is None else 2
        self.weights = np.full(len(problem.edges), 1.0 if split is None else split)
        self.damping = damping
        self.ends = np.repeat(problem.edges, self.clones, axis=0)
        self.q = np.zeros((len(self.ends), 2, problem.d))
        self.r = self.q.copy()
        self.previous_r = self.r.copy()
        self.residual = np.zeros(len(problem.edges))
        self.residual2 = self.residual.copy()
        self.assignment = np.zeros(problem.n, dtype=int)
        self.cost = problem.cost(self.assignment)
        self.best_cost = self.cost
        self.best_assignment = self.assignment.copy()
        self.t = 0
        self.costs: list[float] = []
        self.assignments: list[np.ndarray] = []
        self.residuals: list[float] = []
        self.message_residuals: list[float] = []
        self.flips: list[float] = []
        self._validate()

    def _validate(self) -> None:
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("split weights must be positive and finite")
        if self.clones == 2 and np.any(self.weights >= 1):
            raise ValueError("two-clone weights must be strictly below one")
        if not np.isfinite(self.damping) or not 0 <= self.damping < 1:
            raise ValueError("damping must be finite and in [0, 1)")

    @property
    def tables(self) -> np.ndarray:
        if self.clones == 1:
            return self.problem.costs
        weights = np.stack((self.weights, 1 - self.weights), axis=1)
        return (self.problem.costs[:, None] * weights[:, :, None, None]).reshape(
            -1, self.problem.d, self.problem.d
        )

    def clone(self) -> PairwiseLab:
        return copy.deepcopy(self)

    def act(self, action: Action) -> None:
        """Change future cost decomposition while preserving message history."""
        if not np.isfinite(action.damping) or not 0 <= action.damping < 1:
            raise ValueError("invalid damping action")
        if action.edge >= 0:
            if self.clones != 2 or action.edge >= len(self.weights):
                raise ValueError("invalid split edge")
            if not np.isfinite(action.weight) or not 0 < action.weight < 1:
                raise ValueError("invalid split weight")
            self.weights[action.edge] = action.weight
        self.damping = action.damping

    def beliefs(self) -> np.ndarray:
        beliefs = np.zeros((self.problem.n, self.problem.d))
        np.add.at(beliefs, self.ends.reshape(-1), self.r.reshape(-1, self.problem.d))
        return beliefs

    def step(self) -> None:
        old_q = self.q
        raw_q = self.beliefs()[self.ends] - self.r
        self.q = gauge(self.damping * self.q + (1 - self.damping) * raw_q)
        tables = self.tables
        r0 = (tables + self.q[:, 1, None, :]).min(axis=2)
        r1 = (tables + self.q[:, 0, :, None]).min(axis=1)
        new_r = gauge(np.stack((r0, r1), axis=1))
        shape = (len(self.weights), self.clones, 2, self.problem.d)
        self.residual = np.abs(new_r - self.r).reshape(shape).max(axis=(1, 2, 3))
        self.residual2 = (
            np.abs(new_r - self.previous_r).reshape(shape).max(axis=(1, 2, 3))
        )
        self.previous_r, self.r = self.r, new_r
        previous_assignment = self.assignment
        self.assignment = decode(self.beliefs(), self.problem.scale)
        self.cost = self.problem.cost(self.assignment)
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_assignment = self.assignment.copy()
        self.costs.append(self.cost)
        self.assignments.append(self.assignment.copy())
        self.flips.append(float(np.mean(previous_assignment != self.assignment)))
        self.residuals.append(float(self.residual.max()))
        self.message_residuals.append(
            max(self.residuals[-1], float(np.max(np.abs(self.q - old_q))))
        )
        self.t += 1

    def advance(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def actions(self) -> list[Action]:
        result = [Action(-1, 0.5, damp) for damp in (0.0, 0.5, 0.9)]
        for edge, weight in enumerate(self.weights):
            for target in (0.05, 0.5, 0.95):
                if abs(weight - target) > 1e-8:
                    result.extend(Action(edge, target, damp) for damp in (0, 0.5, 0.9))
        return result

    def features(self, actions: list[Action], horizon: int) -> np.ndarray:
        """Return bounded local/global observations with no exact-solver information."""
        p = self.problem
        scale = p.scale
        b = self.beliefs()
        margins = np.partition(b, 1, axis=1)[:, 1] - b.min(axis=1)
        br = np.zeros_like(b)
        for e, (u, v) in enumerate(p.edges):
            br[u] += p.costs[e, :, self.assignment[v]]
            br[v] += p.costs[e, self.assignment[u], :]
        br_disagree = decode(br, scale) != self.assignment
        sibling = (
            np.abs(self.r)
            .reshape(len(p.edges), self.clones, 2, p.d)
            .max(axis=(1, 2, 3))
        )
        rows = []
        for action in actions:
            edge = action.edge
            selected = edge >= 0
            if selected:
                u, v = p.edges[edge]
                local = [
                    self.residual[edge] / scale,
                    self.residual2[edge] / scale,
                    sibling[edge] / scale,
                    float(min(margins[u], margins[v])) / scale,
                    p.costs[edge, self.assignment[u], self.assignment[v]] / scale,
                    self.weights[edge],
                    action.weight - self.weights[edge],
                    float(br_disagree[[u, v]].mean()),
                ]
            else:
                local = [
                    self.residual.mean() / scale,
                    self.residual2.mean() / scale,
                    sibling.mean() / scale,
                    margins.mean() / scale,
                    self.cost / (len(p.edges) * scale),
                    self.weights.mean(),
                    0.0,
                    float(br_disagree.mean()),
                ]
            rows.append(
                [
                    self.t / horizon,
                    (self.cost - self.best_cost) / scale,
                    float(np.mean(self.flips[-8:])) if self.flips else 0.0,
                    self.residual.max() / scale,
                    self.residual2.max() / scale,
                    *local,
                    self.damping,
                    action.damping,
                    float(selected),
                ]
            )
        return np.clip(np.asarray(rows), -2, 2)

    def reward(self, before: float, block: int) -> float:
        improvement = (before - self.cost) / self.problem.scale
        flip_penalty = 0.02 * float(np.mean(self.flips[-block:]))
        residual_penalty = 0.005 * min(self.residuals[-1] / self.problem.scale, 1)
        return improvement - flip_penalty - residual_penalty

    def metrics(self) -> dict:
        if not self.costs:
            raise ValueError("run at least one iteration before requesting metrics")
        optimum, _ = self.problem.exact()
        tail = self.assignments[-16:]
        period2 = False
        if len(tail) == 16:
            period2 = all(
                [
                    not np.array_equal(tail[-1], tail[-2]),
                    np.array_equal(tail[:-2], tail[2:]),
                ]
            )
        return {
            "final_cost": self.cost,
            "best_cost": self.best_cost,
            "optimum": optimum,
            "final_gap": (self.cost - optimum) / self.problem.scale,
            "best_gap": (self.best_cost - optimum) / self.problem.scale,
            "optimal": bool(abs(self.cost - optimum) < 1e-8),
            "assignment_stable": bool(
                len(tail) == 16 and np.all(np.array(tail) == tail[-1])
            ),
            "message_stable": all(
                [
                    len(self.residuals) >= 16,
                    max(self.message_residuals[-16:]) / self.problem.scale < 1e-7,
                ]
            ),
            "assignment_period2": period2,
            "final_residual": self.residuals[-1] / self.problem.scale,
            "final_qr_residual": self.message_residuals[-1] / self.problem.scale,
        }


class TinyScorer:
    """145-parameter shared action scorer, trained by direct observed-reward regression."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.2, (16, 8))
        self.b1 = np.zeros(8)
        self.w2 = rng.normal(0, 0.05, 8)
        self.b2 = np.zeros(1)
        self.updates = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(np.einsum("ni,ij->nj", x, self.w1) + self.b1)
        return np.einsum("ni,i->n", h, self.w2) + self.b2

    def update(self, x: np.ndarray, y: np.ndarray, rate: float = 0.03) -> None:
        h = np.tanh(np.einsum("ni,ij->nj", x, self.w1) + self.b1)
        error = np.clip(np.einsum("ni,i->n", h, self.w2) + self.b2 - y, -1, 1) / len(x)
        dh = error[:, None] * self.w2[None] * (1 - h * h)
        self.w2 -= rate * np.einsum("ni,n->i", h, error)
        self.b2 -= rate * error.sum()
        self.w1 -= rate * np.einsum("ni,nj->ij", x, dh)
        self.b1 -= rate * dh.sum(axis=0)
        self.updates += 1

    def save(self, path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)


def native_parity(problem: Problem, split: float | None, damping: float) -> float:
    """Compare research Q/R/assignment/cost traces with native PropFlow snapshots."""
    from propflow import BPEngine, FactorAgent, FactorGraph, VariableAgent
    from propflow import MinSumComputator
    from propflow.policies import damp

    variables = [
        VariableAgent(f"x{i if i % 2 == 0 else i * 11}", problem.d)
        for i in range(problem.n)
    ]
    factors, edges = [], {}
    lab = PairwiseLab(problem, split, damping)
    for f, ((u, v), table) in enumerate(zip(lab.ends, lab.tables)):
        factor = FactorAgent.create_from_cost_table(f"f{f}", table)
        factors.append(factor)
        edges[factor] = [variables[u], variables[v]]
    graph = FactorGraph(variable_li=variables, factor_li=factors, edges=edges)

    class CanonicalMinSum(MinSumComputator):
        def get_assignment(self, belief):
            return int(decode(np.asarray(belief), problem.scale))

    class ControlledEngine(BPEngine):
        def post_var_compute(self, var):
            damp(var, self.control_damping)
            for message in var.outbox:
                message.data = gauge(message.data)
            var.append_last_iteration()

        def post_factor_compute(self, factor, iteration):
            for message in factor.outbox:
                message.data = gauge(message.data)

    engine = ControlledEngine(
        graph, computator=CanonicalMinSum(), normalize_messages=False, anytime=False
    )
    engine.control_damping = damping
    worst = 0.0
    for t in range(24):
        if t == 9:
            action = Action(0 if split is not None else -1, 0.95, 0.5)
            lab.act(action)
            engine.control_damping = action.damping
            for factor, table in zip(factors, lab.tables):
                factor.cost_table = table.copy()
        lab.step()
        engine.step(t)
        snapshot = engine.latest_snapshot()
        for f, (u, v) in enumerate(lab.ends):
            for axis, variable in enumerate((u, v)):
                name = variables[variable].name
                for native, research in (
                    (snapshot.Q[(name, f"f{f}")], lab.q[f, axis]),
                    (snapshot.R[(f"f{f}", name)], lab.r[f, axis]),
                ):
                    worst = max(worst, float(np.max(np.abs(gauge(native) - research))))
        native_assignment = np.array([snapshot.assignments[v.name] for v in variables])
        np.testing.assert_array_equal(native_assignment, lab.assignment)
        np.testing.assert_allclose(problem.cost(native_assignment), lab.cost, atol=1e-9)
        np.testing.assert_allclose(snapshot.global_cost, lab.cost, atol=1e-9)
    if worst > 1e-7:
        raise AssertionError(f"native message discrepancy: {worst}")
    return worst
