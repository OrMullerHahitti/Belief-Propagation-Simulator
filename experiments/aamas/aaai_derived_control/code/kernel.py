"""Pairwise DMS kernel using the native AAAI arithmetic and cycle schedule.

The kernel keeps raw messages, the original ``argmin`` decoder, split unary
factors, and native aggregate-then-subtract R updates. Floating point ties are
therefore not replaced by a research-specific tolerance. Full bitwise parity
is tested on the supported builders, not assumed for arbitrary graph layouts.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from itertools import combinations, product
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np


def gauge(messages: np.ndarray) -> np.ndarray:
    """Return reference-label differences for diagnostics only."""
    return messages - messages[..., :1]


@dataclass
class PairwiseProblem:
    """An immutable-by-convention objective with ordered pairwise table axes."""

    edges: np.ndarray
    costs: np.ndarray
    unary: np.ndarray
    family: str
    seed: int
    topology: str
    variable_names: tuple[str, ...] = ()
    factor_names: tuple[str, ...] = ()
    unary_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.edges = np.array(self.edges, dtype=int, copy=True)
        self.costs = np.array(self.costs, dtype=float, copy=True)
        self.unary = np.array(self.unary, dtype=float, copy=True)
        if self.unary.ndim != 2 or min(self.unary.shape) < 1:
            raise ValueError("unary must have shape (n, d) with n,d positive")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape (m, 2)")
        if self.costs.shape != (len(self.edges), self.d, self.d):
            raise ValueError("costs must have shape (m, d, d)")
        if np.any(self.edges < 0) or np.any(self.edges >= self.n):
            raise ValueError("edge endpoint out of range")
        if np.any(self.edges[:, 0] == self.edges[:, 1]):
            raise ValueError("self edges are unsupported")
        if not np.isfinite(self.costs).all() or not np.isfinite(self.unary).all():
            raise ValueError("all costs must be finite")
        self.variable_names = self.variable_names or tuple(
            f"x{i + 1}" for i in range(self.n)
        )
        self.factor_names = self.factor_names or tuple(
            f"f{i:04d}" for i in range(len(self.edges))
        )
        self.unary_names = self.unary_names or tuple(f"u{i + 1}" for i in range(self.n))
        for names, size in (
            (self.variable_names, self.n),
            (self.factor_names, len(self.edges)),
            (self.unary_names, self.n),
        ):
            if len(names) != size or len(set(names)) != size:
                raise ValueError("node names must have the correct size and be unique")
        names = self.variable_names + self.factor_names + self.unary_names
        if len(set(names)) != len(names):
            raise ValueError("variable and factor names must not overlap")

    @property
    def n(self) -> int:
        return self.unary.shape[0]

    @property
    def d(self) -> int:
        return self.unary.shape[1]

    @property
    def scale(self) -> float:
        """Sum of factor ranges, used solely to normalize diagnostics."""
        pairwise_range = float(np.ptp(self.costs, axis=(1, 2)).sum())
        unary_range = float(np.ptp(self.unary, axis=1).sum())
        return max(pairwise_range + unary_range, 1e-12)

    def cost(self, assignment: np.ndarray) -> float:
        """Evaluate the original objective, including tiny unary preferences."""
        x = np.asarray(assignment)
        if x.shape != (self.n,) or not np.issubdtype(x.dtype, np.integer):
            raise ValueError("assignment must be an integer vector of length n")
        if np.any(x < 0) or np.any(x >= self.d):
            raise ValueError("assignment label out of range")
        pairwise = self.costs[
            np.arange(len(self.edges)), x[self.edges[:, 0]], x[self.edges[:, 1]]
        ]
        return float(pairwise.sum() + self.unary[np.arange(self.n), x].sum())

    def exact(self, limit: int = 100_000) -> tuple[float, np.ndarray]:
        """Enumerate tiny objectives for evaluation, never online control."""
        if self.d**self.n > limit:
            raise ValueError("exact enumeration exceeds the tiny-graph limit")
        assignments = np.array(list(product(range(self.d), repeat=self.n)))
        costs = self.unary[np.arange(self.n)[None, :], assignments].sum(axis=1)
        for edge, (u, v) in enumerate(self.edges):
            costs += self.costs[edge, assignments[:, u], assignments[:, v]]
        best = int(costs.argmin())
        return float(costs[best]), assignments[best]

    def to_native(self) -> Any:
        """Build the original factor graph without changing table axis order."""
        from propflow import FactorAgent, FactorGraph, VariableAgent

        variables = [VariableAgent(name, self.d) for name in self.variable_names]
        edges = {}
        for name, (u, v), table in zip(self.factor_names, self.edges, self.costs):
            factor = FactorAgent.create_from_cost_table(name, table)
            edges[factor] = [variables[u], variables[v]]
        for i, (name, table) in enumerate(zip(self.unary_names, self.unary)):
            factor = FactorAgent.create_from_cost_table(name, table)
            edges[factor] = [variables[i]]
        return FactorGraph(variables, list(edges), edges)


def extract_problem(graph: Any, family: str, seed: int) -> PairwiseProblem:
    """Extract actual AAAI binary builders; reject arity and unary ambiguity."""
    variables = list(graph.variables)
    indices = {v.name: i for i, v in enumerate(variables)}
    domains = {v.domain for v in variables}
    if len(domains) != 1:
        raise ValueError("uniform domains are required")
    d = domains.pop()
    edges, costs, names = [], [], []
    unary = np.zeros((len(variables), d))
    unary_names = [""] * len(variables)
    for factor in graph.factors:
        ordered = sorted(factor.connection_number, key=factor.connection_number.get)
        if len(ordered) == 2:
            edges.append([indices[name] for name in ordered])
            costs.append(factor.cost_table.copy())
            names.append(factor.name)
        elif len(ordered) == 1:
            i = indices[ordered[0]]
            if unary_names[i]:
                raise ValueError("multiple unary factors per variable are unsupported")
            unary[i] = factor.cost_table
            unary_names[i] = factor.name
        else:
            raise ValueError("only unary and binary factors are supported")
    if not all(unary_names):
        raise ValueError("actual AAAI input must contain one unary factor per variable")
    return PairwiseProblem(
        np.array(edges),
        np.array(costs),
        unary,
        family,
        seed,
        family,
        tuple(v.name for v in variables),
        tuple(names),
        tuple(unary_names),
    )


def make_problem(topology: str, seed: int, d: int = 3) -> PairwiseProblem:
    """Create tiny integer-cost graphs with the AAAI unary preference scale."""
    if topology == "k4":
        edges = list(combinations(range(4), 2))
    elif topology == "bowtie":
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    elif topology == "triangle":
        edges = [(0, 1), (1, 2), (2, 0)]
    elif topology == "edge":
        edges = [(0, 1)]
    else:
        raise ValueError(f"unsupported topology: {topology}")
    rng = np.random.default_rng(seed)
    n = 1 + max(max(edge) for edge in edges)
    return PairwiseProblem(
        np.array(edges),
        rng.integers(100, 200, size=(len(edges), d, d)),
        rng.uniform(0, 0.01, size=(n, d)),
        "random_integer",
        seed,
        topology,
    )


def make_paper_problem(family: str, seed: int) -> PairwiseProblem:
    """Use the current paper's actual generator, including unary preferences."""
    from experiments.aaai.code import problems

    supported = {
        "random_sparse",
        "random_dense",
        "scale_free",
        "graph_coloring",
        "meeting_scheduling",
    }
    if family not in supported:
        raise ValueError(f"unsupported pairwise paper family: {family}")
    graph = getattr(problems, f"build_{family}")(seed)
    return extract_problem(graph, family, seed)


def make_small_problem(topology: str, family: str, seed: int) -> PairwiseProblem:
    """Build random integer or frustrated binary tiny study inputs.

    The frustrated family has constant off-diagonal cost 100 and identical
    diagonals penalized by an integer in [1,100). On odd cycles, not every
    anti-coordination preference can be met. Only tiny unary costs break ties.
    These two families use domains three and two respectively.
    """
    if family in {"random", "random_integer"}:
        problem = make_problem(topology, seed)
        problem.family = family
        return problem
    if family == "frustrated":
        problem = make_problem(topology, seed, d=2)
        rng = np.random.default_rng(seed)
        penalties = rng.integers(1, 100, len(problem.edges))
        problem.costs[:] = 100
        problem.costs[:, 0, 0] += penalties
        problem.costs[:, 1, 1] += penalties
        problem.unary[:] = rng.uniform(0, 0.01, problem.unary.shape)
        problem.family = family
        return problem
    raise ValueError(f"unsupported tiny family: {family}")


class PairwiseKernel:
    """Vectorized native DMS on complementary scalar splits of binary factors.

    ``q`` and ``r`` have shape ``(2*m, 2, d)`` in interleaved clone order.
    Unary factors remain split 0.5/0.5. Setting ``weights`` or ``damping``
    preserves all messages. The damping coefficient multiplies the old Q.
    ``step`` includes the paper's periodic normalization after decoding.
    """

    def __init__(
        self,
        problem: PairwiseProblem,
        weights: float | np.ndarray = 0.5,
        damping: float = 0.9,
    ) -> None:
        self.problem = problem
        self.weights = np.broadcast_to(weights, (len(problem.edges),)).copy()
        self.damping = float(damping)
        self._validate_controls()
        self.ends = np.repeat(problem.edges, 2, axis=0)
        self.q = np.zeros((len(self.ends), 2, problem.d))
        self.r = np.zeros_like(self.q)
        self.unary_q = np.zeros((problem.n, 2, problem.d))
        self.unary_r = np.zeros_like(self.unary_q)
        self.t = 0
        self.assignment = np.zeros(problem.n, dtype=int)
        self.cost = problem.cost(self.assignment)
        self.message_residual = float("inf")
        self._init_incidence()

    def _validate_controls(self) -> None:
        weights = np.asarray(self.weights)
        if any(
            (
                weights.shape != (len(self.problem.edges),),
                not np.isfinite(weights).all(),
                np.any(weights <= 0),
                np.any(weights >= 1),
            )
        ):
            raise ValueError(
                "one finite split weight strictly between 0 and 1 per edge"
            )
        if not np.isfinite(self.damping) or not 0 <= self.damping < 1:
            raise ValueError("old-Q damping must be finite and in [0,1)")

    def _init_incidence(self) -> None:
        p = self.problem
        pair_names = [
            name + suffix for name in p.factor_names for suffix in ("'", "''")
        ]
        unary_names = [
            name + suffix for name in p.unary_names for suffix in ("'", "''")
        ]
        names = np.repeat(pair_names, 2).tolist() + unary_names
        all_ends = np.concatenate((self.ends.ravel(), np.repeat(np.arange(p.n), 2)))
        self._message_order = np.argsort(names, kind="stable")
        self._ordered_ends = all_ends[self._message_order]
        self._all_ends = all_ends
        primal = nx.Graph()
        primal.add_nodes_from(range(p.n))
        primal.add_edges_from(p.edges)
        if not nx.is_connected(primal):
            raise ValueError("native engine requires a connected factor graph")
        # unary leaves attain the diameter: each primal hop takes two edges,
        # and endpoint unary leaves add one edge at either end
        self.graph_diameter = 2 * nx.diameter(primal) + 2

    @property
    def tables(self) -> np.ndarray:
        weights = np.stack((self.weights, 1 - self.weights), axis=1).reshape(-1)
        return np.repeat(self.problem.costs, 2, axis=0) * weights[:, None, None]

    def _messages(self) -> np.ndarray:
        return np.concatenate(
            (
                self.r.reshape(-1, self.problem.d),
                self.unary_r.reshape(-1, self.problem.d),
            )
        )

    def beliefs(self) -> np.ndarray:
        """Sum current R messages in native factor-name send order."""
        messages = self._messages()
        result = np.zeros((self.problem.n, self.problem.d))
        np.add.at(result, self._ordered_ends, messages[self._message_order])
        return result

    def clone(self) -> PairwiseKernel:
        """Copy dynamic state while sharing immutable objective and incidence."""
        result = copy(self)
        for name in ("weights", "q", "r", "unary_q", "unary_r", "assignment"):
            setattr(result, name, getattr(self, name).copy())
        return result

    def step(self) -> PairwiseKernel:
        """Perform one native-ordered Q/R update and a scheduled cycle event."""
        self._validate_controls()
        previous_q, previous_r = gauge(self.q), gauge(self.r)
        previous_unary_q, previous_unary_r = gauge(self.unary_q), gauge(self.unary_r)
        beliefs = self.beliefs()
        raw_q = beliefs[self.ends] - self.r
        raw_unary_q = beliefs[:, None, :] - self.unary_r
        self.q = self.damping * self.q + (1 - self.damping) * raw_q
        self.unary_q = self.damping * self.unary_q + (1 - self.damping) * raw_unary_q
        # native R arithmetic retains subtraction roundoff rather than simplifying it
        aggregate = self.tables
        aggregate += self.q[:, 0, :, None]
        aggregate += self.q[:, 1, None, :]
        self.r[:, 0] = (aggregate - self.q[:, 0, :, None]).min(axis=2)
        self.r[:, 1] = (aggregate - self.q[:, 1, None, :]).min(axis=1)
        self.unary_r = (
            self.problem.unary[:, None, :] * 0.5 + self.unary_q
        ) - self.unary_q
        self.assignment = self.beliefs().argmin(axis=1)
        self.cost = self.problem.cost(self.assignment)
        self.message_residual = max(
            float(np.max(np.abs(gauge(self.q) - previous_q), initial=0)),
            float(np.max(np.abs(gauge(self.r) - previous_r), initial=0)),
            float(np.max(np.abs(gauge(self.unary_q) - previous_unary_q), initial=0)),
            float(np.max(np.abs(gauge(self.unary_r) - previous_unary_r), initial=0)),
        )
        if self.t % self.graph_diameter == 0:
            for messages in (self.q, self.r, self.unary_q, self.unary_r):
                messages -= messages.min(axis=-1, keepdims=True)
        self.t += 1
        return self

    def advance(self, steps: int) -> PairwiseKernel:
        """Run a fixed number of iterations without convergence early stopping."""
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            self.step()
        return self


def native_parity(
    problem: PairwiseProblem,
    damping: float = 0.9,
    weights: float = 0.5,
    steps: int = 40,
    intervention: bool = False,
) -> dict[str, Any]:
    """Compare against actual SplitEngine plus native Q damping and normalization.

    Return discrepancies instead of silently modifying the decoder to pass.
    The optional intervention changes one binary split and damping at step 9.
    """
    from propflow import DampingEngine, SplitEngine

    class NativeSplitDamping(DampingEngine, SplitEngine):
        pass

    class ParitySnapshotManager:
        def capture_step(self, step_index, step, engine):
            return SimpleNamespace(
                step=step_index,
                global_cost=None,
                assignments={var.name: var.curr_assignment for var in engine.var_nodes},
            )

    engine = NativeSplitDamping(
        problem.to_native(),
        damping_factor=damping,
        split_factor=0.5,
        anytime=False,
        normalize_messages=True,
        snapshot_manager=ParitySnapshotManager(),
    )
    kernel = PairwiseKernel(problem, weights, damping)
    by_name = {factor.name: factor for factor in engine.factor_nodes}

    def apply_tables() -> None:
        for e, name in enumerate(problem.factor_names):
            by_name[name + "'"].cost_table = kernel.weights[e] * problem.costs[e]
            by_name[name + "''"].cost_table = (1 - kernel.weights[e]) * problem.costs[e]
        engine.damping_factor = kernel.damping

    apply_tables()
    variables = {v.name: v for v in engine.var_nodes}
    report = {
        "topology": problem.topology,
        "family": problem.family,
        "seed": problem.seed,
        "damping": damping,
        "weights": weights,
        "steps": steps,
        "intervention": intervention,
        "graph_diameter": kernel.graph_diameter,
        "native_graph_diameter": engine.graph_diameter,
        "max_raw_message_error": 0.0,
        "max_gauge_message_error": 0.0,
        "max_cost_error": 0.0,
        "assignment_mismatch_steps": [],
    }
    for t in range(steps):
        if intervention and t == 9:
            kernel.weights[0] = 0.95
            kernel.damping = 0.5
            apply_tables()
        kernel.step()
        engine.step(t)
        snapshot = engine.latest_snapshot()
        native_x = np.array(
            [snapshot.assignments[name] for name in problem.variable_names]
        )
        if not np.array_equal(native_x, kernel.assignment):
            report["assignment_mismatch_steps"].append(t)
        report["max_cost_error"] = max(
            report["max_cost_error"], abs(float(snapshot.global_cost) - kernel.cost)
        )
        try:
            engine._handle_cycle_events(t)
        except StopIteration:
            pass
        for edge, (u, v) in enumerate(problem.edges):
            for clone, suffix in enumerate(("'", "''")):
                factor_name = problem.factor_names[edge] + suffix
                for axis, endpoint in enumerate((u, v)):
                    var = variables[problem.variable_names[endpoint]]
                    native_q = next(
                        msg.data
                        for msg in var.last_iteration
                        if msg.recipient.name == factor_name
                    )
                    native_r = next(
                        msg.data for msg in var.inbox if msg.sender.name == factor_name
                    )
                    for native, fast in (
                        (native_q, kernel.q[2 * edge + clone, axis]),
                        (native_r, kernel.r[2 * edge + clone, axis]),
                    ):
                        report["max_raw_message_error"] = max(
                            report["max_raw_message_error"],
                            float(np.max(np.abs(native - fast))),
                        )
                        report["max_gauge_message_error"] = max(
                            report["max_gauge_message_error"],
                            float(np.max(np.abs(gauge(native) - gauge(fast)))),
                        )
    return report
