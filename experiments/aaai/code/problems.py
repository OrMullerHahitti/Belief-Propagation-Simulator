"""Benchmark problem generators for the AAAI experiments.

Benchmarks follow the professor's spec; scale-free nets and meeting
scheduling follow the descriptions in Cohen, Galiki & Zivan, "Governing
convergence of Max-sum on DCOPs through damping and splitting", AIJ 279
(2020), Section 6:

- random_sparse:       50 agents, domain 10, p1 = 0.1, integer costs U[100, 200)
- random_dense:        50 agents, domain 10, p1 = 0.6, integer costs U[100, 200)
- random_ternary:      50 agents, domain 10, true arity-3 constraints,
                       p3 = 2 * 0.1 / (50 - 2), integer costs U[100, 200)
- graph_coloring:      50 agents, domain 3 (colors), p1 = 0.05, equal = 10, else 0
- scale_free:          Barabasi-Albert: 7 initial agents randomly connected, each
                       new agent attaches to 3 existing agents preferentially,
                       n = 50, domain 10, integer costs U[100, 200) (the AIJ
                       Section 6.2 cost range used for the splitting experiments)
- meeting_scheduling:  90 agents schedule 20 meetings into 20 time slots; each
                       agent participates in two random meetings; per meeting
                       pair a travel time is drawn from U{6..10}; if the slot
                       difference is smaller than the travel time, the cost is
                       the number of overbooked (shared) agents

All problems additionally receive tiny random unary value preferences
(uniform [0, 1e-2)) for tie breaking, as done in all Max-sum versions in the
AIJ paper (following Farinelli et al. 2008). Without them, perfectly
symmetric problems (graph coloring in particular) make min-sum degenerate.
Their total mass (<= 50 * 1e-2 = 0.5) stays below the smallest structural cost
gap (1 for the random/meeting benchmarks, 10 for coloring), so it cannot change
which assignment is optimal.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np

from propflow import FGBuilder
from propflow.bp.factor_graph import FactorGraph
from propflow.core.agents import FactorAgent, VariableAgent

NUM_AGENTS = 50
PREF_SCALE = 1e-2
RANDOM_SPARSE_DENSITY = 0.1
RANDOM_TERNARY_DENSITY = 2 * RANDOM_SPARSE_DENSITY / (NUM_AGENTS - 2)

SCALE_FREE_INITIAL_AGENTS = 7
SCALE_FREE_ATTACH = 3

MS_MEETINGS = 20
MS_TIME_SLOTS = 20
MS_AGENTS = 90
MS_TRAVEL_LOW, MS_TRAVEL_HIGH = 6, 10  # inclusive


class FixedCostTable:
    """picklable cost-table factory that always returns a fixed table."""

    def __init__(self, ct: np.ndarray) -> None:
        self.ct = np.array(ct, dtype=float, copy=True)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.ct.copy()


def create_coloring_table(n: int, domain: int, cost: float = 10.0) -> np.ndarray:
    """not-equal constraint: equal assignment costs `cost`, anything else 0."""
    idx = np.indices((domain,) * n)
    equal = np.all(idx == idx[0], axis=0)
    return np.where(equal, float(cost), 0.0)


def _with_tiebreak_prefs(fg: FactorGraph, rng: np.random.Generator) -> FactorGraph:
    """add tiny random unary value preferences for tie breaking."""
    unary = {v.name: rng.uniform(0.0, PREF_SCALE, size=v.domain) for v in fg.variables}
    return FGBuilder.build_with_unary_costs(fg, unary)


def _random_uniform(seed: int, density: float) -> FactorGraph:
    # cost tables are drawn from the legacy global rng inside FGBuilder
    np.random.seed(seed)
    fg = FGBuilder.build_random_graph(
        num_vars=NUM_AGENTS,
        domain_size=10,
        ct_factory="random_int",
        ct_params={"low": 100, "high": 200},
        density=density,
        seed=seed,
    )
    return _with_tiebreak_prefs(fg, np.random.default_rng(seed))


def build_random_sparse(seed: int) -> FactorGraph:
    return _random_uniform(seed, density=RANDOM_SPARSE_DENSITY)


def build_random_dense(seed: int) -> FactorGraph:
    return _random_uniform(seed, density=0.6)


def _add_primal_edges(primal: nx.Graph, triple: Tuple[int, int, int]) -> None:
    """Add the pairwise primal edges induced by one ternary factor."""
    primal.add_edges_from(combinations(triple, 2))


def _sample_connected_triples(
    num_agents: int, density: float, rng: np.random.Generator
) -> List[Tuple[int, int, int]]:
    """Sample ternary factors and force their induced primal graph connected."""
    triples = {
        triple
        for triple in combinations(range(num_agents), 3)
        if rng.random() < density
    }

    primal = nx.Graph()
    primal.add_nodes_from(range(num_agents))
    for triple in triples:
        _add_primal_edges(primal, triple)

    while not nx.is_connected(primal):
        components = [tuple(comp) for comp in nx.connected_components(primal)]
        comp_a, comp_b = components[0], components[1]
        u = int(rng.choice(comp_a))
        v = int(rng.choice(comp_b))
        candidates = [idx for idx in range(num_agents) if idx not in {u, v}]
        w = int(rng.choice(candidates))
        triple = tuple(sorted((u, v, w)))
        triples.add(triple)
        _add_primal_edges(primal, triple)

    return sorted(triples)


def build_random_ternary(seed: int) -> FactorGraph:
    """50-agent random benchmark with true arity-3 cost-table constraints."""
    rng = np.random.default_rng(seed)
    variables = [VariableAgent(name=f"x{i + 1}", domain=10) for i in range(NUM_AGENTS)]

    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for triple in _sample_connected_triples(NUM_AGENTS, RANDOM_TERNARY_DENSITY, rng):
        ct = rng.integers(100, 200, size=(10, 10, 10)).astype(float)
        factor = FactorAgent(
            name=f"f{triple[0] + 1}_{triple[1] + 1}_{triple[2] + 1}",
            domain=10,
            ct_creation_func=FixedCostTable(ct),
            param={},
        )
        edges[factor] = [variables[idx] for idx in triple]

    fg = FactorGraph(variables, list(edges.keys()), edges)
    return _with_tiebreak_prefs(fg, rng)


def build_graph_coloring(seed: int) -> FactorGraph:
    np.random.seed(seed)
    fg = FGBuilder.build_random_graph(
        num_vars=NUM_AGENTS,
        domain_size=3,
        ct_factory=create_coloring_table,
        ct_params={"cost": 10.0},
        density=0.1,
        seed=seed,
    )
    return _with_tiebreak_prefs(fg, np.random.default_rng(seed))


def build_scale_free(seed: int) -> FactorGraph:
    rng = np.random.default_rng(seed)

    initial = nx.gnp_random_graph(
        SCALE_FREE_INITIAL_AGENTS, 0.5, seed=int(rng.integers(2**31))
    )
    # the initial set must be connected ("randomly selected and connected")
    components = list(nx.connected_components(initial))
    for comp_a, comp_b in zip(components, components[1:]):
        initial.add_edge(min(comp_a), min(comp_b))

    g = nx.barabasi_albert_graph(
        NUM_AGENTS,
        SCALE_FREE_ATTACH,
        seed=int(rng.integers(2**31)),
        initial_graph=initial,
    )

    variables = [VariableAgent(name=f"x{i + 1}", domain=10) for i in range(NUM_AGENTS)]
    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for u, v in sorted((min(e), max(e)) for e in g.edges()):
        ct = rng.integers(100, 200, size=(10, 10)).astype(float)
        factor = FactorAgent(
            name=f"f{u + 1}_{v + 1}",
            domain=10,
            ct_creation_func=FixedCostTable(ct),
            param={},
        )
        edges[factor] = [variables[u], variables[v]]

    fg = FactorGraph(variables, list(edges.keys()), edges)
    return _with_tiebreak_prefs(fg, rng)


def build_meeting_scheduling(seed: int) -> FactorGraph:
    # EAV (events-as-variables) formulation: each meeting is a single shared
    # variable whose value is its time slot, with a binary constraint between
    # meetings that share agents (not the PEAV private-copy-per-agent model)
    rng = np.random.default_rng(seed)

    # each agent participates in two random meetings; shared[i, j] counts the
    # agents participating in both meetings i and j (the overbooked agents)
    for _ in range(1000):
        shared = np.zeros((MS_MEETINGS, MS_MEETINGS), dtype=int)
        for _ in range(MS_AGENTS):
            i, j = rng.choice(MS_MEETINGS, size=2, replace=False)
            shared[min(i, j), max(i, j)] += 1
        constraint_graph = nx.Graph()
        constraint_graph.add_nodes_from(range(MS_MEETINGS))
        constraint_graph.add_edges_from(
            (int(i), int(j)) for i, j in np.argwhere(shared > 0)
        )
        if nx.is_connected(constraint_graph):
            break
    else:
        raise RuntimeError("failed to sample a connected meeting graph")

    variables = [
        VariableAgent(name=f"x{i + 1}", domain=MS_TIME_SLOTS)
        for i in range(MS_MEETINGS)
    ]
    slots = np.arange(MS_TIME_SLOTS)
    slot_diff = np.abs(np.subtract.outer(slots, slots))

    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for i, j in sorted(map(tuple, np.argwhere(shared > 0))):
        travel = int(rng.integers(MS_TRAVEL_LOW, MS_TRAVEL_HIGH + 1))
        ct = np.where(slot_diff < travel, float(shared[i, j]), 0.0)
        factor = FactorAgent(
            name=f"f{i + 1}_{j + 1}",
            domain=MS_TIME_SLOTS,
            ct_creation_func=FixedCostTable(ct),
            param={},
        )
        edges[factor] = [variables[i], variables[j]]

    fg = FactorGraph(variables, list(edges.keys()), edges)
    return _with_tiebreak_prefs(fg, rng)


def capture_original(
    fg: FactorGraph,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, np.ndarray]]:
    """names + cost tables of the as-built problem, before any engine mutates it.

    Used to score candidate assignments (merges, branch and bound) against the
    original, pre-split cost tables.
    """
    var_names = [v.name for v in fg.variables]
    factor_vars = {f.name: [v.name for v in vs] for f, vs in fg.edges.items()}
    tables = {
        f.name: np.array(f.cost_table, dtype=float, copy=True) for f in fg.factors
    }
    return var_names, factor_vars, tables


BENCHMARKS: Dict[str, Callable[[int], FactorGraph]] = {
    "random_sparse": build_random_sparse,
    "random_dense": build_random_dense,
    "random_ternary": build_random_ternary,
    "graph_coloring": build_graph_coloring,
    "scale_free": build_scale_free,
    "meeting_scheduling": build_meeting_scheduling,
}
