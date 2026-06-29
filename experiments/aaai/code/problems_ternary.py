"""Ternary (true arity-3) analogs of the AAAI benchmarks.

The binary AAAI suite (``problems.py``) studies how min-sum, damping, splitting
and the split-merge variants behave on pairwise DCOPs. This module builds a
**parallel suite of arity-3 problems** so the same questions can be asked of
higher-arity factor graphs. The goal is *not* a binary-vs-ternary comparison but
to check whether the phenomena seen in binary (damping helps, the 0.5 split
helps, mid-run split timing, the MGM/optimal split merges, ...) reappear when
every constraint is genuinely ternary.

Every problem keeps the conventions of ``problems.py``: domain sizes, integer
cost range U[100, 200), tiny U[0, 1e-2) unary tie-break preferences, and a
force-connected primal graph (engines require a connected graph).

Benchmark map (binary name -> ternary analog and how faithful it is):

- ``random_sparse``      -> ``random_sparse_ternary``   (== ``build_random_ternary``;
                            arity-3 random factors at p3 = 2*p1/(n-2), the exact
                            expected-degree match of the binary sparse benchmark)
- ``random_dense``       -> ``random_dense_ternary``     (same construction at the
                            dense degree, p3 = 2*0.6/(n-2); heaviest benchmark)
- ``graph_coloring``     -> ``graph_coloring_ternary``   (constructed: each ternary
                            factor is the *triangle of pairwise not-equals* over its
                            three variables, cost = 10 x (#equal pairs). No canonical
                            ternary coloring exists; this is the natural graded
                            generalization of the binary not-equal constraint)
- ``scale_free``         -> ``scale_free_ternary``       (constructed: a preferential-
                            attachment *hypergraph* -- each new agent forms one
                            ternary hyperedge with two existing agents chosen with
                            probability proportional to their current hyperdegree,
                            the arity-3 analog of Barabasi-Albert)
- ``meeting_scheduling`` -> ``meeting_scheduling_ternary`` (each agent participates in
                            *three* meetings instead of two, inducing a genuine
                            ternary "no two of my meetings too close" constraint)

See README.md ("Ternary suite") for which families are faithful analogs vs.
constructed generalizations, and which won't fit exact search.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np

from propflow.bp.factor_graph import FactorGraph
from propflow.core.agents import FactorAgent, VariableAgent

try:  # bare import (sys.path-mutated runner) vs. package import (tests/notebooks)
    from problems import (
        MS_AGENTS,
        MS_MEETINGS,
        MS_TIME_SLOTS,
        MS_TRAVEL_HIGH,
        MS_TRAVEL_LOW,
        NUM_AGENTS,
        FixedCostTable,
        _sample_connected_triples,
        _with_tiebreak_prefs,
        build_random_ternary,
    )
except ModuleNotFoundError:
    from .problems import (
        MS_AGENTS,
        MS_MEETINGS,
        MS_TIME_SLOTS,
        MS_TRAVEL_HIGH,
        MS_TRAVEL_LOW,
        NUM_AGENTS,
        FixedCostTable,
        _sample_connected_triples,
        _with_tiebreak_prefs,
        build_random_ternary,
    )

# expected-degree-matched ternary densities: a variable's expected number of
# incident triples is p3 * C(n-1, 2); setting it equal to the binary expected
# degree p1 * (n-1) gives p3 = 2 * p1 / (n - 2).
RANDOM_DENSE_TERNARY_DENSITY = 2 * 0.6 / (NUM_AGENTS - 2)
COLORING_TERNARY_DENSITY = 2 * 0.1 / (NUM_AGENTS - 2)
COLORING_DOMAIN = 3
COLORING_COST = 10.0

SCALE_FREE_TERNARY_SEED_NODES = 3  # the initial fully-connected triple


def _random_ternary_graph(seed: int, density: float) -> FactorGraph:
    """random arity-3 benchmark at an arbitrary triple density (domain 10).

    Generalizes ``problems.build_random_ternary`` (which is fixed at the sparse
    density) so the dense analog can reuse the exact same construction.
    """
    rng = np.random.default_rng(seed)
    variables = [VariableAgent(name=f"x{i + 1}", domain=10) for i in range(NUM_AGENTS)]

    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for triple in _sample_connected_triples(NUM_AGENTS, density, rng):
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


def build_random_sparse_ternary(seed: int) -> FactorGraph:
    """Ternary analog of ``random_sparse``: the existing arity-3 random benchmark."""
    return build_random_ternary(seed)


def build_random_dense_ternary(seed: int) -> FactorGraph:
    """Ternary analog of ``random_dense``: random arity-3 factors at the dense degree."""
    return _random_ternary_graph(seed, density=RANDOM_DENSE_TERNARY_DENSITY)


def create_ternary_coloring_table(domain: int, cost: float = COLORING_COST) -> np.ndarray:
    """Ternary not-equal constraint: cost x (number of equal pairs in the triple).

    This is exactly the three pairwise not-equal (graph-coloring) constraints on
    the triple's variables folded into one arity-3 factor: all three equal ->
    3*cost, exactly one equal pair -> cost, all distinct -> 0. The graded cost
    keeps a gradient for min-sum (a pure all-equal indicator would be flat
    almost everywhere).
    """
    idx = np.indices((domain,) * 3)
    equal_pairs = (
        (idx[0] == idx[1]).astype(float)
        + (idx[0] == idx[2]).astype(float)
        + (idx[1] == idx[2]).astype(float)
    )
    return cost * equal_pairs


def build_graph_coloring_ternary(seed: int) -> FactorGraph:
    """Ternary analog of ``graph_coloring`` (domain 3, ternary not-equal factors)."""
    rng = np.random.default_rng(seed)
    variables = [
        VariableAgent(name=f"x{i + 1}", domain=COLORING_DOMAIN)
        for i in range(NUM_AGENTS)
    ]
    table = create_ternary_coloring_table(COLORING_DOMAIN, COLORING_COST)

    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for triple in _sample_connected_triples(NUM_AGENTS, COLORING_TERNARY_DENSITY, rng):
        factor = FactorAgent(
            name=f"f{triple[0] + 1}_{triple[1] + 1}_{triple[2] + 1}",
            domain=COLORING_DOMAIN,
            ct_creation_func=FixedCostTable(table),
            param={},
        )
        edges[factor] = [variables[idx] for idx in triple]

    fg = FactorGraph(variables, list(edges.keys()), edges)
    return _with_tiebreak_prefs(fg, rng)


def _scale_free_hyperedges(
    num_agents: int, rng: np.random.Generator
) -> List[Tuple[int, int, int]]:
    """Preferential-attachment hypergraph: the arity-3 analog of Barabasi-Albert.

    Seed with one triple on the first three agents; every later agent forms one
    new ternary hyperedge with two existing agents drawn without replacement
    with probability proportional to their current hyperdegree (+1 smoothing so
    the seed agents are reachable). One hyperedge per new agent adds three primal
    edges, matching the per-node edge growth of the binary BA benchmark (m = 3).
    The primal graph is connected by construction.
    """
    triples: List[Tuple[int, int, int]] = [(0, 1, 2)]
    hyperdeg = np.zeros(num_agents, dtype=float)
    hyperdeg[:SCALE_FREE_TERNARY_SEED_NODES] = 1.0

    for v in range(SCALE_FREE_TERNARY_SEED_NODES, num_agents):
        existing = np.arange(v)
        weights = hyperdeg[existing] + 1.0
        partners = rng.choice(
            existing, size=2, replace=False, p=weights / weights.sum()
        )
        triple = tuple(sorted((int(partners[0]), int(partners[1]), v)))
        triples.append(triple)
        for node in triple:
            hyperdeg[node] += 1.0

    return triples


def build_scale_free_ternary(seed: int) -> FactorGraph:
    """Ternary analog of ``scale_free`` (preferential-attachment hypergraph, domain 10)."""
    rng = np.random.default_rng(seed)
    variables = [VariableAgent(name=f"x{i + 1}", domain=10) for i in range(NUM_AGENTS)]

    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for triple in _scale_free_hyperedges(NUM_AGENTS, rng):
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


def _ternary_meeting_table(
    triple: Tuple[int, int, int],
    count: int,
    travel: Dict[Tuple[int, int], int],
    slot_diff: np.ndarray,
) -> np.ndarray:
    """arity-3 meeting cost: count x (number of too-close meeting pairs in the triple).

    For each of the triple's three meeting pairs, a pair scheduled closer than
    its travel time costs ``count`` (the agents shared by all three meetings are
    overbooked on that pair). Broadcasting the per-pair SxS penalty over the
    third axis assembles the SxSxS table.
    """
    i, j, k = triple
    size = slot_diff.shape[0]
    table = np.zeros((size, size, size), dtype=float)
    pen_ij = (slot_diff < travel[(i, j)]).astype(float)
    pen_ik = (slot_diff < travel[(i, k)]).astype(float)
    pen_jk = (slot_diff < travel[(j, k)]).astype(float)
    table += count * pen_ij[:, :, None]
    table += count * pen_ik[:, None, :]
    table += count * pen_jk[None, :, :]
    return table


def build_meeting_scheduling_ternary(seed: int) -> FactorGraph:
    """Ternary analog of ``meeting_scheduling``: each agent is in *three* meetings.

    EAV formulation (one variable per meeting, value = time slot). Each of the
    90 agents participates in a random triple of meetings, inducing a genuine
    ternary constraint among them; agents sharing the same triple add up. Travel
    times are drawn once per meeting pair (as in the binary benchmark).
    """
    rng = np.random.default_rng(seed)

    for _ in range(1000):
        counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        for _ in range(MS_AGENTS):
            trio = tuple(
                sorted(int(m) for m in rng.choice(MS_MEETINGS, size=3, replace=False))
            )
            counts[trio] += 1
        primal = nx.Graph()
        primal.add_nodes_from(range(MS_MEETINGS))
        for trio in counts:
            primal.add_edges_from(combinations(trio, 2))
        if nx.is_connected(primal):
            break
    else:
        raise RuntimeError("failed to sample a connected ternary meeting graph")

    travel: Dict[Tuple[int, int], int] = {}
    for trio in counts:
        for pair in combinations(trio, 2):
            if pair not in travel:
                travel[pair] = int(rng.integers(MS_TRAVEL_LOW, MS_TRAVEL_HIGH + 1))

    variables = [
        VariableAgent(name=f"x{i + 1}", domain=MS_TIME_SLOTS)
        for i in range(MS_MEETINGS)
    ]
    slots = np.arange(MS_TIME_SLOTS)
    slot_diff = np.abs(np.subtract.outer(slots, slots))

    edges: Dict[FactorAgent, List[VariableAgent]] = {}
    for trio, count in sorted(counts.items()):
        ct = _ternary_meeting_table(trio, count, travel, slot_diff)
        factor = FactorAgent(
            name=f"f{trio[0] + 1}_{trio[1] + 1}_{trio[2] + 1}",
            domain=MS_TIME_SLOTS,
            ct_creation_func=FixedCostTable(ct),
            param={},
        )
        edges[factor] = [variables[trio[0]], variables[trio[1]], variables[trio[2]]]

    fg = FactorGraph(variables, list(edges.keys()), edges)
    return _with_tiebreak_prefs(fg, rng)


TERNARY_BENCHMARKS: Dict[str, Callable[[int], FactorGraph]] = {
    "random_sparse_ternary": build_random_sparse_ternary,
    "random_dense_ternary": build_random_dense_ternary,
    "graph_coloring_ternary": build_graph_coloring_ternary,
    "scale_free_ternary": build_scale_free_ternary,
    "meeting_scheduling_ternary": build_meeting_scheduling_ternary,
}
