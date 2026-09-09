"""Save exact problem inputs and measure the original variable graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np

from propflow import FGBuilder, FactorAgent, FactorGraph, VariableAgent


@dataclass(frozen=True)
class Settings:
    """Paired experiment settings; density is the original edge probability."""

    nodes: int = 20
    density: float = 0.5
    domain: int = 10
    graph_seed: int = 0
    model_seed: int = 0
    max_iterations: int = 2000
    stable_window: int = 100
    tolerance: float = 0.001
    update_interval: int = 20

    def __post_init__(self) -> None:
        if self.nodes < 2 or self.domain < 2:
            raise ValueError("nodes and domain must both be at least 2")
        if not 0 < self.density <= 1:
            raise ValueError("density must be in (0, 1]")
        if self.max_iterations < 1 or self.stable_window < 2:
            raise ValueError("max_iterations >= 1 and stable_window >= 2 required")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if self.update_interval < 2:
            raise ValueError("update_interval must be at least 2")
        if self.stable_window <= self.update_interval:
            raise ValueError("stable_window must span an optimizer update")
        if min(self.graph_seed, self.model_seed) < 0:
            raise ValueError("seeds must be nonnegative")


def canonical_json(value: Any) -> str:
    """Encode saved inputs without non-finite values or unstable key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    """Return the SHA-256 of canonical input content."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def create_problem(settings: Settings) -> dict:
    """Generate one connected graph and persist tables, axes, and structure."""
    state = np.random.get_state()
    try:
        np.random.seed(settings.graph_seed)
        graph = FGBuilder.build_random_graph(
            num_vars=settings.nodes,
            domain_size=settings.domain,
            ct_factory="random_int",
            ct_params={"low": 100, "high": 200},
            density=settings.density,
            seed=settings.graph_seed,
        )
        rng = np.random.default_rng(settings.graph_seed)
        unary = {v.name: rng.uniform(0, 0.01, settings.domain) for v in graph.variables}
        graph = FGBuilder.build_with_unary_costs(graph, unary)
    finally:
        np.random.set_state(state)
    factors = []
    primal = nx.Graph()
    names = [v.name for v in graph.variables]
    primal.add_nodes_from(names)
    for factor, variables in graph.edges.items():
        axes = [v.name for v in variables]
        factors.append(
            {"id": factor.name, "variables": axes, "table": factor.cost_table.tolist()}
        )
        if len(axes) == 2:
            primal.add_edge(*axes, factor=factor.name)
    if not nx.is_connected(primal):
        raise ValueError("the experiment requires a connected variable graph")
    positions = nx.spring_layout(primal, seed=settings.graph_seed)
    betweenness = nx.betweenness_centrality(primal, normalized=True, weight=None)
    closeness = nx.closeness_centrality(primal)
    clustering = nx.clustering(primal, weight=None)
    return {
        "schema_version": 1,
        "generation": {
            key: asdict(settings)[key]
            for key in ("nodes", "density", "domain", "graph_seed")
        },
        "costs": "integer U[100,200); unary U[0,0.01)",
        "realized_density": nx.density(primal),
        "nodes": [
            {
                "id": name,
                "degree": primal.degree(name),
                "betweenness": betweenness[name],
                "closeness": closeness[name],
                "clustering": clustering[name],
                "neighbors": [n for n in names if primal.has_edge(name, n)],
                "position": positions[name].tolist(),
            }
            for name in names
        ],
        "factors": factors,
    }


def restore_problem(problem: dict) -> FactorGraph:
    """Restore exact tables without drawing new costs or changing axis order."""
    domain = problem["generation"]["domain"]
    variables = [VariableAgent(n["id"], domain=domain) for n in problem["nodes"]]
    by_name = {v.name: v for v in variables}
    factors = []
    edges = {}
    for entry in problem["factors"]:
        table = np.asarray(entry["table"], dtype=np.float64)
        axes = entry["variables"]
        if table.shape != (domain,) * len(axes) or not np.isfinite(table).all():
            raise ValueError(f"invalid cost table for {entry['id']}")
        factor = FactorAgent.create_from_cost_table(entry["id"], table)
        factors.append(factor)
        edges[factor] = [by_name[n] for n in axes]
    if len({f.name for f in factors}) != len(factors):
        raise ValueError("factor names must be unique")
    return FGBuilder.build_from_edges(variables, factors, edges)
