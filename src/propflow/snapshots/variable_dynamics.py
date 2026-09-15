"""Finite-window verdicts and additive cavity diagnostics from snapshots."""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx
import numpy as np

from .analyzer import SnapshotAnalyzer


def observed_period(values: np.ndarray, maximum: int = 64) -> int | None:
    """Find an exact repeated tail, requiring at least four full periods."""
    for period in range(1, min(maximum, len(values) // 4) + 1):
        if np.array_equal(values[period:], values[:-period]):
            return period
    return None


class VariableDynamicsAnalyzer(SnapshotAnalyzer):
    """Separate decoded verdict stability from gauge-invariant belief motion.

    Snapshots must have consecutive steps and explicit beliefs and assignments.
    Cavity diagnostics additionally require every incoming R for the variable.
    Values are array positions, following the snapshot's domain order.
    """

    def verdicts(self, window: int = 300, tolerance: float = 1e-6) -> list[dict]:
        """Describe the final window; a fixed tail is not a convergence proof."""
        if not 2 <= window <= len(self._snapshots):
            raise ValueError("window must contain at least two available snapshots")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        steps = np.array([s.step for s in self._snapshots])
        if not np.all(np.diff(steps) == 1):
            raise ValueError("consecutive snapshots are required")
        rows = []
        for name, size in self._domain.items():
            x = np.array([s.assignments[name] for s in self._snapshots])
            b = np.array([s.beliefs[name] for s in self._snapshots], dtype=float)
            if b.shape != (len(steps), size) or not np.isfinite(b).all():
                raise ValueError(f"invalid beliefs for {name}")
            if np.any(x < 0) or np.any(x >= size):
                raise ValueError(f"invalid assignments for {name}")
            tail = x[-window:]
            differences = b[-window:] - b[-window:, :1]
            span = float(np.ptp(differences, axis=0).max())
            changed = np.flatnonzero(x[1:] != x[:-1]) + 1
            ordered = np.sort(b[-window:], axis=1)
            rows.append(
                dict(
                    variable=name,
                    unsettled=bool(np.any(tail[1:] != tail[:-1])),
                    tail_switches=int(np.sum(tail[1:] != tail[:-1])),
                    tail_values=np.unique(tail).tolist(),
                    observed_period=observed_period(tail),
                    last_switch_step=int(steps[changed[-1]]) if len(changed) else None,
                    belief_span=span,
                    belief_fixed=span <= tolerance,
                    min_margin=float((ordered[:, 1] - ordered[:, 0]).min()),
                    median_margin=float(np.median(ordered[:, 1] - ordered[:, 0])),
                )
            )
        return rows

    def cavities(self, variable: str, factors: Sequence[str]) -> dict[str, Any]:
        """Return current belief and next-send cavities in a common label gauge.

        Removing one clone retains its sibling; removing both returns the
        external cavity of the original factor. These are *next* Q candidates,
        not the Q already sent in the snapshot's update, and are undamped.
        """
        if not factors or len(set(factors)) != len(factors):
            raise ValueError("provide distinct factors to exclude")
        incoming = []
        beliefs = []
        for snapshot in self._snapshots:
            neighbors = snapshot.N_var[variable]
            if not set(factors) <= set(neighbors):
                raise ValueError("excluded factors must be variable neighbors")
            messages = [snapshot.R[(f, variable)] for f in neighbors]
            b = np.asarray(snapshot.beliefs[variable])
            total = np.sum(messages, axis=0)
            if not np.allclose(total - total[0], b - b[0], atol=1e-6, rtol=1e-10):
                raise ValueError("incoming messages do not reconstruct belief")
            beliefs.append(b - b[0])
            incoming.append([snapshot.R[(f, variable)] for f in factors])
        b = np.array(beliefs)
        r = np.array(incoming)
        r -= r[..., :1]
        return dict(
            steps=np.array([s.step for s in self._snapshots]),
            belief=b,
            incoming=r,
            clone_cavities=b[:, None, :] - r,
            external_cavity=b - r.sum(axis=1),
        )


def component_profile(graph: nx.Graph, unsettled: set) -> dict:
    """Measure original variable topology and its unsettled induced components."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("an undirected simple variable graph is required")
    if not unsettled <= set(graph):
        raise ValueError("unsettled nodes must occur in graph")
    centralities = {
        "betweenness": nx.betweenness_centrality(graph, normalized=True),
        "closeness": nx.closeness_centrality(graph),
        "clustering": nx.clustering(graph),
        "triangles": nx.triangles(graph),
        "core": nx.core_number(graph),
        "pagerank": nx.pagerank(graph),
    }
    articulation = set(nx.articulation_points(graph))
    communities = list(nx.community.greedy_modularity_communities(graph))
    membership = {n: j for j, c in enumerate(communities) for n in c}
    induced = graph.subgraph(unsettled)
    components = sorted(
        nx.connected_components(induced), key=lambda c: (-len(c), min(c))
    )
    component_ids = {n: j + 1 for j, c in enumerate(components) for n in c}
    nodes = {}
    for n in graph:
        neighbors = set(graph[n])
        by_community = np.bincount([membership[v] for v in neighbors])
        nodes[n] = dict(
            degree=graph.degree(n),
            unsettled_neighbors=len(neighbors & unsettled),
            settled_neighbors=len(neighbors - unsettled),
            articulation=n in articulation,
            community=membership[n],
            participation=float(
                1 - np.sum((by_community / max(len(neighbors), 1)) ** 2)
            ),
            component=component_ids.get(n),
            **{key: values[n] for key, values in centralities.items()},
        )
    summaries = []
    for index, c in enumerate(components, 1):
        h = graph.subgraph(c)
        cut = list(nx.edge_boundary(graph, c))
        volume = sum(dict(graph.degree(c)).values())
        other_volume = 2 * graph.number_of_edges() - volume
        denominator = min(volume, other_volume)
        summaries.append(
            dict(
                component=index,
                variables=sorted(c),
                edges=h.number_of_edges(),
                density=nx.density(h),
                diameter=nx.diameter(h),
                cycle_rank=h.number_of_edges() - len(c) + 1,
                cut_edges=len(cut),
                boundary_variables=sorted({v for _, v in cut}),
                conductance=len(cut) / denominator if denominator else None,
                articulation_points=sorted(nx.articulation_points(h)),
                bridges=[list(e) for e in nx.bridges(h)],
                biconnected_sizes=sorted(
                    map(len, nx.biconnected_components(h)), reverse=True
                ),
            )
        )
    return dict(nodes=nodes, components=summaries)


def cost_table_profile(table: np.ndarray) -> dict:
    """Describe cost scale and interaction after removing row/column effects."""
    c = np.asarray(table, dtype=float)
    if c.ndim != 2 or min(c.shape) < 2 or not np.isfinite(c).all():
        raise ValueError("a finite pairwise table with at least two labels is required")
    interaction = c - c.mean(0, keepdims=True) - c.mean(1, keepdims=True) + c.mean()
    singular = np.linalg.svd(interaction, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    return dict(
        minimum=float(c.min()),
        maximum=float(c.max()),
        std=float(c.std()),
        interaction_rms=float(np.sqrt(np.mean(interaction**2))),
        interaction_fraction=(
            float(np.var(interaction) / np.var(c)) if np.var(c) else 0.0
        ),
        effective_rank=float(total**2 / np.sum(energy**2)) if total else 0.0,
        row_minimizer_diversity=int(len(np.unique(c.argmin(axis=1)))),
        column_minimizer_diversity=int(len(np.unique(c.argmin(axis=0)))),
        row_margin_mean=float(np.diff(np.sort(c, axis=1)[:, :2], axis=1).mean()),
        column_margin_mean=float(np.diff(np.sort(c, axis=0)[:2], axis=0).mean()),
    )
