"""Cost-table-only route diagnostics for small factor graphs."""

from __future__ import annotations

from itertools import product
from typing import Any

import networkx as nx
import numpy as np

from propflow import FactorGraph


def analyze_routes_from_graph(
    graph: FactorGraph,
    *,
    simulation_classification: str | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Analyze candidate local-minimum routes for a small single-cycle graph.

    Exact route classification is only attempted for pairwise single-cycle
    graphs. Symmetric split chain/non-cycle graphs return a skip warning and
    local factor diagnostics so reports can still reason about candidate
    signatures.
    """

    cycle_info = _single_cycle_info(graph)
    if cycle_info is None:
        return {
            "status": "skipped",
            "warning": "Exact route analysis is only implemented for pairwise single-cycle graphs.",
            "local_factor_diagnostics": _local_factor_diagnostics(graph, tolerance),
            "simulation_classification": simulation_classification,
        }

    factors = cycle_info["factors"]
    local_choices = [_minimal_entries(factor, tolerance) for factor in factors]
    candidates: list[dict[str, Any]] = []
    for route_index, entries in enumerate(product(*local_choices)):
        induced: dict[str, set[int]] = {}
        total_cost = 0.0
        repeated_entries: list[dict[str, Any]] = []
        for factor, entry in zip(factors, entries):
            table = np.asarray(factor.cost_table, dtype=float)
            total_cost += float(table[entry])
            labels = _factor_labels(factor)
            repeated_entries.append(
                {
                    "factor": factor.name,
                    "entry": list(entry),
                    "cost": float(table[entry]),
                }
            )
            for axis, variable_name in enumerate(labels):
                induced.setdefault(variable_name, set()).add(int(entry[axis]))
        consistent = all(len(values) == 1 for values in induced.values())
        candidates.append(
            {
                "route_id": f"route_{route_index}",
                "entries": repeated_entries,
                "total_route_cost": total_cost,
                "normalized_route_cost": total_cost / max(1, len(factors)),
                "induced_variable_assignments": {
                    name: sorted(values) for name, values in sorted(induced.items())
                },
                "consistent": consistent,
                "inconsistent": not consistent,
                "repeated_entries_M": repeated_entries,
                "tail_prefix": [],
            }
        )

    candidates.sort(key=lambda item: (item["total_route_cost"], item["route_id"]))
    return {
        "status": "ok",
        "num_candidates": len(candidates),
        "candidate_routes": candidates,
        "best_route": candidates[0] if candidates else None,
        "binary_inequality_summary": _binary_inequality_summary(candidates),
        "simulation_classification": simulation_classification,
        "prediction_matches_simulation": _compare_prediction(
            candidates, simulation_classification
        ),
    }


def _single_cycle_info(graph: FactorGraph) -> dict[str, Any] | None:
    primal = nx.Graph()
    edge_to_factor = {}
    for variable in graph.variables:
        primal.add_node(variable.name)
    for factor in graph.factors:
        labels = _factor_labels(factor)
        if len(labels) != 2:
            return None
        edge = tuple(sorted(labels))
        primal.add_edge(*edge)
        edge_to_factor[edge] = factor
    if not nx.is_connected(primal):
        return None
    if primal.number_of_edges() != primal.number_of_nodes():
        return None
    if any(degree != 2 for _, degree in primal.degree()):
        return None
    factors = [edge_to_factor[tuple(sorted(edge))] for edge in nx.find_cycle(primal)]
    return {"primal": primal, "factors": factors}


def _factor_labels(factor: Any) -> list[str]:
    return [
        name
        for name, _ in sorted(
            factor.connection_number.items(), key=lambda item: item[1]
        )
    ]


def _minimal_entries(factor: Any, tolerance: float) -> list[tuple[int, ...]]:
    table = np.asarray(factor.cost_table, dtype=float)
    min_value = float(np.min(table))
    return [
        tuple(int(v) for v in entry)
        for entry in np.argwhere(np.isclose(table, min_value, atol=tolerance))
    ]


def _local_factor_diagnostics(
    graph: FactorGraph, tolerance: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in sorted(graph.factors, key=lambda item: item.name):
        table = np.asarray(factor.cost_table, dtype=float)
        entries = _minimal_entries(factor, tolerance)
        rows.append(
            {
                "factor": factor.name,
                "variables": _factor_labels(factor),
                "minimal_entries": [list(entry) for entry in entries],
                "minimum_cost": float(np.min(table)),
                "mean_cost": float(np.mean(table)),
            }
        )
    return rows


def _binary_inequality_summary(candidates: list[dict[str, Any]]) -> str:
    if len(candidates) < 2:
        return ""
    left = candidates[0]
    right = candidates[1]
    left_terms = " + ".join(
        f"{entry['factor']}[{','.join(str(v) for v in entry['entry'])}]"
        for entry in left["entries"]
    )
    right_terms = " + ".join(
        f"{entry['factor']}[{','.join(str(v) for v in entry['entry'])}]"
        for entry in right["entries"]
    )
    return f"{left['route_id']} is preferred when {left_terms} < {right_terms}."


def _compare_prediction(
    candidates: list[dict[str, Any]], simulation_classification: str | None
) -> bool | None:
    if not candidates or simulation_classification is None:
        return None
    best_inconsistent = bool(candidates[0]["inconsistent"])
    if "oscillation" in simulation_classification:
        return best_inconsistent
    if simulation_classification == "converged":
        return not best_inconsistent
    return None
