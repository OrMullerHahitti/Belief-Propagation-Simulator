"""Selected-entry and diagonal diagnostics for factor updates."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


MAIN_DIAGONAL = {(0, 0), (1, 1)}
ANTI_DIAGONAL = {(0, 1), (1, 0)}


def binary_diagonal_orientation(
    selected_entries: list[list[int]] | list[tuple[int, int]],
) -> str:
    """Classify selected binary entries as main, anti, mixed, or unknown."""

    coords = {
        tuple(int(v) for v in entry) for entry in selected_entries if len(entry) == 2
    }
    if not coords:
        return "unknown"
    if coords.issubset(MAIN_DIAGONAL):
        return "main"
    if coords.issubset(ANTI_DIAGONAL):
        return "anti"
    if coords & MAIN_DIAGONAL or coords & ANTI_DIAGONAL:
        return "mixed"
    return "unknown"


def generalized_selected_entry_signature(
    selected_entries: list[list[int]],
    row_minimizer_map: dict[str, list[int]] | None = None,
    column_minimizer_map: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Return non-binary selected-entry metadata without diagonal language."""

    normalized = [tuple(int(v) for v in entry) for entry in selected_entries]
    transitions = Counter(normalized)
    return {
        "selected_cell_coordinates": [list(entry) for entry in normalized],
        "row_minimizer_map": row_minimizer_map or {},
        "column_minimizer_map": column_minimizer_map or {},
        "active_minimizer_transition_signature": {
            str(list(entry)): int(count) for entry, count in sorted(transitions.items())
        },
    }


def analyze_trace_diagonals(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute selected diagonal orientation per factor and iteration."""

    rows: list[dict[str, Any]] = []
    for entry in trace:
        factor_orientations: dict[str, str] = {}
        factor_signatures: dict[str, Any] = {}
        for factor_name, by_recipient in entry.get("selected_minimizers", {}).items():
            selected_entries: list[list[int]] = []
            row_map: dict[str, list[int]] = {}
            col_map: dict[str, list[int]] = {}
            for metadata in by_recipient.values():
                selected_entries.extend(metadata.get("selected_entries", []))
                row_map = metadata.get("row_minimizer_map", row_map)
                col_map = metadata.get("column_minimizer_map", col_map)
            domain_size = _infer_domain_size(selected_entries)
            if domain_size == 2:
                factor_orientations[factor_name] = binary_diagonal_orientation(
                    selected_entries
                )
            else:
                factor_orientations[factor_name] = "unknown"
                factor_signatures[factor_name] = generalized_selected_entry_signature(
                    selected_entries,
                    row_map,
                    col_map,
                )

        rows.append(
            {
                "iteration": entry["iteration"],
                "parity": entry.get("parity"),
                "factor_orientations": factor_orientations,
                "opposing_diagonals": _opposing_diagonal_pairs(factor_orientations),
                "non_binary_signatures": factor_signatures,
            }
        )
    return rows


def summarize_diagonal_diagnostics(
    diagonal_trace: list[dict[str, Any]], tail_start: int | None = None
) -> dict[str, Any]:
    """Summarize parity and tail changes in selected diagonal orientation."""

    by_factor: dict[str, list[tuple[int, str, str]]] = {}
    for row in diagonal_trace:
        for factor, orientation in row["factor_orientations"].items():
            by_factor.setdefault(factor, []).append(
                (int(row["iteration"]), str(row.get("parity")), orientation)
            )

    summary: dict[str, Any] = {"per_factor": {}, "opposite_pairs_seen": False}
    for row in diagonal_trace:
        if row["opposing_diagonals"]:
            summary["opposite_pairs_seen"] = True

    for factor, values in by_factor.items():
        even = {orientation for _, parity, orientation in values if parity == "even"}
        odd = {orientation for _, parity, orientation in values if parity == "odd"}
        before = {
            orientation
            for iteration, _, orientation in values
            if tail_start is not None and iteration < tail_start
        }
        after = {
            orientation
            for iteration, _, orientation in values
            if tail_start is not None and iteration >= tail_start
        }
        summary["per_factor"][factor] = {
            "even_orientations": sorted(even),
            "odd_orientations": sorted(odd),
            "differs_between_even_and_odd": bool(even and odd and even != odd),
            "changes_after_tail": bool(
                tail_start is not None and before and after and before != after
            ),
        }
    return summary


def _infer_domain_size(selected_entries: list[list[int]]) -> int | None:
    if not selected_entries:
        return None
    max_index = max(max(entry) for entry in selected_entries if entry)
    return int(max_index) + 1


def _opposing_diagonal_pairs(orientations: dict[str, str]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    items = sorted(orientations.items())
    for idx, (left_factor, left_orientation) in enumerate(items):
        for right_factor, right_orientation in items[idx + 1 :]:
            if {left_orientation, right_orientation} == {"main", "anti"}:
                pairs.append(
                    {
                        "left_factor": left_factor,
                        "left_orientation": left_orientation,
                        "right_factor": right_factor,
                        "right_orientation": right_orientation,
                    }
                )
    return pairs


def factor_table_orientation(cost_table: np.ndarray) -> str:
    """Classify the global-minimum cells of a binary factor cost table."""

    table = np.asarray(cost_table, dtype=float)
    if table.shape != (2, 2):
        return "unknown"
    selected = np.argwhere(np.isclose(table, np.min(table))).astype(int).tolist()
    return binary_diagonal_orientation(selected)
