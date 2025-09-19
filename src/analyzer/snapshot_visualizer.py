"""Utilities for visualising argmin trajectories from belief propagation snapshots."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt

SnapshotDict = Dict[str, object]


@dataclass
class _SnapshotRecord:
    step: int
    messages: List[Dict[str, object]]
    assignments: Dict[str, int]
    cost: float | None


class SnapshotVisualizer:
    """Visualise belief minimisers over iterations from BP snapshots."""

    _MAX_AUTO_VARS = 20
    _SMALL_PLOT_THRESHOLD = 8

    def __init__(self, snapshots: Sequence[SnapshotDict]):
        if not snapshots:
            raise ValueError("Snapshots are empty")
        self._records = self._normalise_snapshots(snapshots)
        self._steps = [rec.step for rec in self._records]
        self._variables = self._collect_variables(self._records)

    # ------------------------------- factories ----------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> "SnapshotVisualizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError("Snapshot JSON must contain a list of step dictionaries")
        return cls(data)

    @classmethod
    def from_object(cls, snapshots: List[SnapshotDict]) -> "SnapshotVisualizer":
        return cls(snapshots)

    # ------------------------------ public API ----------------------------
    def variables(self) -> List[str]:
        return sorted(self._variables)

    def argmin_series(self, vars_filter: List[str] | None = None) -> Dict[str, List[int | None]]:
        target_vars = self._select_variables(vars_filter)
        belief_cache = self._compute_belief_argmins(target_vars)
        return {var: belief_cache[var] for var in target_vars}

    def plot_argmin_per_variable(
        self,
        vars_filter: List[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        savepath: str | None = None,
        combined_savepath: str | None = None,
    ) -> None:
        """Plot argmin trajectories per variable and optionally a combined chart.

        One subplot per variable if the selection is small; otherwise the per-variable
        figure collapses into a single multi-line chart. When a `savepath` is supplied
        and more than one variable is plotted, an additional file with suffix
        ``_combined`` is written automatically (unless ``combined_savepath`` overrides
        the target). A combined figure is also rendered when ``combined_savepath`` is
        provided or when ``show`` is true for multiple variables.
        """
        target_vars = self._select_variables(vars_filter, enforce_limit=True)
        series = self._compute_belief_argmins(target_vars)

        steps = self._steps
        if not steps:
            raise ValueError("No steps to plot")

        per_var_fig = None

        if len(target_vars) <= self._SMALL_PLOT_THRESHOLD:
            per_var_fig, axes = plt.subplots(len(target_vars), 1, figsize=figsize or (10, 3 * len(target_vars)))
            if len(target_vars) == 1:
                axes = [axes]
            for ax, var in zip(axes, target_vars):
                self._plot_variable_series(ax, var, steps, series[var])
            plt.tight_layout()
        else:
            per_var_fig, ax = plt.subplots(figsize=figsize or (12, 6))
            for var in target_vars:
                ax.plot(steps, series[var], marker="o", label=var)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Argmin index")
            ax.set_title("Belief argmin trajectories")
            ax.grid(True, alpha=0.3)
            ax.legend()
            self._set_integer_ticks(ax, series)
            plt.tight_layout()

        derived_combined = combined_savepath
        if savepath:
            save_path_obj = Path(savepath)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            per_var_fig.savefig(save_path_obj, dpi=150)
            if derived_combined is None and len(target_vars) > 1:
                derived_combined = str(
                    save_path_obj.with_name(f"{save_path_obj.stem}_combined{save_path_obj.suffix}")
                )

        if derived_combined or (show and len(target_vars) > 1):
            combined_fig, combined_ax = plt.subplots(figsize=figsize or (12, 6))
            for var in target_vars:
                combined_ax.plot(steps, series[var], marker="o", label=var)
            combined_ax.set_xlabel("Iteration")
            combined_ax.set_ylabel("Argmin index")
            combined_ax.set_title("Belief argmin trajectories (combined)")
            combined_ax.grid(True, alpha=0.3)
            combined_ax.legend()
            self._set_integer_ticks(combined_ax, series)
            plt.tight_layout()

            if derived_combined:
                combined_path_obj = Path(derived_combined)
                combined_path_obj.parent.mkdir(parents=True, exist_ok=True)
                combined_fig.savefig(combined_path_obj, dpi=150)

            if show:
                combined_fig.show()
            else:
                plt.close(combined_fig)

        if show and len(target_vars) <= self._SMALL_PLOT_THRESHOLD:
            per_var_fig.show()
        else:
            plt.close(per_var_fig)

    # ------------------------------ internals -----------------------------
    @staticmethod
    def _set_integer_ticks(ax, series: Dict[str, List[int | None]]) -> None:
        values = [v for seq in series.values() for v in seq if v is not None]
        if values:
            ax.set_yticks(sorted(set(values)))

    def _plot_variable_series(self, ax, var: str, steps: Sequence[int], series: Sequence[int | None]) -> None:
        ax.plot(steps, series, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Argmin index")
        ax.set_title(var)
        ax.grid(True, alpha=0.3)
        valid_values = [value for value in series if value is not None]
        if valid_values:
            ax.set_yticks(sorted(set(valid_values)))

    def _select_variables(self, vars_filter: List[str] | None, *, enforce_limit: bool = False) -> List[str]:
        if vars_filter:
            unknown = [var for var in vars_filter if var not in self._variables]
            if unknown:
                raise ValueError(f"Unknown variables requested: {', '.join(unknown)}")
            return list(dict.fromkeys(vars_filter))

        if enforce_limit and len(self._variables) > self._MAX_AUTO_VARS:
            raise ValueError(
                f"{len(self._variables)} variables available; provide vars_filter to select a subset"
            )
        return sorted(self._variables)

    def _compute_belief_argmins(self, variables: Iterable[str]) -> Dict[str, List[int | None]]:
        result: Dict[str, List[int | None]] = {var: [] for var in variables}
        for record in self._records:
            message_groups = self._group_r_messages(record.messages)
            for var in variables:
                argmin_value = self._argmin_for_variable(var, message_groups, record.assignments)
                result[var].append(argmin_value)
        return result

    @staticmethod
    def _group_r_messages(messages: List[Dict[str, object]]) -> Dict[str, List[List[float]]]:
        grouped: Dict[str, List[List[float]]] = {}
        for message in messages or []:
            sender = str(message.get("sender"))
            recipient = str(message.get("recipient"))
            values = message.get("values")
            if not isinstance(values, list):
                continue
            if sender in (recipient, None):
                continue
            key = recipient
            grouped.setdefault(key, []).append(values)
        return grouped

    @staticmethod
    def _argmin_for_variable(
        var: str,
        grouped_messages: Dict[str, List[List[float]]],
        assignments: Dict[str, int],
    ) -> int | None:
        if var in grouped_messages:
            vectors = grouped_messages[var]
            length = len(vectors[0])
            for vec in vectors:
                if len(vec) != length:
                    raise ValueError(
                        f"Inconsistent message length for variable {var}: {len(vec)} vs {length}"
                    )
            combined = [sum(values) for values in zip(*vectors)]
            min_val = min(combined)
            for idx, value in enumerate(combined):
                if math.isclose(value, min_val, rel_tol=1e-9, abs_tol=1e-12):
                    return idx
        return assignments.get(var)

    @staticmethod
    def _collect_variables(records: Sequence[_SnapshotRecord]) -> List[str]:
        vars_set = set()
        for rec in records:
            vars_set.update(str(key) for key in rec.assignments.keys())
        if not vars_set:
            raise ValueError("No variable assignments found in snapshots")
        return sorted(vars_set)

    @staticmethod
    def _normalise_snapshots(snapshots: Sequence[SnapshotDict]) -> List[_SnapshotRecord]:
        records: Dict[int, _SnapshotRecord] = {}
        for entry in snapshots:
            if not isinstance(entry, dict):
                raise TypeError("Each snapshot must be a dict")
            if "step" not in entry:
                raise KeyError("Snapshot missing 'step' key")
            step = int(entry["step"])
            messages = entry.get("messages")
            assignments = entry.get("assignments")
            cost = entry.get("cost")
            if not isinstance(messages, list):
                raise TypeError("Snapshot 'messages' must be a list")
            if not isinstance(assignments, dict):
                raise TypeError("Snapshot 'assignments' must be a dict")
            record = _SnapshotRecord(step=step, messages=messages, assignments=assignments, cost=cost)
            # dedupe on highest step entry
            if step in records:
                records[step] = record
            else:
                records[step] = record
        ordered = sorted(records.values(), key=lambda rec: rec.step)
        expected_steps = list(range(ordered[0].step, ordered[0].step + len(ordered)))
        actual_steps = [rec.step for rec in ordered]
        if actual_steps != expected_steps:
            raise ValueError("Step numbering is inconsistent or has gaps")
        return ordered


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot argmin trajectories from snapshot JSON")
    parser.add_argument("json_path", help="Path to snapshot JSON file")
    parser.add_argument("--vars", nargs="*", help="Variable names to plot")
    parser.add_argument("--save", help="Optional path to save the plot image")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    args = parser.parse_args(argv)

    visualizer = SnapshotVisualizer.from_json(args.json_path)
    visualizer.plot_argmin_per_variable(
        vars_filter=args.vars,
        show=not args.no_show,
        savepath=args.save,
    )


if __name__ == "__main__":  # pragma: no cover
    _main()
