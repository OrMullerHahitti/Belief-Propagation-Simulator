"""Visualization utilities for belief propagation snapshot trajectories."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .types import SnapshotRecord


class SnapshotVisualizer:
    """Visualize belief argmin trajectories from propflow snapshots."""

    _MAX_AUTO_VARS = 20
    _SMALL_PLOT_THRESHOLD = 8

    def __init__(self, snapshots: Sequence[SnapshotRecord]):
        """Initialize the visualizer with snapshot records.

        Args:
            snapshots: A sequence of SnapshotRecord objects.

        Raises:
            ValueError: If snapshots is empty or contains no variables.
        """
        if not snapshots:
            raise ValueError("Snapshots are empty")

        self._records = sorted(list(snapshots), key=lambda rec: rec.data.step)
        self._steps = [rec.data.step for rec in self._records]
        self._variables = self._collect_variables(self._records)

        if not self._variables:
            raise ValueError("No variable assignments found in snapshots")

    def variables(self) -> List[str]:
        """Return sorted list of all variables in the snapshots."""
        return sorted(self._variables)

    def argmin_series(
        self, vars_filter: List[str] | None = None
    ) -> Dict[str, List[int | None]]:
        """Get argmin trajectories for selected variables.

        Args:
            vars_filter: Optional list of variable names to include.
                If None, all variables are included.

        Returns:
            Dictionary mapping variable names to their argmin trajectories.
        """
        target_vars = self._select_variables(vars_filter)
        result: Dict[str, List[int | None]] = {var: [] for var in target_vars}

        for rec in self._records:
            data = rec.data

            # Group R messages by recipient (variable)
            r_grouped: Dict[str, List[np.ndarray]] = {}
            for (f, v), r_msg in data.R.items():
                values = np.asarray(r_msg, dtype=float)
                r_grouped.setdefault(v, []).append(values)

            for var in target_vars:
                vectors = r_grouped.get(var, [])
                if vectors:
                    combined = np.sum(vectors, axis=0)
                    argmin = int(np.argmin(combined))
                    result[var].append(argmin)
                else:
                    # Fallback to stored assignment
                    result[var].append(data.assignments.get(var))

        return result

    def plot_argmin_per_variable(
        self,
        vars_filter: List[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        savepath: str | None = None,
        combined_savepath: str | None = None,
    ) -> None:
        """Plot argmin trajectories for selected variables.

        Args:
            vars_filter: Optional list of variable names to plot.
            figsize: Figure size tuple (width, height).
            show: Whether to display the plot.
            savepath: Optional path to save individual variable plots.
            combined_savepath: Optional path to save combined multi-variable plot.
        """
        target_vars = self._select_variables(vars_filter, enforce_limit=True)
        series = self.argmin_series(target_vars)

        steps = self._steps
        if not steps:
            raise ValueError("No steps to plot")

        per_var_fig = None

        # Create per-variable subplots if small number of variables
        if len(target_vars) <= self._SMALL_PLOT_THRESHOLD:
            per_var_fig, axes = plt.subplots(
                len(target_vars), 1, figsize=figsize or (10, 3 * len(target_vars))
            )
            if len(target_vars) == 1:
                axes = [axes]
            for ax, var in zip(axes, target_vars):
                self._plot_variable_series(ax, var, steps, series[var])
            plt.tight_layout()
        else:
            # For many variables, use combined plot
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

        # Save per-variable plot if requested
        derived_combined = combined_savepath
        if savepath:
            save_path_obj = Path(savepath)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            per_var_fig.savefig(save_path_obj, dpi=150)
            if derived_combined is None and len(target_vars) > 1:
                derived_combined = str(
                    save_path_obj.with_name(f"{save_path_obj.stem}_combined{save_path_obj.suffix}")
                )

        # Create and save combined plot if needed
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

        # Show or close per-variable figure
        if show and len(target_vars) <= self._SMALL_PLOT_THRESHOLD:
            per_var_fig.show()
        else:
            plt.close(per_var_fig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_integer_ticks(ax, series: Dict[str, List[int | None]]) -> None:
        """Set y-axis ticks to integer values found in series."""
        values = [v for seq in series.values() for v in seq if v is not None]
        if values:
            ax.set_yticks(sorted(set(values)))

    @staticmethod
    def _plot_variable_series(
        ax, var: str, steps: Sequence[int], series: Sequence[int | None]
    ) -> None:
        """Plot a single variable's argmin trajectory."""
        ax.plot(steps, series, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Argmin index")
        ax.set_title(var)
        ax.grid(True, alpha=0.3)
        valid_values = [value for value in series if value is not None]
        if valid_values:
            ax.set_yticks(sorted(set(valid_values)))

    def _select_variables(
        self, vars_filter: List[str] | None, *, enforce_limit: bool = False
    ) -> List[str]:
        """Select variables to plot.

        Args:
            vars_filter: Optional filter list.
            enforce_limit: If True, raise error if too many variables.

        Returns:
            List of selected variable names.
        """
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

    @staticmethod
    def _collect_variables(records: Sequence[SnapshotRecord]) -> set[str]:
        """Collect all variable names from snapshot assignments."""
        vars_set = set()
        for rec in records:
            vars_set.update(str(key) for key in rec.data.assignments.keys())
        return vars_set


__all__ = ["SnapshotVisualizer"]
