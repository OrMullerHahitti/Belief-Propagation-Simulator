"""Visualization utilities for belief propagation snapshot trajectories."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Any

import matplotlib.pyplot as plt
import numpy as np

from .types import SnapshotRecord
from propflow.utils.tools.bct import BCTCreator


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

    def plot_bct(
        self,
        variable_name: str,
        iteration: int | None = None,
        *,
        show: bool = True,
        savepath: str | None = None,
    ) -> BCTCreator:
        """Plot a Backtrack Cost Tree (BCT) for a variable from snapshots.

        Reconstructs BCT data from snapshot Q and R messages, then visualizes
        how costs and beliefs from earlier iterations contribute to the final
        belief of the specified variable.

        Args:
            variable_name: The name of the variable to visualize the BCT for.
            iteration: The iteration to trace back from. Defaults to None (last step).
                If None, uses -1 (the last captured iteration).
            show: Whether to display the plot.
            savepath: Optional path to save the generated figure.

        Returns:
            The BCTCreator object for further analysis (e.g., convergence analysis,
            variable comparisons). Can be used to call methods like
            analyze_convergence(), compare_variables(), export_analysis(), etc.

        Raises:
            ValueError: If the variable_name is not found in the snapshots.
        """
        if variable_name not in self._variables:
            raise ValueError(f"Variable {variable_name} not found in snapshots")

        if iteration is None:
            iteration = -1

        # Reconstruct BCT data from snapshots
        bct_data = self._reconstruct_bct_data_from_snapshots()

        # Create a mock history object that works with BCTCreator
        mock_history = _SnapshotBasedHistory(bct_data)

        # Create BCTCreator with the reconstructed data
        creator = BCTCreator(
            factor_graph=None,  # Not needed when using snapshot-based data
            history=mock_history,
            damping_factor=self._records[0].data.lambda_ if self._records else 0.0,
        )

        # Visualize the BCT
        creator.visualize_bct(variable_name, iteration=iteration, save_path=savepath)

        if show:
            plt.show()

        return creator

    def _reconstruct_bct_data_from_snapshots(self) -> Dict[str, Any]:
        """Reconstruct BCT data structure from snapshot messages.

        Builds a data structure compatible with BCTCreator that contains:
        - Belief evolution per variable
        - Assignment evolution per variable
        - Message flows between agents
        - Metadata and costs

        Returns:
            A dictionary with keys: 'beliefs', 'assignments', 'messages', 'costs', 'metadata'.
        """
        bct_data: Dict[str, Any] = {
            "beliefs": {},
            "assignments": {},
            "messages": {},
            "costs": [],
            "metadata": {"total_steps": len(self._records)},
        }

        # Extract belief and assignment trajectories for all variables
        all_variables = self.variables()
        for var in all_variables:
            beliefs = []
            assignments = []
            for rec in self._records:
                data = rec.data
                # Get belief, fallback to assignment
                belief = data.beliefs.get(var, data.assignments.get(var, 0.0))
                beliefs.append(float(belief))
                assignments.append(data.assignments.get(var))

            bct_data["beliefs"][var] = beliefs
            bct_data["assignments"][var] = assignments

        # Extract message flows from Q and R messages
        message_flows: Dict[str, List[float]] = {}

        for rec in self._records:
            data = rec.data

            # Process Q messages (variable -> factor)
            for (var, factor), q_msg in data.Q.items():
                flow_key = f"{var}->{factor}"
                if flow_key not in message_flows:
                    message_flows[flow_key] = []
                # Store the sum of Q-message values
                q_value = float(np.sum(np.asarray(q_msg, dtype=float)))
                message_flows[flow_key].append(q_value)

            # Process R messages (factor -> variable)
            for (factor, var), r_msg in data.R.items():
                flow_key = f"{factor}->{var}"
                if flow_key not in message_flows:
                    message_flows[flow_key] = []
                # Store the sum of R-message values
                r_value = float(np.sum(np.asarray(r_msg, dtype=float)))
                message_flows[flow_key].append(r_value)

            # Extract global cost if available
            if data.global_cost is not None:
                bct_data["costs"].append(data.global_cost)

        bct_data["messages"] = message_flows

        return bct_data


class _SnapshotBasedHistory:
    """Mock history object that provides BCTCreator-compatible interface from snapshots.

    This allows BCTCreator to work with snapshot-derived data without requiring
    the original History object.
    """

    def __init__(self, bct_data: Dict[str, Any]):
        """Initialize with pre-constructed BCT data.

        Args:
            bct_data: Dictionary with keys 'beliefs', 'assignments', 'messages', 'costs', 'metadata'.
        """
        self.bct_data = bct_data
        self.use_bct_history = True

    def get_bct_data(self) -> Dict[str, Any]:
        """Return the reconstructed BCT data.

        Returns:
            The BCT data dictionary.
        """
        return self.bct_data


__all__ = ["SnapshotVisualizer"]
