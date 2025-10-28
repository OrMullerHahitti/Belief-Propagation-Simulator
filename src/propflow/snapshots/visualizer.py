"""Visualization utilities for belief propagation snapshot trajectories."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from .types import SnapshotRecord
from propflow.utils.tools.bct import BCTCreator
from propflow.core.agents import FactorAgent

FactorLike = str | FactorAgent


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

    def steps(self) -> List[int]:
        """Return the ordered simulation steps captured in the snapshots."""
        return list(self._steps)

    def _snapshot_by_step(self, step: int) -> SnapshotRecord:
        try:
            idx = self._steps.index(step)
        except ValueError as exc:  # pragma: no cover - defensive
            available = ", ".join(str(s) for s in self._steps)
            raise ValueError(f"No snapshot recorded for step {step}. Available steps: {available}") from exc
        return self._records[idx]

    @staticmethod
    def _factor_name(factor: FactorLike) -> str:
        if isinstance(factor, FactorAgent):
            return factor.name
        if isinstance(factor, str):
            return factor
        name = getattr(factor, "name", None)
        if not name:
            raise ValueError("Factor reference must be a name or an object with a 'name' attribute")
        return str(name)

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
                if vectors := r_grouped.get(var, []):
                    combined = np.sum(vectors, axis=0)
                    argmin = int(np.argmin(combined))
                    result[var].append(argmin)
                else:
                    # Fallback to stored assignment
                    result[var].append(data.assignments.get(var))

        return result

    def factor_cost_matrix(self, factor: FactorLike, step: int) -> np.ndarray:
        """Return a copy of a factor's cost table at a given step."""
        factor_name = self._factor_name(factor)
        record = self._snapshot_by_step(step)
        tables = getattr(record.data, "cost_tables", {})
        if factor_name not in tables:
            available = ", ".join(sorted(tables.keys())) or "none"
            raise ValueError(
                f"Snapshot {step} does not contain a cost table for factor '{factor_name}'."
                f" Available factors: {available}"
            )
        return np.asarray(tables[factor_name], dtype=float).copy()

    def factor_cost_labels(self, factor: FactorLike, step: int) -> Tuple[List[str], List[str]]:
        factor_name = self._factor_name(factor)
        record = self._snapshot_by_step(step)
        labels = getattr(record.data, "cost_labels", {}).get(factor_name)
        if not labels:
            raise ValueError(f"No variable ordering stored for factor '{factor_name}' at step {step}.")
        if len(labels) != 2:
            raise ValueError(
                f"Cost visualisation currently supports binary factors only."
                f" Factor '{factor_name}' has {len(labels)} variables."
            )
        row_var, col_var = labels
        dom = record.data.dom
        row_labels = dom.get(row_var) or [f"{row_var}:{i}" for i in range(self._infer_domain_size(row_var, record))]
        col_labels = dom.get(col_var) or [f"{col_var}:{i}" for i in range(self._infer_domain_size(col_var, record))]
        return row_labels, col_labels

    @staticmethod
    def _infer_domain_size(var: str, record: SnapshotRecord) -> int:
        size = len(record.data.dom.get(var, []))
        if size:
            return size
        assignments = record.data.assignments.get(var)
        return int(assignments) + 1 if assignments is not None else 0

    def _prepare_cost_display(
        self,
        matrix: np.ndarray,
        factor: str,
        step: int,
    ) -> Tuple[np.ndarray, List[str], List[str], str, str]:
        if matrix.ndim != 2:
            raise ValueError(
                f"Factor '{factor}' cost table has shape {matrix.shape}; only 2D tables are supported for plotting."
            )
        row_labels, col_labels = self.factor_cost_labels(factor, step)
        if matrix.shape != (len(row_labels), len(col_labels)):
            raise ValueError(
                "Cost table shape does not match domain sizes: "
                f"matrix={matrix.shape}, rows={len(row_labels)}, cols={len(col_labels)}"
            )
        record = self._snapshot_by_step(step)
        row_var, col_var = getattr(record.data, "cost_labels", {}).get(factor, ["rows", "cols"])
        return matrix, row_labels, col_labels, row_var, col_var

    def _draw_cost_heatmap(
        self,
        ax: plt.Axes,
        matrix: np.ndarray,
        factor: str,
        step: int,
        row_labels: List[str],
        col_labels: List[str],
        row_name: str,
        col_name: str,
        cmap: str,
    ) -> plt.Axes: # type: ignore
        im = ax.imshow(matrix, aspect="equal", cmap=cmap)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.set_xlabel(col_name)
        ax.set_ylabel(row_name)
        ax.set_title(f"{factor} cost table (step {step})")
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1.0, alpha=0.6)
        return im # type: ignore

    def _infer_message_mode(self) -> Literal["min", "max"]:
        """Infer whether to highlight minima or maxima from snapshot metadata."""
        if not self._records:
            return "min"
        metadata = getattr(self._records[0].data, "metadata", {}) or {}
        name = str(metadata.get("computator", "")).lower()
        return "max" if "max" in name else "min"

    def _render_factor_panel(
        self,
        ax: plt.Axes, # pyright: ignore[reportPrivateImportUsage]
        from_variable: str,
        factor: FactorLike,
        step: int,
        mode: Literal["min", "max"],
        cmap: str,
        annotate: bool,
        highlight_color: str,
        text_color: str,
        fmt: str,
    ) -> Tuple[np.ndarray, np.ndarray, plt.AxesImage]: # pyright: ignore[reportPrivateImportUsage]
        # sourcery skip: low-code-quality
        factor_name = self._factor_name(factor)
        record = self._snapshot_by_step(step)
        neighbours = record.data.N_fac.get(factor_name, [])
        if from_variable not in neighbours:
            raise ValueError(
                f"Variable '{from_variable}' is not connected to factor '{factor_name}' at step {step}."
            )

        labels = getattr(record.data, "cost_labels", {}).get(factor_name)
        if not labels or len(labels) != 2:
            raise ValueError(
                "plot_factor_costs currently supports binary factors only. "
                f"Factor '{factor_name}' has variable ordering {labels}."
            )

        try:
            target_index = labels.index(from_variable)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Variable '{from_variable}' not present in factor '{factor_name}' ordering {labels}."
            ) from exc

        other_index = 1 - target_index
        other_variable = labels[other_index]

        matrix = self.factor_cost_matrix(factor_name, step)
        if matrix.ndim != 2:
            raise ValueError(
                f"Factor '{factor_name}' cost table has shape {matrix.shape}; only 2D tables are supported."
            )

        aligned = matrix if target_index == 0 else np.swapaxes(matrix, 0, 1)

        row_labels = record.data.dom.get(from_variable)
        col_labels = record.data.dom.get(other_variable)
        if not row_labels or not col_labels:
            raise ValueError(
                "Domain labels missing for factor visualisation: "
                f"rows={row_labels}, cols={col_labels}"
            )

        # Get Q message from from_variable only (not both variables)
        from_q_message = record.data.Q.get((from_variable, factor_name))
        from_msg = np.zeros(len(row_labels)) if from_q_message is None else np.asarray(from_q_message, dtype=float)

        # Compute effective cost by adding Q message to from_variable's dimension only
        if target_index == 0:  # from_variable is rows
            effective = aligned + from_msg[:, None]
        else:  # from_variable is columns
            effective = aligned + from_msg[None, :]

        # Compute R message by reducing over the OTHER variable's dimension
        # This mimics what compute_R does: remove target's Q, reduce over target's dimension
        reduce_axis = 1 - target_index  # Opposite of from_variable's axis!
        r_message = np.min(effective, axis=reduce_axis) if mode == "min" else np.max(effective, axis=reduce_axis)

        # Determine tolerance for comparisons
        tol = 1e-12 + 1e-9 * max(1.0, np.ptp(effective))

        # Find cells that produce each R message value (primary highlighting)
        winners = np.zeros_like(effective, dtype=bool)
        if target_index == 0:  # from_var is rows → reduce over cols (axis 1) → R is per row
            for i in range(len(row_labels)):
                winners[i, :] = np.abs(effective[i, :] - r_message[i]) <= tol
        else:  # from_var is cols → reduce over rows (axis 0) → R is per column
            for j in range(len(col_labels)):
                winners[:, j] = np.abs(effective[:, j] - r_message[j]) <= tol

        # Find the absolute minimum/maximum of R message (secondary highlighting)
        r_best = np.min(r_message) if mode == "min" else np.max(r_message)
        r_best_indices = np.where(np.abs(r_message - r_best) <= tol)[0]

        # Mark cells that produce the best R value
        best_winners = np.zeros_like(effective, dtype=bool)
        if target_index == 0:  # R is per row
            for i in r_best_indices:
                best_winners[i, :] = winners[i, :]
        else:  # R is per column
            for j in r_best_indices:
                best_winners[:, j] = winners[:, j]

        # Draw heatmap showing original cost table (not effective cost)
        im = self._draw_cost_heatmap(
            ax,
            aligned,
            factor_name,
            step,
            row_labels,
            col_labels,
            from_variable,
            other_variable,
            cmap,
        )

        # Draw two-level highlighting with improved visibility
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                # Primary highlighting (red) for all cells that produce R message values
                if winners[i, j] and not best_winners[i, j]:
                    # Inset slightly for rounded appearance
                    rect = Rectangle(
                        (j - 0.45, i - 0.45),
                        0.9,
                        0.9,
                        fill=True,
                        facecolor='red',
                        alpha=0.15,
                        linewidth=3.5,
                        edgecolor='red',
                    )
                    ax.add_patch(rect)

                # Secondary highlighting (gold/orange) for cells that produce best R value
                if best_winners[i, j]:
                    rect = Rectangle(
                        (j - 0.45, i - 0.45),
                        0.9,
                        0.9,
                        fill=True,
                        facecolor='gold',
                        alpha=0.4,
                        linewidth=4,
                        edgecolor='darkorange',
                    )
                    ax.add_patch(rect)

                if annotate:
                    ax.text(
                        j, i,
                        fmt.format(aligned[i, j]),
                        ha='center',
                        va='center',
                        color=text_color,
                        fontsize=10,
                        fontweight='bold' if winners[i, j] else 'normal'
                    )

        ax.set_xlim(-0.5, len(col_labels) - 0.5)

        # Compute message for return compatibility (sum of incoming R messages)
        message = record.data.R.get((factor_name, from_variable))
        message_arr = np.zeros(len(row_labels)) if message is None else np.asarray(message, dtype=float)
        winners_mask = np.any(winners, axis=1)  # At least one winner per row

        return message_arr, winners_mask, im

    def _plot_factor_grid(
        self,
        pairs: Sequence[tuple[str, FactorLike]],
        *,
        step: int,
        mode: Literal["min", "max"],
        cmap: str,
        annotate: bool,
        show: bool,
        savepath: str | None,
        highlight_color: str,
        text_color: str,
        fmt: str,
    ) -> plt.Figure:
        if not pairs:
            raise ValueError("No factor pairs supplied for plotting.")

        ncols = min(3, len(pairs))
        nrows = math.ceil(len(pairs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        flat_axes = np.atleast_1d(axes).flatten()
        ims: List[plt.AxesImage] = []

        for ax, (var, fac) in zip(flat_axes, pairs):
            message_arr, winners_mask, im = self._render_factor_panel(
                ax,
                var,
                fac,
                step,
                mode,
                cmap,
                annotate,
                highlight_color,
                text_color,
                fmt,
            )
            ims.append(im)

        for ax in flat_axes[len(pairs):]:
            ax.axis("off")

        if ims:
            fig.colorbar(ims[0], ax=flat_axes[:len(pairs)], shrink=0.85)

        fig.tight_layout()

        if savepath:
            save_path = Path(savepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            fig.show()
        else:  # pragma: no cover - interactive branch
            plt.close(fig)

        return fig




    def plot_factor_costs(
        self,
        from_variable: str | Sequence[tuple[str, FactorLike]],
        to_factor: FactorLike | None = None,
        step: int | None = None,
        *,
        mode: Literal["auto", "min", "max"] = "auto",
        cmap: str = "viridis",
        annotate: bool = True,
        show: bool = True,
        savepath: str | None = None,
        return_data: bool = False,
        highlight_color: str = "tab:red",
        text_color: str = "black",
        fmt: str = "{:.3g}",
    ) -> plt.Figure | Tuple[plt.Figure, np.ndarray, np.ndarray]:
        """Visualise factor cost tables induced by factor→variable messages."""
        if isinstance(from_variable, (list, tuple)) and not isinstance(from_variable, str):
            if to_factor is not None:
                raise ValueError("When providing multiple factor pairs, omit the 'to_factor' argument.")
            if step is None:
                raise ValueError("Step must be provided when plotting multiple factor panels.")
            if return_data:
                raise ValueError("return_data is only supported for single factor visualisations.")
            pairs: List[tuple[str, FactorLike]] = []
            for item in from_variable:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError("Each entry must be a (variable, factor) pair.")
                var, fac = item
                pairs.append((str(var), fac))
            real_mode = self._infer_message_mode() if mode == "auto" else mode
            return self._plot_factor_grid(
                pairs,
                step=step,
                mode=real_mode,
                cmap=cmap,
                annotate=annotate,
                show=show,
                savepath=savepath,
                highlight_color=highlight_color,
                text_color=text_color,
                fmt=fmt,
            )

        if to_factor is None:
            raise ValueError("to_factor must be provided when plotting a single factor panel.")
        if step is None:
            raise ValueError("step must be provided.")

        real_mode = self._infer_message_mode() if mode == "auto" else mode

        fig, ax = plt.subplots(figsize=(6, 5))
        message_arr, winners_mask, im = self._render_factor_panel(
            ax,
            str(from_variable),
            to_factor,
            step,
            real_mode,
            cmap,
            annotate,
            highlight_color,
            text_color,
            fmt,
        )
        fig.colorbar(im, ax=ax, shrink=0.85)

        subtitle = "argmin" if real_mode == "min" else "argmax"
        factor_name = self._factor_name(to_factor)
        ax.set_title(f"{factor_name} → {from_variable} ({subtitle} of message)")

        fig.tight_layout()

        if savepath:
            save_path = Path(savepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            fig.show()
        else:  # pragma: no cover - interactive branch
            plt.close(fig)

        return (fig, message_arr, winners_mask) if return_data else fig
    def plot_argmin_per_variable(
        self,
        vars_filter: List[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        savepath: str | None = None,
        combined_savepath: str | None = None,
        layout: Literal["separate", "combined"] = "separate",
    ) -> None:
        """Plot argmin trajectories for selected variables.

        Args:
            vars_filter: Optional list of variable names to plot.
            figsize: Figure size tuple (width, height).
            show: Whether to display the plot.
            savepath: Optional path to save individual variable plots (separate layout).
            combined_savepath: Optional path to save a combined plot.
            layout: Choose "separate" for per-variable panels or "combined" for a single figure.
        """
        layout_choice = layout.lower()
        if layout_choice not in {"separate", "combined"}:
            raise ValueError("layout must be 'separate' or 'combined'")

        target_vars = self._select_variables(vars_filter, enforce_limit=True)
        series = self.argmin_series(target_vars)

        steps = self._steps
        if not steps:
            raise ValueError("No steps to plot")

        if layout_choice == "combined" or len(target_vars) > self._SMALL_PLOT_THRESHOLD:
            fig, ax = plt.subplots(figsize=figsize or (12, 6))
            for var in target_vars:
                ax.plot(steps, series[var], marker="o", label=var) # type: ignore
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Argmin index")
            ax.set_title("Belief argmin trajectories")
            ax.grid(True, alpha=0.3)
            ax.legend()
            self._set_integer_ticks(ax, series)
            plt.tight_layout()

            if savepath:
                self._extracted_from_plot_argmin_per_variable_45(savepath, fig)
            if combined_savepath and layout_choice != "combined":
                self._extracted_from_plot_argmin_per_variable_45(combined_savepath, fig)
            if show:
                fig.show()
            else:
                plt.close(fig)
            return

        per_var_fig, axes = plt.subplots(
            len(target_vars), 1, figsize=figsize or (10, 3 * len(target_vars))
        )
        if len(target_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, target_vars):
            self._plot_variable_series(ax, var, steps, series[var])
        plt.tight_layout()

        if savepath:
            self._extracted_from_plot_argmin_per_variable_45(savepath, per_var_fig)
        if combined_savepath and len(target_vars) > 1:
            self._extracted_from_plot_argmin_per_variable_45(
                combined_savepath, per_var_fig
            )
        if show:
            per_var_fig.show()
        else:
            plt.close(per_var_fig)

    # TODO Rename this here and in `plot_argmin_per_variable`
    def _extracted_from_plot_argmin_per_variable_45(self, arg0, arg1):
        save_obj = Path(arg0)
        save_obj.parent.mkdir(parents=True, exist_ok=True)
        arg1.savefig(save_obj, dpi=150)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_integer_ticks(ax, series: Dict[str, List[int | None]]) -> None:
        """Set y-axis ticks to integer values found in series."""
        if values := [v for seq in series.values() for v in seq if v is not None]:
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
        if valid_values := [value for value in series if value is not None]:
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
            if unknown := [
                var for var in vars_filter if var not in self._variables
            ]:
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

    __all__ = ["SnapshotVisualizer"]



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
