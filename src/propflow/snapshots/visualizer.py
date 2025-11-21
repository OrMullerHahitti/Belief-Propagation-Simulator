"""Visualization utilities for belief propagation snapshot trajectories."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from .types import EngineSnapshot
from propflow.utils.tools.bct import BCTCreator, SnapshotBCTBuilder
from propflow.core.agents import FactorAgent

FactorLike = str | FactorAgent


class SnapshotVisualizer:
    """Visualize belief argmin trajectories from propflow snapshots."""

    _MAX_AUTO_VARS = 20
    _SMALL_PLOT_THRESHOLD = 8
    _MAX_AUTO_MESSAGE_PAIRS = 6

    def __init__(self, snapshots: Sequence[EngineSnapshot]):
        """Initialize the visualizer with snapshot records.

        Args:
            snapshots: A sequence of EngineSnapshot objects.

        Raises:
            ValueError: If snapshots is empty or contains no variables.
        """
        if not snapshots:
            raise ValueError("Snapshots are empty")

        self._records = sorted(list(snapshots), key=lambda rec: rec.step)
        self._steps = [rec.step for rec in self._records]
        self._variables = self._collect_variables(self._records)
        self._bct_builder: SnapshotBCTBuilder | None = None

        if not self._variables:
            raise ValueError("No variable assignments found in snapshots")

    def variables(self) -> List[str]:
        """Return sorted list of all variables in the snapshots."""
        return sorted(self._variables)

    def steps(self) -> List[int]:
        """Return the ordered simulation steps captured in the snapshots."""
        return list(self._steps)

    def _snapshot_by_step(self, step: int) -> EngineSnapshot:
        try:
            idx = self._steps.index(step)
        except ValueError as exc:  # pragma: no cover - defensive
            available = ", ".join(str(s) for s in self._steps)
            raise ValueError(
                f"No snapshot recorded for step {step}. Available steps: {available}"
            ) from exc
        return self._records[idx]

    @staticmethod
    def _factor_name(factor: FactorLike) -> str:
        if isinstance(factor, FactorAgent):
            return factor.name
        if isinstance(factor, str):
            return factor
        name = getattr(factor, "name", None)
        if not name:
            raise ValueError(
                "Factor reference must be a name or an object with a 'name' attribute"
            )
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
            data = rec

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

    def global_cost_series(
        self,
        *,
        include_missing: bool = False,
        fill_value: float = float("nan"),
    ) -> Tuple[List[int], List[float]]:
        """Return the global cost trajectory extracted from the snapshots.

        Args:
            include_missing: If True, include steps with missing costs using ``fill_value``.
            fill_value: Value to substitute whenever a snapshot lacks a global cost.

        Returns:
            A tuple ``(steps, costs)`` with matching lengths.

        Raises:
            ValueError: If no snapshots contain global cost information.
        """
        steps: List[int] = []
        costs: List[float] = []

        for rec in self._records:
            step = rec.step
            cost = rec.global_cost
            if cost is None:
                if include_missing:
                    steps.append(step)
                    costs.append(float(fill_value))
                continue
            steps.append(step)
            costs.append(float(cost))

        if not costs:
            raise ValueError("No global cost data available in snapshots")

        return steps, costs

    def factor_cost_matrix(self, factor: FactorLike, step: int) -> np.ndarray:
        """Return a copy of a factor's cost table at a given step."""
        factor_name = self._factor_name(factor)
        record = self._snapshot_by_step(step)
        tables = getattr(record, "cost_tables", {})
        if factor_name not in tables:
            available = ", ".join(sorted(tables.keys())) or "none"
            raise ValueError(
                f"Snapshot {step} does not contain a cost table for factor '{factor_name}'."
                f" Available factors: {available}"
            )
        return np.asarray(tables[factor_name], dtype=float).copy()

    def factor_cost_labels(
        self, factor: FactorLike, step: int
    ) -> Tuple[List[str], List[str]]:
        factor_name = self._factor_name(factor)
        record = self._snapshot_by_step(step)
        labels = getattr(record, "cost_labels", {}).get(factor_name)
        if not labels:
            raise ValueError(
                f"No variable ordering stored for factor '{factor_name}' at step {step}."
            )
        if len(labels) != 2:
            raise ValueError(
                f"Cost visualisation currently supports binary factors only."
                f" Factor '{factor_name}' has {len(labels)} variables."
            )
        row_var, col_var = labels
        dom = record.dom
        row_labels = dom.get(row_var) or [
            f"{row_var}:{i}" for i in range(self._infer_domain_size(row_var, record))
        ]
        col_labels = dom.get(col_var) or [
            f"{col_var}:{i}" for i in range(self._infer_domain_size(col_var, record))
        ]
        return row_labels, col_labels

    @staticmethod
    def _infer_domain_size(var: str, record: EngineSnapshot) -> int:
        size = len(record.dom.get(var, []))
        if size:
            return size
        assignments = record.assignments.get(var)
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
        row_var, col_var = getattr(record, "cost_labels", {}).get(
            factor, ["rows", "cols"]
        )
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
    ) -> plt.Axes:  # type: ignore
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
        return im  # type: ignore

    def _infer_message_mode(self) -> Literal["min", "max"]:
        """Infer whether to highlight minima or maxima from snapshot metadata."""
        if not self._records:
            return "min"
        metadata = getattr(self._records[0], "metadata", {}) or {}
        name = str(metadata.get("computator", "")).lower()
        return "max" if "max" in name else "min"

    def _render_factor_panel(
        self,
        ax: plt.Axes,  # pyright: ignore[reportPrivateImportUsage]
        from_variable: str,
        factor: FactorLike,
        step: int,
        mode: Literal["min", "max"],
        cmap: str,
        annotate: bool,
        highlight_color: str,
        text_color: str,
        fmt: str,
    ) -> Tuple[
        np.ndarray, np.ndarray, plt.AxesImage
    ]:  # pyright: ignore[reportPrivateImportUsage]
        # sourcery skip: low-code-quality
        factor_name = self._factor_name(factor)
        record = self._snapshot_by_step(step)
        neighbours = record.N_fac.get(factor_name, [])
        if from_variable not in neighbours:
            raise ValueError(
                f"Variable '{from_variable}' is not connected to factor '{factor_name}' at step {step}."
            )

        labels = getattr(record, "cost_labels", {}).get(factor_name)
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

        row_labels = record.dom.get(from_variable)
        col_labels = record.dom.get(other_variable)
        if not row_labels or not col_labels:
            raise ValueError(
                "Domain labels missing for factor visualisation: "
                f"rows={row_labels}, cols={col_labels}"
            )

        # Get Q message from from_variable only (not both variables)
        from_q_message = record.Q.get((from_variable, factor_name))
        from_msg = (
            np.zeros(len(row_labels))
            if from_q_message is None
            else np.asarray(from_q_message, dtype=float)
        )

        # Compute effective cost by adding Q message to from_variable's dimension only
        if target_index == 0:  # from_variable is rows
            effective = aligned + from_msg[:, None]
        else:  # from_variable is columns
            effective = aligned + from_msg[None, :]

        # Compute R message by reducing over the OTHER variable's dimension
        # This mimics what compute_R does: remove target's Q, reduce over target's dimension
        reduce_axis = 1 - target_index  # Opposite of from_variable's axis!
        r_message = (
            np.min(effective, axis=reduce_axis)
            if mode == "min"
            else np.max(effective, axis=reduce_axis)
        )

        # Determine tolerance for comparisons
        tol = 1e-12 + 1e-9 * max(1.0, np.ptp(effective))

        # Find cells that produce each R message value (primary highlighting)
        winners = np.zeros_like(effective, dtype=bool)
        if (
            target_index == 0
        ):  # from_var is rows → reduce over cols (axis 1) → R is per row
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
                        facecolor="red",
                        alpha=0.15,
                        linewidth=3.5,
                        edgecolor="red",
                    )
                    ax.add_patch(rect)

                # Secondary highlighting (gold/orange) for cells that produce best R value
                if best_winners[i, j]:
                    rect = Rectangle(
                        (j - 0.45, i - 0.45),
                        0.9,
                        0.9,
                        fill=True,
                        facecolor="gold",
                        alpha=0.4,
                        linewidth=4,
                        edgecolor="darkorange",
                    )
                    ax.add_patch(rect)

                if annotate:
                    ax.text(
                        j,
                        i,
                        fmt.format(aligned[i, j]),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                        fontweight="bold" if winners[i, j] else "normal",
                    )

        ax.set_xlim(-0.5, len(col_labels) - 0.5)

        # Compute message for return compatibility (sum of incoming R messages)
        message = record.R.get((factor_name, from_variable))
        message_arr = (
            np.zeros(len(row_labels))
            if message is None
            else np.asarray(message, dtype=float)
        )
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

        for ax in flat_axes[len(pairs) :]:
            ax.axis("off")

        if ims:
            fig.colorbar(ims[0], ax=flat_axes[: len(pairs)], shrink=0.85)

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
        if isinstance(from_variable, (list, tuple)) and not isinstance(
            from_variable, str
        ):
            if to_factor is not None:
                raise ValueError(
                    "When providing multiple factor pairs, omit the 'to_factor' argument."
                )
            if step is None:
                raise ValueError(
                    "Step must be provided when plotting multiple factor panels."
                )
            if return_data:
                raise ValueError(
                    "return_data is only supported for single factor visualisations."
                )
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
            raise ValueError(
                "to_factor must be provided when plotting a single factor panel."
            )
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

    def plot_global_cost(
        self,
        *,
        show: bool = True,
        savepath: str | None = None,
        include_missing: bool = False,
        fill_value: float = float("nan"),
        rolling_window: int | None = None,
        return_data: bool = False,
    ) -> plt.Figure | Tuple[plt.Figure, Dict[str, Any]]:
        """Plot the evolution of the global cost captured in the snapshots.

        Args:
            show: Whether to display the plot window.
            savepath: Optional file path to save the figure.
            include_missing: If True, include steps without a recorded cost using ``fill_value``.
            fill_value: Value used when ``include_missing`` is True and a step lacks a cost.
            rolling_window: Size of a trailing window to compute and overlay a rolling mean.
            return_data: If True, return the underlying data alongside the figure.

        Returns:
            The created matplotlib figure, optionally accompanied by the plotted data.
        """
        steps, costs = self.global_cost_series(
            include_missing=include_missing,
            fill_value=fill_value,
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, costs, marker="o", label="Global cost")

        rolling_info: Dict[str, Any] | None = None
        if rolling_window is not None and rolling_window > 1:
            if len(costs) >= rolling_window:
                smooth_steps, smooth_values = self._rolling_window_average(
                    costs,
                    steps,
                    rolling_window,
                )
                ax.plot(
                    smooth_steps,
                    smooth_values,
                    linestyle="--",
                    color="tab:orange",
                    label=f"{rolling_window}-step rolling mean",
                )
                rolling_info = {
                    "window": rolling_window,
                    "steps": smooth_steps,
                    "values": smooth_values,
                }
            else:
                rolling_info = {
                    "window": rolling_window,
                    "steps": [],
                    "values": [],
                }

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Global cost")
        ax.set_title("Global cost trajectory")
        ax.grid(True, alpha=0.3)
        if rolling_info is not None or len(costs) > 1:
            ax.legend()

        fig.tight_layout()

        if savepath:
            save_path = Path(savepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            fig.show()
        else:
            plt.close(fig)

        data = {
            "steps": steps,
            "costs": costs,
            "rolling": rolling_info,
        }
        return (fig, data) if return_data else fig

    def plot_message_norms(
        self,
        *,
        message_type: Literal["Q", "R"] = "Q",
        pairs: Sequence[tuple[str, str]] | None = None,
        norm: Literal["l2", "l1", "linf"] = "l2",
        show: bool = True,
        savepath: str | None = None,
        return_data: bool = False,
    ) -> plt.Figure | Tuple[plt.Figure, Dict[str, Any]]:
        """Plot the per-step norms of Q or R messages.

        Args:
            message_type: Select ``\"Q\"`` (variable→factor) or ``\"R\"`` (factor→variable``) messages.
            pairs: Optional explicit list of message pairs to include. Each tuple is ``(sender, recipient)``.
            norm: Vector norm used to summarise each message. Supported values are ``\"l2\"``, ``\"l1\"``, ``\"linf\"``.
            show: Whether to display the plot.
            savepath: Optional path to save the figure.
            return_data: If True, include the computed series in the return value.

        Returns:
            The created figure, optionally with the underlying message norm series.
        """
        msg_type = message_type.upper()
        if msg_type not in {"Q", "R"}:
            raise ValueError("message_type must be 'Q' or 'R'")

        available_pairs = self._collect_message_pairs(msg_type)
        if not available_pairs:
            raise ValueError(f"No {msg_type} messages recorded in the snapshots.")

        if pairs is None:
            target_pairs = available_pairs[: self._MAX_AUTO_MESSAGE_PAIRS]
        else:
            target_pairs = [(str(src), str(dst)) for src, dst in pairs]
            missing = [pair for pair in target_pairs if pair not in available_pairs]
            if missing:
                missing_str = ", ".join(f"{a}->{b}" for a, b in missing)
                raise ValueError(f"Requested message pairs not present: {missing_str}")

        series: Dict[tuple[str, str], List[float]] = {pair: [] for pair in target_pairs}

        for rec in self._records:
            messages = rec.Q if msg_type == "Q" else rec.R
            for pair in target_pairs:
                payload = messages.get(pair)
                if payload is None:
                    series[pair].append(float("nan"))
                else:
                    series[pair].append(self._message_norm(payload, norm))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for pair, values in series.items():
            label = f"{pair[0]}→{pair[1]}"
            ax.plot(self._steps, values, marker="o", label=label)

        direction = "Variable→Factor" if msg_type == "Q" else "Factor→Variable"
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{norm.upper()}-norm")
        ax.set_title(f"{msg_type} message norms ({direction})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        if savepath:
            save_path = Path(savepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            fig.show()
        else:
            plt.close(fig)

        data = {
            "steps": list(self._steps),
            "series": series,
            "message_type": msg_type,
            "norm": norm,
        }
        return (fig, data) if return_data else fig

    def plot_assignment_heatmap(
        self,
        vars_filter: List[str] | None = None,
        *,
        show: bool = True,
        savepath: str | None = None,
        cmap: str = "viridis",
        missing_value: float = float("nan"),
        annotate: bool = True,
        value_labels: Mapping[int, str] | Sequence[str] | None = None,
        return_data: bool = False,
    ) -> plt.Figure | Tuple[plt.Figure, Dict[str, Any]] | None:
        """Plot variable assignments over time as a heatmap.

        Args:
            vars_filter: Optional subset of variables to include.
            show: Whether to display the figure window.
            savepath: Optional path to save the generated figure.
            cmap: Matplotlib colormap name to use for the heatmap.
            missing_value: Value inserted when an assignment is missing for a step.
                Defaults to ``NaN`` so gaps appear as empty cells.
            annotate: Whether to write assignment values inside each cell.
            value_labels: Optional mapping or ordered list that converts assignment
                indices to display labels (e.g., ``{0: \"A\", 1: \"B\"}`` or
                ``[\"A\", \"B\", \"C\"]``).
            return_data: If True, return the data used for plotting alongside the figure.

        Returns:
            The heatmap figure, optionally with the underlying matrix.
        """
        target_vars = self._select_variables(vars_filter)
        if not target_vars:
            raise ValueError("No variables available to plot assignments.")

        steps = self._steps
        matrix = np.full((len(target_vars), len(steps)), missing_value, dtype=float)

        for col, rec in enumerate(self._records):
            assignments = rec.assignments
            for row, var in enumerate(target_vars):
                value = assignments.get(var)
                if value is None:
                    continue
                matrix[row, col] = float(value)

        if np.all(np.isnan(matrix)):
            raise ValueError("No assignments recorded for the selected variables.")

        label_lookup: Dict[int, str] = {}
        if value_labels is not None:
            if isinstance(value_labels, Mapping):
                label_lookup = {int(k): str(v) for k, v in value_labels.items()}
            elif isinstance(value_labels, Sequence) and not isinstance(
                value_labels, (str, bytes)
            ):
                label_lookup = {
                    idx: str(name) for idx, name in enumerate(value_labels)
                }
            else:
                raise TypeError(
                    "value_labels must be a mapping or a sequence of labels"
                )

        fig_width = max(6.0, 0.6 * len(steps))
        fig_height = max(4.0, 0.5 * len(target_vars))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=45, ha="right") # type: ignore
        ax.set_yticks(range(len(target_vars)))
        ax.set_yticklabels(target_vars)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Variable")
        ax.set_title("Assignment heatmap")

        color_threshold = np.nanmean(matrix)
        if np.isnan(color_threshold):
            color_threshold = 0.0

        if annotate:
            for row in range(len(target_vars)):
                for col in range(len(steps)):
                    value = matrix[row, col]
                    if np.isnan(value):
                        continue
                    int_value = int(round(value))
                    label = label_lookup.get(int_value, f"{int_value}")
                    ax.text(
                        col,
                        row,
                        label,
                        ha="center",
                        va="center",
                        color="white" if value > color_threshold else "black",
                        fontsize=10,
                        fontweight="bold",
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        if label_lookup:
            unique_values = sorted(
                {
                    int(round(v))
                    for v in np.unique(matrix[~np.isnan(matrix)])
                }
            )
            cbar.set_ticks(unique_values)
            cbar.set_ticklabels(
                [label_lookup.get(val, str(val)) for val in unique_values]
            )
            cbar.set_label("Assignment label")
        else:
            cbar.set_label("Assignment index")
        fig.tight_layout()

        if savepath:
            save_path = Path(savepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            fig.show()
            return None
        else:
            plt.close(fig)

        payload = {
            "variables": target_vars,
            "steps": steps,
            "matrix": matrix,
            "value_labels": label_lookup or None,
        }
        return (fig, payload) if return_data else fig

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
                ax.plot(steps, series[var], marker="o", label=var)  # type: ignore
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
    def _rolling_window_average(
        values: Sequence[float],
        steps: Sequence[int],
        window: int,
    ) -> Tuple[List[int], List[float]]:
        """Compute a trailing rolling mean and align it with the corresponding steps."""
        if window <= 1:
            return list(steps), [float(v) for v in values]

        arr = np.asarray(values, dtype=float)
        if len(arr) < window:
            raise ValueError("window length exceeds number of available points")

        cumulative = np.cumsum(arr, dtype=float)
        cumulative[window:] = cumulative[window:] - cumulative[:-window]
        averages = (cumulative[window - 1 :] / window).tolist()
        step_list = list(steps)
        return step_list[window - 1 :], averages

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
            if unknown := [var for var in vars_filter if var not in self._variables]:
                raise ValueError(f"Unknown variables requested: {', '.join(unknown)}")
            return list(dict.fromkeys(vars_filter))

        if enforce_limit and len(self._variables) > self._MAX_AUTO_VARS:
            raise ValueError(
                f"{len(self._variables)} variables available; provide vars_filter to select a subset"
            )
        return sorted(self._variables)

    @staticmethod
    def _collect_variables(records: Sequence[EngineSnapshot]) -> set[str]:
        """Collect all variable names from snapshot assignments."""
        vars_set = set()
        for rec in records:
            vars_set.update(str(key) for key in rec.assignments.keys())
        return vars_set

    def _collect_message_pairs(
        self, message_type: Literal["Q", "R"]
    ) -> List[tuple[str, str]]:
        """Collect all unique sender/recipient pairs for a message type."""
        pairs: set[tuple[str, str]] = set()
        for rec in self._records:
            store = rec.Q if message_type == "Q" else rec.R
            for src, dst in store.keys():
                pairs.add((str(src), str(dst)))
        return sorted(pairs)

    @staticmethod
    def _message_norm(values: Any, norm: Literal["l2", "l1", "linf"]) -> float:
        """Compute a vector norm for a message payload."""
        arr = np.asarray(values, dtype=float)
        if norm == "l2":
            return float(np.linalg.norm(arr))
        if norm == "l1":
            return float(np.linalg.norm(arr, ord=1))
        if norm == "linf":
            return float(np.linalg.norm(arr, ord=np.inf))
        raise ValueError(f"Unsupported norm: {norm}")

    def plot_bct(
        self,
        variable_name: str,
        iteration: int | None = None,
        *,
        value_index: int | None = None,
        steps_back: int | None = None,
        show: bool = True,
        savepath: str | None = None,
        verbose: bool = False,
    ) -> BCTCreator:
        """Plot a Backtrack Cost Tree (BCT) for a variable from snapshots.

        Reconstructs BCT data from snapshot Q and R messages, then visualizes
        how costs and beliefs from earlier iterations contribute to the final
        belief of the specified variable.

        Args:
            variable_name: The name of the variable to visualize the BCT for.
            iteration: The iteration index to trace back from. Defaults to None (last step).
                If None, uses -1 (the last captured iteration).
            steps_back: Optional number of steps from the end to anchor the tree.
                For example, ``steps_back=10`` traces the state 10 steps before the last
                recorded snapshot. When provided, overrides ``iteration``.
            show: Whether to display the plot.
            savepath: Optional path to save the generated figure.
            verbose: If True, annotate edges with message costs that generated
                each contribution in addition to assignments and table costs.

        Returns:
            The BCTCreator object for further analysis (e.g., convergence analysis,
            variable comparisons). Can be used to call methods like
            analyze_convergence(), compare_variables(), export_analysis(), etc.

        Raises:
            ValueError: If the variable_name is not found in the snapshots.
        """
        if variable_name not in self._variables:
            raise ValueError(f"Variable {variable_name} not found in snapshots")

        builder = self._ensure_bct_builder()
        total_steps = len(self._records)
        resolved_step = self._resolve_bct_iteration(iteration, steps_back, total_steps)
        target_value = value_index
        if target_value is None:
            target_value = builder.assignment_for(variable_name, resolved_step)
        if target_value is None:
            target_value = 0
        root = builder.belief_root(variable_name, resolved_step, int(target_value))
        creator = BCTCreator(builder.graph, root)
        creator.visualize_bct(show=show, save_path=savepath, verbose=verbose)
        return creator

    @staticmethod
    def _resolve_bct_iteration(
        iteration: int | None, steps_back: int | None, total_steps: int
    ) -> int:
        """Resolve the effective iteration index for BCT visualization."""
        if total_steps <= 0:
            return 0

        resolved = iteration if iteration is not None else -1

        if steps_back is not None:
            if steps_back <= 0:
                raise ValueError("steps_back must be positive")
            resolved = max(0, total_steps - steps_back)

        if resolved < 0:
            resolved = max(0, total_steps + resolved)

        if resolved >= total_steps:
            resolved = total_steps - 1

        return resolved

    def _ensure_bct_builder(self) -> SnapshotBCTBuilder:
        if self._bct_builder is None:
            self._bct_builder = SnapshotBCTBuilder(self._records)
        return self._bct_builder

    __all__ = ["SnapshotVisualizer"]
