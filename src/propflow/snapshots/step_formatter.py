"""Step-by-step formatter for belief propagation simulation output.

This module provides tools to format BP simulation steps in an Excel-like
tabular format, showing Q/R messages, cost tables, assignments, and solution
costs per iteration.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Literal, Tuple

import numpy as np
import re

from .types import EngineSnapshot


def _letter_label(idx: int) -> str:
    """Convert numeric index to letter label (0->a, 1->b, etc.)."""
    return chr(ord('a') + idx)


def _format_array(arr: np.ndarray, precision: int = 3) -> str:
    """Format array as comma-separated values."""
    values = [f"{v:.{precision}g}" if isinstance(v, float) else str(int(v)) 
              for v in arr.flatten()]
    return ", ".join(values)


def _normalize_min_zero(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return arr - float(np.min(arr))


def _render_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    rows_list = [list(row) for row in rows]
    if not rows_list:
        return []
    col_widths = [len(header) for header in headers]
    for row in rows_list:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def _format_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    header_row = _format_row(headers)
    sep_row = "|-" + "-|-".join("-" * width for width in col_widths) + "-|"
    body_rows = [_format_row(row) for row in rows_list]
    return [header_row, sep_row, *body_rows]


def _infer_route_order(variable_names: Sequence[str]) -> List[str]:
    def _parse(name: str) -> tuple[int | None, str]:
        match = re.search(r"(\\d+)$", name)
        if not match:
            return (None, name)
        return (int(match.group(1)), name[: match.start()])

    parsed = []
    for name in variable_names:
        number, prefix = _parse(name)
        parsed.append((prefix.lower(), number, name))

    if any(number is not None for _, number, _ in parsed):
        parsed.sort(key=lambda item: (item[0], item[1] is None, item[1] or 0, item[2]))
    else:
        parsed.sort(key=lambda item: item[2])

    return [name for _, _, name in parsed]


class StepByStepFormatter:
    """Formats BP simulation steps in Excel-like tabular format.
    
    This class takes a sequence of EngineSnapshots and provides methods to
    format them as readable step-by-step output showing:
    - Cost tables for all factors
    - Q messages (variable -> factor) per iteration
    - R messages (factor -> variable) per iteration
    - Variable assignments and beliefs per iteration
    - Solution cost per iteration
    
    Example:
        >>> from propflow.snapshots import StepByStepFormatter
        >>> formatter = StepByStepFormatter(engine.snapshot_manager.snapshots)
        >>> print(formatter.format_all_steps())
    """
    
    def __init__(
        self,
        snapshots: Sequence[EngineSnapshot],
        normalize_messages: bool = True,
        route_filter: Literal["both", "cw", "ccw"] = "both",
        route_order: Optional[Sequence[str]] = None,
        ignore_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        """Initialize formatter with snapshot records.
        
        Args:
            snapshots: A sequence of EngineSnapshot objects from a BP simulation.
            normalize_messages: If True, normalize Q/R messages by subtracting
                the minimum value per message before formatting.
            route_filter: Filter messages by route direction (``"cw"``,
                ``"ccw"``, or ``"both"``). Defaults to ``"both"``.
            route_order: Optional explicit variable ordering used to infer
                clockwise/counter-clockwise direction. If omitted, the order
                is inferred from variable names.
            ignore_pairs: Optional list of ``(sender, recipient)`` message pairs
                to omit from both Q and R output.
            
        Raises:
            ValueError: If snapshots is empty.
        """
        if not snapshots:
            raise ValueError("Cannot format empty snapshot sequence")
        
        self._snapshots = list(snapshots)
        self._sorted_steps = sorted(s.step for s in self._snapshots)
        self._step_to_snapshot = {s.step: s for s in self._snapshots}
        self._normalize_messages = normalize_messages
        self._route_filter = route_filter.lower()
        self._ignore_pairs = set(ignore_pairs or [])
        if self._route_filter not in {"both", "cw", "ccw"}:
            raise ValueError("route_filter must be 'both', 'cw', or 'ccw'")
        
        # extract variable and factor names from first snapshot
        first = self._snapshots[0]
        self._variables = sorted(first.dom.keys())
        self._factors = sorted(first.cost_tables.keys()) if first.cost_tables else []

        if route_order is None:
            self._route_order = _infer_route_order(self._variables)
        else:
            missing = set(self._variables) - set(route_order)
            extra = set(route_order) - set(self._variables)
            if missing:
                raise ValueError(
                    f"route_order is missing variables: {', '.join(sorted(missing))}"
                )
            if extra:
                raise ValueError(
                    f"route_order contains unknown variables: {', '.join(sorted(extra))}"
                )
            self._route_order = list(route_order)

        self._route_index = {name: idx for idx, name in enumerate(self._route_order)}
        if first.cost_labels:
            self._factor_neighbors = {
                name: list(labels) for name, labels in first.cost_labels.items()
            }
        else:
            self._factor_neighbors = {
                name: list(neighbors) for name, neighbors in first.N_fac.items()
            }
    
    @property
    def variables(self) -> List[str]:
        """List of variable names in the problem."""
        return list(self._variables)
    
    @property
    def factors(self) -> List[str]:
        """List of factor names in the problem."""
        return list(self._factors)
    
    @property
    def steps(self) -> List[int]:
        """List of step numbers available."""
        return list(self._sorted_steps)

    def _route_direction(self, var_name: str, factor_name: str) -> Literal["cw", "ccw", "both"]:
        neighbors = self._factor_neighbors.get(factor_name, [])
        if var_name not in neighbors or len(neighbors) != 2:
            return "both"

        other = neighbors[0] if neighbors[1] == var_name else neighbors[1]
        if var_name not in self._route_index or other not in self._route_index:
            return "both"

        total = len(self._route_order)
        if total < 2:
            return "both"

        var_idx = self._route_index[var_name]
        other_idx = self._route_index[other]
        if (var_idx + 1) % total == other_idx:
            return "cw"
        if (var_idx - 1) % total == other_idx:
            return "ccw"
        return "both"

    def _route_allows(self, var_name: str, factor_name: str) -> bool:
        if self._route_filter == "both":
            return True
        return self._route_direction(var_name, factor_name) == self._route_filter

    def _other_variable(self, var_name: str, factor_name: str) -> Optional[str]:
        neighbors = self._factor_neighbors.get(factor_name, [])
        if var_name not in neighbors or len(neighbors) != 2:
            return None
        return neighbors[0] if neighbors[1] == var_name else neighbors[1]

    def _route_allows_message(
        self,
        *,
        kind: Literal["Q", "R"],
        sender: str,
        recipient: str,
    ) -> bool:
        if self._route_filter == "both":
            return True
        if kind == "Q":
            return self._route_direction(sender, recipient) == self._route_filter
        other = self._other_variable(recipient, sender)
        if other is None:
            return False
        return self._route_direction(other, sender) == self._route_filter

    def _iter_messages(
        self,
        messages: Dict[Tuple[str, str], np.ndarray],
        *,
        kind: Literal["Q", "R"],
    ) -> Iterable[Tuple[str, str, str]]:
        for (sender, recipient), data in sorted(messages.items()):
            if (sender, recipient) in self._ignore_pairs:
                continue
            if not self._route_allows_message(kind=kind, sender=sender, recipient=recipient):
                continue
            arr = np.asarray(data)
            if self._normalize_messages:
                arr = _normalize_min_zero(arr)
            formatted = _format_array(arr)
            yield sender, recipient, formatted

    @property
    def domain_size(self) -> int:
        """Domain size of variables (from first snapshot)."""
        first = self._snapshots[0]
        if first.dom:
            return len(next(iter(first.dom.values())))
        return 0
    
    def format_cost_tables(self, use_letters: bool = True) -> str:
        """Format cost tables for all factors.
        
        Args:
            use_letters: If True, use letter labels (a, b, ...) instead of numbers.
            
        Returns:
            Formatted string showing all cost tables.
        """
        first = self._snapshots[0]
        if not first.cost_tables:
            return "No cost tables available\n"
        
        lines = ["=" * 60, "COST TABLES", "=" * 60, ""]
        
        for factor_name in self._factors:
            table = first.cost_tables.get(factor_name)
            labels = first.cost_labels.get(factor_name, [])
            
            if table is None:
                continue
            
            lines.append(f"Factor: {factor_name.upper()}")
            if labels:
                lines.append(f"  Connected variables: {', '.join(labels)}")
            
            # format as 2D table if binary factor
            if table.ndim == 2:
                domain = table.shape[0]
                row_labels = [_letter_label(i) if use_letters else str(i) 
                             for i in range(domain)]
                col_labels = row_labels.copy()
                
                # header row
                header = "     " + "  ".join(f"{lbl:>5}" for lbl in col_labels)
                lines.append(header)
                
                # data rows
                for i, row_label in enumerate(row_labels):
                    row_vals = "  ".join(f"{table[i, j]:>5.3g}" for j in range(domain))
                    lines.append(f"  {row_label}  {row_vals}")
            else:
                # 1D unary factor
                domain = table.shape[0]
                labels_str = [_letter_label(i) if use_letters else str(i) 
                              for i in range(domain)]
                lines.append("  " + "  ".join(f"{lbl}: {table[i]:.3g}" 
                                              for i, lbl in enumerate(labels_str)))
            
            lines.append("")
        
        return "\n".join(lines)
    
    def format_iteration(
        self,
        step: int,
        use_letters: bool = True,
        show: Literal["text", "table"] = "text",
    ) -> str:
        """Format Q/R messages, assignments, and cost for one iteration.
        
        Args:
            step: The step number to format.
            use_letters: If True, use letter labels for domain values.
            show: ``"text"`` (default) prints the existing format; ``"table"``
                renders Q/R messages in a tabular layout.
            
        Returns:
            Formatted string for the iteration.
        """
        if step not in self._step_to_snapshot:
            return f"Step {step} not found\n"
        
        snapshot = self._step_to_snapshot[step]
        lines = ["-" * 60, f"ITERATION {step}", "-" * 60, ""]
        
        # q messages (variable -> factor)
        lines.append("Q Messages (Variable -> Factor):")
        if show == "table":
            q_rows = list(self._iter_messages(snapshot.Q, kind="Q"))
            if q_rows:
                lines.extend(_render_table(["Sender", "Recipient", "Message"], q_rows))
            else:
                lines.append("  (no Q messages)")
        else:
            wrote_q = False
            for sender, recipient, formatted in self._iter_messages(snapshot.Q, kind="Q"):
                lines.append(f"  {sender} -> {recipient}: [{formatted}]")
                wrote_q = True
        
        if show != "table" and not wrote_q:
            lines.append("  (no Q messages)")
        
        lines.append("")
        
        # r messages (factor -> variable)
        lines.append("R Messages (Factor -> Variable):")
        if show == "table":
            r_rows = list(self._iter_messages(snapshot.R, kind="R"))
            if r_rows:
                lines.extend(_render_table(["Sender", "Recipient", "Message"], r_rows))
            else:
                lines.append("  (no R messages)")
        else:
            wrote_r = False
            for sender, recipient, formatted in self._iter_messages(snapshot.R, kind="R"):
                lines.append(f"  {sender} -> {recipient}: [{formatted}]")
                wrote_r = True
        
        if show != "table" and not wrote_r:
            lines.append("  (no R messages)")
        
        lines.append("")
        
        # assignments
        lines.append("Assignments:")
        for var_name in self._variables:
            assignment = snapshot.assignments.get(var_name, "?")
            if use_letters and isinstance(assignment, int):
                assignment_label = _letter_label(assignment)
            else:
                assignment_label = str(assignment)
            lines.append(f"  {var_name} = {assignment_label}")
        
        lines.append("")
        
        # beliefs (optional)
        if snapshot.beliefs:
            lines.append("Beliefs:")
            for var_name in self._variables:
                belief = snapshot.beliefs.get(var_name)
                if belief is not None:
                    formatted = _format_array(np.asarray(belief))
                    lines.append(f"  {var_name}: [{formatted}]")
            lines.append("")
        
        # global cost
        if snapshot.global_cost is not None:
            lines.append(f"Solution Cost: {snapshot.global_cost:.3g}")
        else:
            lines.append("Solution Cost: (not computed)")
        
        # damping factor
        if snapshot.lambda_ != 0:
            lines.append(f"Damping Factor: {snapshot.lambda_:.3g}")
        
        lines.append("")
        return "\n".join(lines)
    
    def format_all_steps(
        self,
        include_cost_tables: bool = True,
        show: Literal["text", "table"] = "text",
    ) -> str:
        """Return complete step-by-step output.
        
        Args:
            include_cost_tables: If True, include cost tables at the beginning.
            show: ``"text"`` (default) prints the existing format; ``"table"``
                renders Q/R messages in a tabular layout.
            
        Returns:
            Complete formatted output for all iterations.
        """
        parts = []
        
        # header
        parts.append("=" * 60)
        parts.append("BELIEF PROPAGATION STEP-BY-STEP OUTPUT")
        parts.append(f"Variables: {', '.join(self._variables)}")
        parts.append(f"Factors: {', '.join(self._factors)}")
        parts.append(f"Domain size: {self.domain_size}")
        parts.append(f"Total iterations: {len(self._sorted_steps)}")
        parts.append("=" * 60)
        parts.append("")
        
        # cost tables
        if include_cost_tables:
            parts.append(self.format_cost_tables())
        
        # iterations
        for step in self._sorted_steps:
            parts.append(self.format_iteration(step, show=show))
        
        return "\n".join(parts)
    
    def format_summary(self) -> str:
        """Return a compact summary of the simulation.
        
        Returns:
            Summary string with initial/final costs and assignments.
        """
        lines = ["SIMULATION SUMMARY", "-" * 40]
        
        if self._sorted_steps:
            first_snapshot = self._step_to_snapshot[self._sorted_steps[0]]
            last_snapshot = self._step_to_snapshot[self._sorted_steps[-1]]
            
            initial_cost = first_snapshot.global_cost
            final_cost = last_snapshot.global_cost
            
            lines.append(f"Total iterations: {len(self._sorted_steps)}")
            
            if initial_cost is not None:
                lines.append(f"Initial cost: {initial_cost:.3g}")
            if final_cost is not None:
                lines.append(f"Final cost: {final_cost:.3g}")
            if initial_cost is not None and final_cost is not None:
                improvement = initial_cost - final_cost
                lines.append(f"Cost improvement: {improvement:.3g}")
            
            lines.append("")
            lines.append("Final assignments:")
            for var_name in self._variables:
                assignment = last_snapshot.assignments.get(var_name, "?")
                lines.append(f"  {var_name} = {_letter_label(assignment) if isinstance(assignment, int) else assignment}")
        
        return "\n".join(lines)


__all__ = ["StepByStepFormatter"]
