"""Snapshot-driven Backtrack Cost Tree utilities.

This module implements Backtrack Cost Trees (BCTs) directly from per-step
snapshots. It follows the simulator-level specification used throughout the
project: message nodes reference specific Q or R message entries, edges are
labeled by the contribution weight (including damping coefficients), and leaves
reference concrete cost-table entries.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


MessageDirection = Literal["Q", "R"]


@dataclass(frozen=True)
class MessageKey:
    """Unique identifier for a message entry in the BCT graph."""

    direction: MessageDirection
    sender: str
    recipient: str
    iteration: int
    value_index: int

    def label(self) -> str:
        return f"{self.direction}:{self.sender}->{self.recipient}[{self.value_index}]@{self.iteration}"


@dataclass(frozen=True)
class CostKey:
    """Identifier for a cost-table entry."""

    factor: str
    assignment: Tuple[int, ...]

    def label(self) -> str:
        values = ", ".join(str(v) for v in self.assignment)
        return f"cost:{self.factor}({values})"


@dataclass(frozen=True)
class BeliefKey:
    """Identifier for a belief entry (variable value at a specific step)."""

    variable: str
    iteration: int
    value_index: int

    def label(self) -> str:
        return f"belief:{self.variable}[{self.value_index}]@{self.iteration}"


@dataclass
class BCTNode:
    """Node metadata stored inside the BCT graph."""

    key: Hashable
    kind: Literal["message", "cost", "belief"]
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BCTGraph:
    """Directed acyclic graph capturing message/cost dependencies."""

    def __init__(self) -> None:
        self.nodes: Dict[Hashable, BCTNode] = {}
        self._edges: Dict[Hashable, Dict[Hashable, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Node creation helpers
    # ------------------------------------------------------------------
    def ensure_message_node(self, key: MessageKey, **metadata: Any) -> MessageKey:
        node_meta = dict(metadata)
        node_meta.setdefault("iteration", getattr(key, "iteration", 0))
        node_meta.setdefault("message_role", "variable" if key.direction == "Q" else "factor")
        if key not in self.nodes:
            self.nodes[key] = BCTNode(key=key, kind="message", label=key.label(), metadata=node_meta)
        else:
            self.nodes[key].metadata.update(node_meta)
        return key

    def ensure_cost_node(self, key: CostKey, **metadata: Any) -> CostKey:
        node_meta = dict(metadata)
        node_meta.setdefault("iteration", metadata.get("iteration", 0))
        if key not in self.nodes:
            self.nodes[key] = BCTNode(key=key, kind="cost", label=key.label(), metadata=node_meta)
        else:
            self.nodes[key].metadata.update(node_meta)
        return key

    def ensure_belief_node(self, key: BeliefKey, **metadata: Any) -> BeliefKey:
        node_meta = dict(metadata)
        node_meta.setdefault("iteration", getattr(key, "iteration", 0))
        if key not in self.nodes:
            self.nodes[key] = BCTNode(key=key, kind="belief", label=key.label(), metadata=node_meta)
        else:
            self.nodes[key].metadata.update(node_meta)
        return key

    # ------------------------------------------------------------------
    # Graph edges
    # ------------------------------------------------------------------
    def add_edge(self, source: Hashable, target: Hashable, weight: float) -> None:
        if abs(weight) <= 0.0:
            return
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both source and target nodes must exist before adding an edge")
        bucket = self._edges[source]
        bucket[target] = bucket.get(target, 0.0) + float(weight)

    def children(self, key: Hashable) -> Dict[Hashable, float]:
        return self._edges.get(key, {})

    def reachable_nodes(self, root: Hashable) -> List[Hashable]:
        visited: set[Hashable] = set()
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for child in self.children(node):
                queue.append(child)
        return list(visited)


@dataclass
class _FactorContext:
    cost_table: np.ndarray
    labels: List[str]
    aggregate: np.ndarray
    broadcasts: List[np.ndarray]
    q_vectors: Dict[str, np.ndarray]


class SnapshotBCTBuilder:
    """Construct BCT DAGs directly from a sequence of snapshots."""

    def __init__(
        self,
        snapshots: Sequence[Any],
        *,
        tolerance: float = 1e-9,
        max_minimizers: int = 128,
    ) -> None:
        if not snapshots:
            raise ValueError("Snapshots are required to build a BCT")
        self._records = [rec.data if hasattr(rec, "data") else rec for rec in snapshots]
        self._records.sort(key=lambda rec: rec.step)
        self.tolerance = tolerance
        self.max_minimizers = max_minimizers
        self.graph = BCTGraph()
        self._factor_cache: Dict[Tuple[int, str], Optional[_FactorContext]] = {}
        self._steps = [rec.step for rec in self._records]
        self._snapshots_by_step = {rec.step: rec for rec in self._records}
        self._assignments_by_step = {rec.step: dict(getattr(rec, "assignments", {})) for rec in self._records}
        self._build()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def resolve_step(self, iteration: Optional[int], steps_back: Optional[int]) -> int:
        if iteration is not None:
            if iteration not in self._snapshots_by_step:
                raise ValueError(f"Snapshot for step {iteration} not available")
            return iteration
        total = len(self._steps)
        if total == 0:
            raise ValueError("Snapshots are empty")
        if steps_back is None:
            return self._steps[-1]
        if steps_back <= 0:
            raise ValueError("steps_back must be positive")
        index = max(0, total - steps_back)
        return self._steps[index]

    def assignment_for(self, variable: str, step: int) -> Optional[int]:
        return self._assignments_by_step.get(step, {}).get(variable)

    def belief_root(self, variable: str, step: int, value_index: int) -> BeliefKey:
        key = BeliefKey(variable=variable, iteration=step, value_index=value_index)
        if key not in self.graph.nodes:
            raise ValueError(f"Belief node for {variable} at step {step} is not available")
        return key

    # ------------------------------------------------------------------
    def _build(self) -> None:
        prev_snapshot: Optional[Any] = None
        for snapshot in self._records:
            self._build_q_nodes(snapshot, prev_snapshot)
            self._build_r_nodes(snapshot)
            self._build_beliefs(snapshot)
            prev_snapshot = snapshot

    # ------------------------------------------------------------------
    def _build_q_nodes(self, snapshot: Any, prev_snapshot: Optional[Any]) -> None:
        lambda_coeff = float(getattr(snapshot, "lambda_", 0.0) or 0.0)
        lambda_coeff = min(max(lambda_coeff, 0.0), 1.0)
        prev_step = getattr(prev_snapshot, "step", None)
        for (var_name, factor_name), values in snapshot.Q.items():
            arr = np.asarray(values, dtype=float).ravel()
            for value_index, _ in enumerate(arr):
                key = MessageKey("Q", var_name, factor_name, snapshot.step, value_index)
                node_meta = {
                    "variable": var_name,
                    "factor": factor_name,
                    "value": float(arr[value_index]) if value_index < len(arr) else 0.0,
                }
                self.graph.ensure_message_node(key, **node_meta)
                if lambda_coeff > 0.0 and prev_step is not None:
                    prev_key = MessageKey("Q", var_name, factor_name, prev_step, value_index)
                    if prev_key in self.graph.nodes:
                        self.graph.add_edge(key, prev_key, lambda_coeff)
                weight = 1.0 - lambda_coeff
                if weight <= 0.0 or prev_step is None:
                    continue
                for neighbor in snapshot.N_var.get(var_name, []):
                    if neighbor == factor_name:
                        continue
                    prev_array = None
                    if prev_snapshot is not None:
                        prev_array = prev_snapshot.R.get((neighbor, var_name))
                    if prev_array is None:
                        continue
                    prev_arr = np.asarray(prev_array, dtype=float).ravel()
                    if value_index >= len(prev_arr):
                        continue
                    child_key = MessageKey("R", neighbor, var_name, prev_step, value_index)
                    child_value = float(prev_arr[value_index])
                    self.graph.ensure_message_node(
                        child_key,
                        variable=var_name,
                        factor=neighbor,
                        value=child_value,
                    )
                    self.graph.add_edge(key, child_key, weight)

    # ------------------------------------------------------------------
    def _build_r_nodes(self, snapshot: Any) -> None:
        for (factor_name, var_name), values in snapshot.R.items():
            context = self._factor_context(snapshot, factor_name)
            if context is None:
                # Create placeholder node so belief edges can attach even if cost tables were missing
                arr = np.asarray(values, dtype=float).ravel()
                for value_index, value in enumerate(arr):
                    key = MessageKey("R", factor_name, var_name, snapshot.step, value_index)
                    self.graph.ensure_message_node(
                        key,
                        variable=var_name,
                        factor=factor_name,
                        value=float(value),
                    )
                continue
            labels = context.labels
            if var_name not in labels:
                continue
            axis = labels.index(var_name)
            r_arr = np.asarray(values, dtype=float).ravel()
            domain = min(len(r_arr), context.cost_table.shape[axis])
            exclusion = context.broadcasts[axis]
            reduced = context.aggregate - exclusion
            for value_index in range(domain):
                key = MessageKey("R", factor_name, var_name, snapshot.step, value_index)
                node_meta = {
                    "variable": var_name,
                    "factor": factor_name,
                    "value": float(r_arr[value_index]) if value_index < len(r_arr) else 0.0,
                }
                self.graph.ensure_message_node(key, **node_meta)
                slice_view = np.take(reduced, indices=value_index, axis=axis)
                assignments = self._enumerate_minimizers(
                    slice_view,
                    axis=axis,
                    axis_value=value_index,
                    shape=context.cost_table.shape,
                )
                if len(assignments) > 1:
                    self.graph.nodes[key].metadata["has_ties"] = True
                for assignment in assignments:
                    assignment_tuple: Tuple[int, ...] = tuple(int(x) for x in assignment)
                    cost_key = CostKey(factor=factor_name, assignment=assignment_tuple)
                    assignment_map = {labels[i]: assignment_tuple[i] for i in range(len(labels))}
                    cost_value = float(context.cost_table[assignment_tuple])
                    cost_metadata = {
                        "factor": factor_name,
                        "assignment": assignment_map,
                        "iteration": snapshot.step,
                        "value": cost_value,
                    }
                    self.graph.ensure_cost_node(cost_key, **cost_metadata)
                    self.graph.add_edge(key, cost_key, 1.0)
                    for idx, neighbor in enumerate(labels):
                        if neighbor == var_name:
                            continue
                        assigned_value = assignment_tuple[idx]
                        child_key = MessageKey("Q", neighbor, factor_name, snapshot.step, assigned_value)
                        child_value_vector = snapshot.Q.get((neighbor, factor_name))
                        value_payload = 0.0
                        if child_value_vector is not None:
                            arr_child = np.asarray(child_value_vector, dtype=float).ravel()
                            if assigned_value < len(arr_child):
                                value_payload = float(arr_child[assigned_value])
                        self.graph.ensure_message_node(
                            child_key,
                            variable=neighbor,
                            factor=factor_name,
                            value=value_payload,
                        )
                        self.graph.add_edge(key, child_key, 1.0)

    # ------------------------------------------------------------------
    def _build_beliefs(self, snapshot: Any) -> None:
        assignments = getattr(snapshot, "assignments", {}) or {}
        for var_name in snapshot.dom.keys():
            value_index = assignments.get(var_name)
            if value_index is None:
                belief = snapshot.beliefs.get(var_name)
                if belief is not None and len(belief):
                    value_index = int(np.argmin(np.asarray(belief, dtype=float)))
                else:
                    value_index = 0
            belief_value = None
            belief_arr = snapshot.beliefs.get(var_name)
            if belief_arr is not None:
                belief_np = np.asarray(belief_arr, dtype=float)
                if int(value_index) < len(belief_np):
                    belief_value = float(belief_np[int(value_index)])
            key = BeliefKey(variable=var_name, iteration=snapshot.step, value_index=int(value_index))
            extra_meta = {}
            if belief_value is not None:
                extra_meta["value"] = belief_value
            self.graph.ensure_belief_node(key, **extra_meta)
            for factor_name in snapshot.N_var.get(var_name, []):
                r_values = snapshot.R.get((factor_name, var_name))
                if r_values is None:
                    continue
                arr = np.asarray(r_values, dtype=float).ravel()
                if value_index >= len(arr):
                    continue
                child_key = MessageKey("R", factor_name, var_name, snapshot.step, int(value_index))
                child_value = float(arr[int(value_index)])
                self.graph.ensure_message_node(
                    child_key,
                    variable=var_name,
                    factor=factor_name,
                    value=child_value,
                )
                self.graph.add_edge(key, child_key, 1.0)

    # ------------------------------------------------------------------
    def _factor_context(self, snapshot: Any, factor_name: str) -> Optional[_FactorContext]:
        cache_key = (snapshot.step, factor_name)
        if cache_key in self._factor_cache:
            return self._factor_cache[cache_key]
        cost_table = snapshot.cost_tables.get(factor_name)
        labels = snapshot.cost_labels.get(factor_name)
        if cost_table is None or not labels:
            self._factor_cache[cache_key] = None
            return None
        table = np.asarray(cost_table, dtype=float)
        agg = table.copy()
        broadcasts: List[np.ndarray] = []
        q_vectors: Dict[str, np.ndarray] = {}
        for axis, var_name in enumerate(labels):
            q_values = snapshot.Q.get((var_name, factor_name))
            if q_values is None:
                vector = np.zeros(table.shape[axis], dtype=float)
            else:
                vector = np.asarray(q_values, dtype=float).ravel()
            if vector.size != table.shape[axis]:
                vector = np.resize(vector, table.shape[axis])
            broadcast = vector.reshape(
                [table.shape[i] if i == axis else 1 for i in range(table.ndim)]
            )
            broadcasts.append(broadcast)
            q_vectors[var_name] = vector
            agg = agg + broadcast
        context = _FactorContext(cost_table=table, labels=list(labels), aggregate=agg, broadcasts=broadcasts, q_vectors=q_vectors)
        self._factor_cache[cache_key] = context
        return context

    # ------------------------------------------------------------------
    def _enumerate_minimizers(
        self,
        slice_array: np.ndarray,
        *,
        axis: int,
        axis_value: int,
        shape: Tuple[int, ...],
    ) -> List[Tuple[int, ...]]:
        arr = np.asarray(slice_array, dtype=float)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return [tuple(axis_value if i == axis else 0 for i in range(len(shape)))]
        min_value = float(np.min(flat))
        tolerance = max(self.tolerance, 0.0)
        indices = np.argwhere(np.isclose(arr, min_value, atol=tolerance))
        if indices.size == 0:
            indices = np.array([[0] * arr.ndim], dtype=int)
        assignments: List[Tuple[int, ...]] = []
        for idx_tuple in indices[: self.max_minimizers]:
            full: List[int] = []
            cursor = 0
            for dim in range(len(shape)):
                if dim == axis:
                    full.append(int(axis_value))
                else:
                    full.append(int(idx_tuple[cursor]))
                    cursor += 1
            assignments.append(tuple(full))
        return assignments


class BCTCreator:
    """Convenience wrapper around a BCTGraph for visualization and queries."""

    def __init__(self, graph: BCTGraph, root: Hashable):
        self.graph = graph
        self.root = root

    # ------------------------------------------------------------------
    def visualize_bct(self, *, show: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        reachable = self.graph.reachable_nodes(self.root)
        if not reachable:
            raise ValueError("BCT root has no reachable nodes")
        positions: Dict[Hashable, Tuple[float, float]] = {}
        labels: Dict[Hashable, str] = {}
        node_meta: Dict[Hashable, BCTNode] = {node: self.graph.nodes[node] for node in reachable}

        def _child_nodes(node_key: Hashable) -> List[Hashable]:
            children = [
                child for child in self.graph.children(node_key) if child in node_meta
            ]
            return sorted(children, key=lambda child: node_meta[child].label)

        layout_cache: Dict[Hashable, float] = {}
        depth_map: Dict[Hashable, int] = {}
        x_cursor = {"value": 0.0}

        def _assign(node_key: Hashable, depth: int) -> float:
            if node_key in layout_cache:
                depth_map[node_key] = min(depth_map[node_key], depth)
                return layout_cache[node_key]
            children = _child_nodes(node_key)
            if not children:
                x = float(x_cursor["value"])
                x_cursor["value"] += 1.0
            else:
                child_positions = [_assign(child, depth + 1) for child in children]
                x = float(sum(child_positions) / len(child_positions))
            node = node_meta[node_key]
            label_text = node.label
            value = node.metadata.get("value")
            should_show_value = (
                node.kind == "cost"
                or (
                    node.kind == "message"
                    and node.metadata.get("message_role") == "factor"
                )
                or node.kind == "belief"
            )
            if value is not None and should_show_value:
                label_text = f"{label_text}\n{value:.3f}"
            labels[node_key] = label_text
            depth_map[node_key] = depth
            positions[node_key] = (x, -self._depth_level(node, depth))
            layout_cache[node_key] = x
            return x

        _assign(self.root, 0)

        level_nodes: Dict[int, List[Hashable]] = defaultdict(list)
        for node, depth in depth_map.items():
            level_nodes[depth].append(node)

        min_spacing = 1.5
        for depth in sorted(level_nodes):
            ordered = sorted(level_nodes[depth], key=lambda n: positions[n][0])
            current_x: float | None = None
            for node in ordered:
                x, y = positions[node]
                if current_x is None:
                    target = x
                else:
                    target = max(x, current_x + min_spacing)
                positions[node] = (target, y)
                current_x = target

        if positions:
            xs = [x for x, _ in positions.values()]
            center_shift = (min(xs) + max(xs)) / 2.0
            for node, (x, y) in positions.items():
                positions[node] = (x - center_shift, y)

        base_graph = nx.DiGraph()
        for node in reachable:
            base_graph.add_node(node)
            for child, weight in self.graph.children(node).items():
                if child in reachable:
                    base_graph.add_edge(node, child, weight=weight)
        visible_nodes = [n for n in base_graph.nodes() if node_meta[n].kind != "cost"]
        visible_positions = {n: positions[n] for n in visible_nodes}
        visible_labels = {n: labels[n] for n in visible_nodes}
        G = nx.DiGraph()
        G.add_nodes_from(visible_nodes)
        for u, v, data in base_graph.edges(data=True):
            if u in visible_nodes and v in visible_nodes:
                G.add_edge(u, v, **data)

        total_levels = max(depth_map.values(), default=0) + 1
        fig_height = max(8.0, total_levels * 1.2)
        fig_width = max(14.0, total_levels * 1.6)
        fig = plt.figure(figsize=(fig_width, fig_height))
        raw_edge_labels = nx.get_edge_attributes(G, "weight")
        edge_labels = {
            edge: weight
            for edge, weight in raw_edge_labels.items()
            if abs(weight - 1.0) > 1e-9
        }

        ax = plt.gca()
        if visible_positions:
            min_x = min(x for x, _ in visible_positions.values())
            depth_to_y: Dict[int, List[float]] = defaultdict(list)
            for node in visible_nodes:
                depth_to_y[depth_map[node]].append(visible_positions[node][1])
            for depth, ys in depth_to_y.items():
                level_y = sum(ys) / len(ys)
                ax.axhline(y=level_y, color="#dddddd", linestyle="--", linewidth=0.7, zorder=0)
                ax.text(
                    min_x - 0.5,
                    level_y,
                    f"Level {depth + 1}",
                    fontsize=8,
                    va="center",
                    ha="right",
                    color="#666666",
                )

        nx.draw_networkx_edges(G, visible_positions, arrows=True, arrowsize=18)
        belief_nodes = [n for n in G.nodes() if node_meta[n].kind == "belief"]
        variable_nodes = [
            n
            for n in G.nodes()
            if node_meta[n].kind == "message"
            and node_meta[n].metadata.get("message_role") == "variable"
        ]
        factor_nodes = [
            n
            for n in G.nodes()
            if node_meta[n].kind == "message"
            and node_meta[n].metadata.get("message_role") == "factor"
        ]
        nx.draw_networkx_nodes(
            G,
            visible_positions,
            nodelist=belief_nodes,
            node_color="#FFD966",
            node_shape="o",
            node_size=2600,
        )
        nx.draw_networkx_nodes(
            G,
            visible_positions,
            nodelist=variable_nodes,
            node_color="#6FA8DC",
            node_shape="o",
            node_size=2600,
        )
        nx.draw_networkx_nodes(
            G,
            visible_positions,
            nodelist=factor_nodes,
            node_color="#93C47D",
            node_shape="s",
            node_size=2400,
        )
        nx.draw_networkx_labels(G, visible_positions, visible_labels, font_size=7)
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, visible_positions, edge_labels=edge_labels, font_color="gray"
            )
        plt.title("Backtrack Cost Tree")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    def cost_contributions(self) -> Dict[CostKey, float]:
        contributions: Dict[CostKey, float] = defaultdict(float)
        stack: List[Tuple[Hashable, float]] = [(self.root, 1.0)]
        while stack:
            node_key, coeff = stack.pop()
            node = self.graph.nodes.get(node_key)
            if node is None:
                continue
            if node.kind == "cost" and isinstance(node.key, CostKey):
                contributions[node.key] += coeff
                continue
            for child, weight in self.graph.children(node_key).items():
                stack.append((child, coeff * weight))
        return contributions

    # ------------------------------------------------------------------
    @staticmethod
    def _depth_level(node: BCTNode, depth: int) -> float:
        offset = 0.0
        if node.kind == "message":
            offset = 1.0 if node.metadata.get("message_role") == "factor" else 2.0
        elif node.kind == "cost":
            offset = 3.0
        return depth * 4.0 + offset

__all__ = ["BCTGraph", "BCTCreator", "SnapshotBCTBuilder", "MessageKey", "BeliefKey", "CostKey"]
