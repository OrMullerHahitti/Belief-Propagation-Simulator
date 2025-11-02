from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Hashable, Iterable, Optional, Tuple, TypeVar

from .frontier import PriorityFrontier
from .node import Node
from .protocols import (
    CostAccumulationPolicy,
    DuplicateDetectionPolicy,
    ExpansionPolicy,
    GoalPolicy,
    HeuristicPolicy,
    StateKeyFn,
)

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class EngineHooks:
    """No-op hook container. Users can subclass to intercept events."""

    def on_start(
        self, engine: "SearchEngine[S, A]"
    ) -> None:  # pragma: no cover - trivial
        return

    def on_expand(
        self, engine: "SearchEngine[S, A]", node: Node[S, A]
    ) -> None:  # pragma: no cover - trivial
        return

    def on_generate(
        self, engine: "SearchEngine[S, A]", node: Node[S, A]
    ) -> None:  # pragma: no cover - trivial
        return

    def on_goal(
        self, engine: "SearchEngine[S, A]", node: Node[S, A]
    ) -> None:  # pragma: no cover - trivial
        return

    def on_finish(
        self, engine: "SearchEngine[S, A]"
    ) -> None:  # pragma: no cover - trivial
        return


@dataclass
class EngineHistory:
    """Minimal history tracker mirroring the BP snapshot surface."""

    records: list[Dict[str, object]] = field(default_factory=list)
    step_count: int = 0

    def push_snapshot(self, payload: Dict[str, object]) -> None:
        snapshot = dict(payload)
        snapshot.setdefault("tick", self.step_count)
        self.records.append(snapshot)
        self.step_count += 1


class SearchEngine(Generic[S, A]):
    """
    Generic best-first search engine driven entirely by policies.

    The engine orchestrates expansion order (via ``frontier``), duplicate
    detection, and scoring. Hook and history objects reuse the familiar BP
    lifecycle events so analyzers can consume the same surface.
    """

    def __init__(
        self,
        *,
        expander: ExpansionPolicy[S, A, float],
        heuristic: HeuristicPolicy[S, float],
        goal: GoalPolicy[S],
        cost: CostAccumulationPolicy,
        duplicate: DuplicateDetectionPolicy[S],
        state_key: StateKeyFn[S],
        frontier=None,
        hooks: Optional[EngineHooks] = None,
        history: Optional[EngineHistory] = None,
    ) -> None:
        self.expander = expander
        self.heuristic = heuristic
        self.goal = goal
        self.cost = cost
        self.duplicate = duplicate
        self.state_key = state_key
        self.frontier = frontier or PriorityFrontier()
        self.hooks = hooks or EngineHooks()
        self.history = history or EngineHistory()
        self._nodes: Dict[Hashable, Node[S, A]] = {}

    # ------------------------------------------------------------------#
    # Lifecycle
    # ------------------------------------------------------------------#
    def run(self, start_state: S) -> Optional[Node[S, A]]:
        """Execute the search until goal found or frontier exhausted."""
        self.hooks.on_start(self)
        start_node = self._build_start_node(start_state)
        start_key = self.state_key(start_state)

        self.duplicate.record(start_key, start_node.g)
        self.frontier.push(start_node)
        self._register_node(start_key, start_node)
        self._snapshot({"event": "seed", "key": start_key, "f": start_node.f})

        best_goal: Optional[Node[S, A]] = None

        while len(self.frontier):
            current = self.frontier.pop()
            current_key = self.state_key(current.state)
            self.hooks.on_expand(self, current)
            self._snapshot(
                {
                    "event": "expand",
                    "key": current_key,
                    "g": current.g,
                    "f": current.f,
                    "frontier": len(self.frontier),
                }
            )

            if self.goal.is_goal(current.state):
                best_goal = current
                self.hooks.on_goal(self, current)
                self._snapshot(
                    {
                        "event": "goal",
                        "key": current_key,
                        "g": current.g,
                        "f": current.f,
                    }
                )
                break

            for action, next_state, step_cost in self.expander.expand(current.state):
                g_val = self.cost.g_update(current.g, float(step_cost))
                h_val = self.heuristic.h(next_state)
                f_val = self.cost.f_score(g_val, h_val)
                next_key = self.state_key(next_state)
                if not self.duplicate.better_path(next_key, g_val):
                    continue
                self.duplicate.record(next_key, g_val)
                child = Node(
                    state=next_state,
                    parent_key=current_key,
                    action=action,
                    g=g_val,
                    h=h_val,
                    f=f_val,
                    depth=current.depth + 1,
                )
                self._register_node(next_key, child)
                self.frontier.push(child)
                self.hooks.on_generate(self, child)
                self._snapshot(
                    {
                        "event": "generate",
                        "parent": current_key,
                        "key": next_key,
                        "action": action,
                        "g": g_val,
                        "f": f_val,
                    }
                )

        self.hooks.on_finish(self)
        self._snapshot(
            {
                "event": "finish",
                "frontier": len(self.frontier),
                "best_goal": self.state_key(best_goal.state) if best_goal else None,
            }
        )
        return best_goal

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def reconstruct_path(self, terminal: Node[S, A]) -> Iterable[Tuple[S, Optional[A]]]:
        """Yield ``(state, action)`` pairs from start to ``terminal``."""
        key = self.state_key(terminal.state)
        chain: list[Tuple[S, Optional[A]]] = []
        node = terminal
        while True:
            chain.append((node.state, node.action))
            if node.parent_key is None:
                break
            parent = self._nodes.get(node.parent_key)
            if parent is None:
                break
            key = node.parent_key
            node = parent
        return reversed(chain)

    def _build_start_node(self, start_state: S) -> Node[S, A]:
        h_val = self.heuristic.h(start_state)
        node = Node(
            state=start_state,
            parent_key=None,
            action=None,
            g=0.0,
            h=h_val,
            f=0.0,
            depth=0,
        )
        node.f = self.cost.f_score(node.g, node.h)
        return node

    def _register_node(self, key: Hashable, node: Node[S, A]) -> None:
        self._nodes[key] = node

    def _snapshot(self, payload: Dict[str, object]) -> None:
        self.history.push_snapshot(payload)


# ----------------------------------------------------------------------#
# Backward compatibility with legacy local-search engines
# ----------------------------------------------------------------------#
try:  # pragma: no cover - exercised indirectly
    from .legacy_engines import (  # type: ignore
        SearchEngine as LegacySearchEngine,
        DSAEngine,
        MGM2Engine,
        MGMEngine,
    )

    __all__ = [
        "SearchEngine",
        "EngineHooks",
        "EngineHistory",
        "LegacySearchEngine",
        "DSAEngine",
        "MGMEngine",
        "MGM2Engine",
    ]
except Exception:  # pragma: no cover - optional legacy support
    __all__ = ["SearchEngine", "EngineHooks", "EngineHistory"]
