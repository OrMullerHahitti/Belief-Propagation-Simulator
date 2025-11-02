from __future__ import annotations

import heapq
from collections import deque
from typing import Deque, Generic, List, Tuple, TypeVar

from .node import Node

S = TypeVar("S")
A = TypeVar("A")


class PriorityFrontier(Generic[S, A]):
    """Heap-backed best-first frontier for A* / greedy searches."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, Node[S, A]]] = []
        self._counter = 0

    def push(self, node: Node[S, A]) -> None:
        heapq.heappush(self._heap, (node.f, self._counter, node))
        self._counter += 1

    def pop(self) -> Node[S, A]:
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._heap)


class FIFOFrontier(Generic[S, A]):
    """Simple queue frontier for BFS-style searches."""

    def __init__(self) -> None:
        self._queue: Deque[Node[S, A]] = deque()

    def push(self, node: Node[S, A]) -> None:
        self._queue.append(node)

    def pop(self) -> Node[S, A]:
        return self._queue.popleft()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)


class BeamFrontier(Generic[S, A]):
    """Width-limited frontier that keeps the best ``width`` nodes."""

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError("Beam width must be a positive integer.")
        self.width = width
        self._buffer: List[Node[S, A]] = []

    def push(self, node: Node[S, A]) -> None:
        self._buffer.append(node)
        self._buffer.sort(key=lambda cand: cand.f)
        if len(self._buffer) > self.width:
            self._buffer.pop()

    def pop(self) -> Node[S, A]:
        return self._buffer.pop(0)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)
