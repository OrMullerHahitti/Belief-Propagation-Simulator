from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

S = TypeVar("S")
A = TypeVar("A")


@dataclass(slots=True)
class Node(Generic[S, A]):
    """Lightweight container for states tracked by the search frontier."""

    state: S
    parent_key: Optional[object]
    action: Optional[A]
    g: float
    h: float
    f: float
    depth: int

