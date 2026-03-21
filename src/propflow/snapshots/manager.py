"""Minimal snapshot manager.

This module exposes a streamlined `SnapshotManager` that captures an
`EngineSnapshot` for each engine step without retention policies or
automatic persistence.
"""

from __future__ import annotations

from typing import Any

from .builder import build_snapshot_from_engine
from .types import EngineSnapshot


class SnapshotManager:
    """Lightweight helper that captures a snapshot for a single engine step."""

    def capture_step(self, step_index: int, step: Any, engine: Any) -> EngineSnapshot:
        """Capture the engine state after a completed step."""
        return build_snapshot_from_engine(step_index, step, engine)


__all__ = ["SnapshotManager"]
