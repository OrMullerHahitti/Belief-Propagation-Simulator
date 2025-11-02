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
        snapshot = build_snapshot_from_engine(step_index, step, engine)
        if getattr(engine, "use_bct_history", False):
            self.capture_bct_data(snapshot, engine)
        return snapshot

    def capture_bct_data(
        self, snapshot: EngineSnapshot, engine: Any
    ) -> None:  # pragma: no cover - hook
        """Optional hook to enrich a snapshot with BCT-specific data."""
        return None


__all__ = ["SnapshotManager"]
