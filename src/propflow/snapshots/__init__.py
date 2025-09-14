"""
Per-step snapshot capture and analysis for BP engines.

This module provides a small, modular API to record a lightweight snapshot of
the engine state at each step (when enabled), and to compute Jacobian blocks
and cycle metrics for focused, iteration-level analysis.

Top-level exports:
- SnapshotConfig: toggles and limits for capture/analysis
- SnapshotRecord: container of snapshot data + Jacobians + metrics
- SnapshotManager: attaches to engine and records per-step snapshots
"""

from .types import SnapshotsConfig, SnapshotRecord
from .manager import SnapshotManager
from . import utils as snapshot_utils

__all__ = [
    "SnapshotsConfig",
    "SnapshotRecord",
    "SnapshotManager",
    "snapshot_utils",
]
