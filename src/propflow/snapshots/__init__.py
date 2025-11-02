"""
Per-step snapshot capture and analysis for BP engines.

This module provides a small, modular API to record a lightweight snapshot of
the engine state at each step (when enabled), and to compute Jacobian blocks
and cycle metrics for focused, iteration-level analysis.

Top-level exports:
- EngineSnapshot: container of snapshot data + Jacobians + metrics
- SnapshotManager: attaches to engine and records per-step snapshots
- SnapshotAnalyzer: derive Jacobian matrices and convergence metrics from snapshots
- AnalysisReport: generate analysis reports and export results
- SnapshotVisualizer: plot belief argmin trajectories from snapshots
"""

from .types import EngineSnapshot, SnapshotRecord, Jacobians, CycleMetrics
from .manager import SnapshotManager
from .analyzer import SnapshotAnalyzer, AnalysisReport
from .visualizer import SnapshotVisualizer
from . import utils as snapshot_utils

__all__ = [
    "EngineSnapshot",
    "SnapshotRecord",
    "Jacobians",
    "CycleMetrics",
    "SnapshotManager",
    "SnapshotAnalyzer",
    "AnalysisReport",
    "SnapshotVisualizer",
    "snapshot_utils",
]
