"""Convenience Helpers for Accessing Snapshot Artifacts.

This module provides a set of utility functions to easily retrieve snapshot
records and their computed artifacts (like Jacobians and cycle metrics) from
a simulation engine instance that has a `SnapshotManager` attached.
"""
from __future__ import annotations
from typing import Optional, Any
from .types import SnapshotRecord, Jacobians, CycleMetrics


def latest_snapshot(engine: Any) -> Optional[SnapshotRecord]:
    """Retrieves the most recent snapshot record from the engine.

    Args:
        engine: The simulation engine instance, expected to have a
            `latest_snapshot` method.

    Returns:
        The latest `SnapshotRecord` object, or `None` if no snapshots are available.
    """
    return getattr(engine, "latest_snapshot", lambda: None)()


def latest_jacobians(engine: Any) -> Optional[Jacobians]:
    """Retrieves the Jacobians from the most recent snapshot record.

    Args:
        engine: The simulation engine instance.

    Returns:
        The latest `Jacobians` object, or `None` if not available.
    """
    rec = latest_snapshot(engine)
    return getattr(rec, "jacobians", None) if rec else None


def latest_cycles(engine: Any) -> Optional[CycleMetrics]:
    """Retrieves the cycle metrics from the most recent snapshot record.

    Args:
        engine: The simulation engine instance.

    Returns:
        The latest `CycleMetrics` object, or `None` if not available.
    """
    rec = latest_snapshot(engine)
    return getattr(rec, "cycles", None) if rec else None


def latest_winners(engine: Any) -> Optional[dict]:
    """Retrieves the 'winners' dictionary from the most recent snapshot record.

    Args:
        engine: The simulation engine instance.

    Returns:
        The latest winners dictionary, or `None` if not available.
    """
    rec = latest_snapshot(engine)
    return getattr(rec, "winners", None) if rec else None


def get_snapshot(engine: Any, step_index: int) -> Optional[SnapshotRecord]:
    """Retrieves a snapshot record for a specific step index from the engine.

    Args:
        engine: The simulation engine instance, expected to have a
            `get_snapshot` method.
        step_index: The step index for which to retrieve the snapshot.

    Returns:
        The `SnapshotRecord` for the given step, or `None` if not found.
    """
    return getattr(engine, "get_snapshot", lambda _i: None)(step_index)
