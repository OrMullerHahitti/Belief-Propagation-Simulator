from __future__ import annotations

"""
Convenience helpers to fetch snapshot artifacts from a BPEngine.
"""

from typing import Optional


def latest_snapshot(engine) -> Optional[object]:
    return getattr(engine, "latest_snapshot", lambda: None)()


def latest_jacobians(engine):
    rec = latest_snapshot(engine)
    return getattr(rec, "jacobians", None) if rec else None


def latest_cycles(engine):
    rec = latest_snapshot(engine)
    return getattr(rec, "cycles", None) if rec else None


def latest_winners(engine):
    rec = latest_snapshot(engine)
    return getattr(rec, "winners", None) if rec else None


def get_snapshot(engine, step_index: int):
    return getattr(engine, "get_snapshot", lambda _i: None)(step_index)
