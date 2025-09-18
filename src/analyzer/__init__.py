"""PropFlow snapshot recorder utilities.

This lightweight package exposes the `EngineSnapshotRecorder`, a helper for
capturing per-iteration message traffic, assignments, and costs from Belief
Propagation engines. See `src/analyzer/README.md` for a comprehensive guide.
"""

from .snapshot_recorder import EngineSnapshotRecorder, MessageSnapshot

__all__ = ["EngineSnapshotRecorder", "MessageSnapshot"]
