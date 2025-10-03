"""PropFlow analysis helpers for recording and interpreting engine runs.

The package exposes:

* `EngineSnapshotRecorder` for capturing per-iteration message traffic.
* `SnapshotAnalyzer` and `AnalysisReport` for deriving graph metrics, influence
  scores, and connected-component insights from recorded snapshots.

See `src/analyzer/README.md` and `examples/` for end-to-end usage guidance.
"""

from .snapshot_recorder import EngineSnapshotRecorder, MessageSnapshot

# Reporting module imports - temporarily disabled due to missing files
# from .reporting import (
#     AnalysisReport,
#     MessageRecord,
#     SnapshotAnalyzer,
#     SnapshotRecord,
#     load_snapshots,
#     parse_snapshots,
# )

__all__ = [
    "EngineSnapshotRecorder",
    "MessageSnapshot",
    # "SnapshotAnalyzer",
    # "AnalysisReport",
    # "MessageRecord",
    # "SnapshotRecord",
    # "parse_snapshots",
    # "load_snapshots",
]
