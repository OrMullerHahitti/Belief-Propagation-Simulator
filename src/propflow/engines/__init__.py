"""User-facing engines module.

Provides convenient imports like:

    from propflow.engines import BPEngine, DampingEngine, SplitEngine

These map to implementations in `propflow.bp.engine_base` and
`propflow.bp.engines`.
"""

from ..bp.engine_base import BPEngine
from ..bp.engines import (
    Engine,
    SplitEngine,
    DampingEngine,
    DiffusionEngine,
    CostReductionOnceEngine,
    DampingCROnceEngine,
    DampingSCFGEngine,
    MessagePruningEngine,
)

# Optional convenience registry
ENGINES = {
    "BPEngine": BPEngine,
    "Engine": Engine,
    "SplitEngine": SplitEngine,
    "DampingEngine": DampingEngine,
    "DiffusionEngine": DiffusionEngine,
    "CostReductionOnceEngine": CostReductionOnceEngine,
    "DampingCROnceEngine": DampingCROnceEngine,
    "DampingSCFGEngine": DampingSCFGEngine,
    "MessagePruningEngine": MessagePruningEngine,
}

__all__ = [
    "BPEngine",
    "Engine",
    "SplitEngine",
    "DampingEngine",
    "DiffusionEngine",
    "CostReductionOnceEngine",
    "DampingCROnceEngine",
    "DampingSCFGEngine",
    "MessagePruningEngine",
    "ENGINES",
]
