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
    CostReductionOnceEngine,
    DampingCROnceEngine,
    DampingSCFGEngine,
    DiscountEngine,
    MessagePruningEngine,
)

# Optional convenience registry
ENGINES = {
    "BPEngine": BPEngine,
    "Engine": Engine,
    "SplitEngine": SplitEngine,
    "DampingEngine": DampingEngine,
    "CostReductionOnceEngine": CostReductionOnceEngine,
    "DampingCROnceEngine": DampingCROnceEngine,
    "DampingSCFGEngine": DampingSCFGEngine,
    "DiscountEngine": DiscountEngine,
    "MessagePruningEngine": MessagePruningEngine,
}

__all__ = [
    "BPEngine",
    "Engine",
    "SplitEngine",
    "DampingEngine",
    "CostReductionOnceEngine",
    "DampingCROnceEngine",
    "DampingSCFGEngine",
    "DiscountEngine",
    "MessagePruningEngine",
    "ENGINES",
]
