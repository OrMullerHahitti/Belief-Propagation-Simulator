"""User-facing engines module.

Provides convenient imports like:

    from propflow.engines import BPEngine, DampingEngine, SplitEngine

These map to implementations in `propflow.bp.engine_base` and
`propflow.bp.engines`.
"""

from ..bp.engine_base import BPEngine
from ..bp.engines import (
    CostReductionOnceEngine,
    DampingCROnceEngine,
    DampingEngine,
    DampingSCFGEngine,
    DiffusionEngine,
    Engine,
    MessagePruningEngine,
    MidRunSplitEngine,
    QRDampingEngine,
    RDampingEngine,
    SplitEngine,
)


def _load_dabp_engine():
    """lazily import DABPEngine (needs the optional 'dabp' extra: torch + PyG)."""
    try:
        from ..integrations.dabp import DABPEngine
    except ImportError as exc:  # torch / torch-geometric not installed
        raise ImportError(
            "DABPEngine requires the optional 'dabp' extra. Install it with: "
            "uv pip install -e '.[dabp]'"
        ) from exc
    return DABPEngine


class _LazyEngines(dict):
    """engine registry that resolves the optional DABPEngine on first access,
    so importing this module never pulls in torch."""

    def __missing__(self, key):
        if key == "DABPEngine":
            engine = _load_dabp_engine()
            self[key] = engine
            return engine
        raise KeyError(key)


# Optional convenience registry
ENGINES = _LazyEngines({
    "BPEngine": BPEngine,
    "Engine": Engine,
    "SplitEngine": SplitEngine,
    "DampingEngine": DampingEngine,
    "QRDampingEngine": QRDampingEngine,
    "RDampingEngine": RDampingEngine,
    "DiffusionEngine": DiffusionEngine,
    "CostReductionOnceEngine": CostReductionOnceEngine,
    "DampingCROnceEngine": DampingCROnceEngine,
    "DampingSCFGEngine": DampingSCFGEngine,
    "MessagePruningEngine": MessagePruningEngine,
    "MidRunSplitEngine": MidRunSplitEngine,
})


def __getattr__(name):
    # `from propflow.engines import DABPEngine` without forcing torch otherwise
    if name == "DABPEngine":
        return _load_dabp_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BPEngine",
    "Engine",
    "SplitEngine",
    "DampingEngine",
    "QRDampingEngine",
    "RDampingEngine",
    "DiffusionEngine",
    "CostReductionOnceEngine",
    "DampingCROnceEngine",
    "DampingSCFGEngine",
    "MessagePruningEngine",
    "MidRunSplitEngine",
    "ENGINES",
]
