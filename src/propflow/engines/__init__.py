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


_DABP_ENGINE_NAMES = {"DABPEngine", "DABPEngineNoSplit", "DABPEngine_No_Split"}


def _load_dabp_engine(name: str = "DABPEngine"):
    """lazily import a DABP engine (needs the optional 'dabp' extra: torch + PyG)."""
    try:
        from ..integrations import dabp as dabp_module
    except ImportError as exc:  # torch / torch-geometric not installed
        raise ImportError(
            f"{name} requires the optional 'dabp' extra. Install it with: "
            "uv pip install -e '.[dabp]'"
        ) from exc
    return getattr(dabp_module, name)


class _LazyEngines(dict):
    """engine registry that resolves the optional DABPEngine on first access,
    so importing this module never pulls in torch."""

    def __missing__(self, key):
        if key in _DABP_ENGINE_NAMES:
            engine = _load_dabp_engine(key)
            self[key] = engine
            return engine
        raise KeyError(key)


# Optional convenience registry
ENGINES = _LazyEngines(
    {
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
    }
)


def __getattr__(name):
    # `from propflow.engines import DABPEngine` without forcing torch otherwise
    if name in _DABP_ENGINE_NAMES:
        return _load_dabp_engine(name)
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
