from ._version import __version__

# Lazy imports - only import when accessed
def __getattr__(name):
    if name == "BPEngine":
        from .bp_base.engine_base import BPEngine
        return BPEngine
    elif name == "FactorGraph":
        from .bp_base.factor_graph import FactorGraph
        return FactorGraph
    elif name == "VariableAgent":
        from .base_models.agents import VariableAgent
        return VariableAgent
    elif name == "FactorAgent":
        from .base_models.agents import FactorAgent
        return FactorAgent
    elif name == "SplitEngine":
        from .bp_base.engines_realizations import SplitEngine
        return SplitEngine
    elif name == "DampingEngine":
        from .bp_base.engines_realizations import DampingEngine
        return DampingEngine
    elif name == "CostReductionOnceEngine":
        from .bp_base.engines_realizations import CostReductionOnceEngine
        return CostReductionOnceEngine
    elif name == "DampingCROnceEngine":
        from .bp_base.engines_realizations import DampingCROnceEngine
        return DampingCROnceEngine
    elif name == "DampingSCFGEngine":
        from .bp_base.engines_realizations import DampingSCFGEngine
        return DampingSCFGEngine
    elif name == "DiscountEngine":
        from .bp_base.engines_realizations import DiscountEngine
        return DiscountEngine
    elif name == "MessagePruningEngine":
        from .bp_base.engines_realizations import MessagePruningEngine
        return MessagePruningEngine
    elif name == "FGBuilder":
        from .utils.fg_utils import FGBuilder
        return FGBuilder
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "__version__",
    "BPEngine",
    "FactorGraph",
    "VariableAgent",
    "FactorAgent",
    "SplitEngine",
    "DampingEngine",
    "CostReductionOnceEngine",
    "DampingCROnceEngine",
    "DampingSCFGEngine",
    "DiscountEngine",
    "MessagePruningEngine",
    "FGBuilder",
]
