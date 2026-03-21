from ._version import __version__
from .bp.computators import (
    MaxProductComputator,
    MaxSumComputator,
    MinSumComputator,
    SumProductComputator,
)
from .bp.engine_base import BPEngine
from .bp.engines import (
    CostReductionOnceEngine,
    DampingCROnceEngine,
    DampingEngine,
    DampingSCFGEngine,
    DampingTRWEngine,
    DiffusionEngine,
    MessagePruningEngine,
    QRDampingEngine,
    RDampingEngine,
    SplitEngine,
    TRWEngine,
)
from .bp.factor_graph import FactorGraph
from .configs import (
    CTFactories,
    create_poisson_table,
    create_random_int_table,
    create_uniform_float_table,
)
from .core import FactorAgent, VariableAgent
from .simulator import Simulator
from .utils import FGBuilder

# Snapshot configuration and analysis
from .snapshots import (
    EngineSnapshot,
    SnapshotManager,
    SnapshotAnalyzer,
    AnalysisReport,
    SnapshotVisualizer,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BPEngine",
    "FactorGraph",
    "VariableAgent",
    "FactorAgent",
    # Engines
    "SplitEngine",
    "DampingEngine",
    "QRDampingEngine",
    "RDampingEngine",
    "DiffusionEngine",
    "CostReductionOnceEngine",
    "DampingCROnceEngine",
    "DampingSCFGEngine",
    "DampingTRWEngine",
    "MessagePruningEngine",
    "TRWEngine",
    # Computators
    "MinSumComputator",
    "MaxSumComputator",
    "SumProductComputator",
    "MaxProductComputator",
    # Builders & Simulation
    "FGBuilder",
    "Simulator",
    "CTFactories",
    # Configs
    "create_random_int_table",
    "create_uniform_float_table",
    "create_poisson_table",
    # Snapshots
    "EngineSnapshot",
    "SnapshotManager",
    "SnapshotAnalyzer",
    "AnalysisReport",
    "SnapshotVisualizer",
]

try:
    from .nn.torch_computators import SoftMinTorchComputator  # type: ignore

    __all__.append("SoftMinTorchComputator")
except Exception:
    pass
