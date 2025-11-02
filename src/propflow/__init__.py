from ._version import __version__

# Core graph and agent components
from .bp.engine_base import BPEngine
from .bp.factor_graph import FactorGraph
from .core import VariableAgent, FactorAgent

# Graph builders and simulation
from .utils import FGBuilder
from propflow.configs.global_config_mapping import CTFactories
from .simulator import Simulator

# Engine variants
from .bp.engines import (
    SplitEngine,
    DampingEngine,
    CostReductionOnceEngine,
    DampingCROnceEngine,
    DampingSCFGEngine,
    DiscountEngine,
    MessagePruningEngine,
)

# Computators
from .bp.computators import (
    MinSumComputator,
    MaxSumComputator,
    SumProductComputator,
    MaxProductComputator,
)

# Commonly used configs and utilities
from .configs import (
    CTFactories,
    create_random_int_table,
    create_uniform_float_table,
    create_poisson_table,
)

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
    "CostReductionOnceEngine",
    "DampingCROnceEngine",
    "DampingSCFGEngine",
    "DiscountEngine",
    "MessagePruningEngine",
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

# --- Optional (only if PyTorch is available) ---
try:
    # Importing directly avoids failing when __all__ is empty in propflow.nn
    from .nn.torch_computators import SoftMinTorchComputator  # type: ignore
    __all__.append("SoftMinTorchComputator")
except Exception:
    # PyTorch not installed; keep base API intact
    pass
