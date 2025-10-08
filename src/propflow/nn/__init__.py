"""
PyTorch-backed, differentiable components for PropFlow.
These imports are optional and only available if `torch` is installed.
"""

__all__ = []

try:
    from .torch_computators import SoftMinTorchComputator  # noqa: F401
    from .trainable_bp import TrainableBPModule, BPTrainer  # noqa: F401
    __all__.extend(["SoftMinTorchComputator", "TrainableBPModule", "BPTrainer"])
except Exception:
    # PyTorch not installed or failed to import – expose nothing.
    pass

