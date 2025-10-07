"""
PyTorch-backed, differentiable components for PropFlow.
These imports are optional and only available if `torch` is installed.
"""

__all__ = []

try:
    from .torch_computators import SoftMinTorchComputator  # noqa: F401
    __all__.append("SoftMinTorchComputator")
except Exception:
    # PyTorch not installed or failed to import – expose nothing.
    pass

