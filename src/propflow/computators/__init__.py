"""User-facing computators module.

Provides convenient imports like:

    from propflow.computators import MinSumComputator, MaxSumComputator

These map to the implementations in `propflow.bp.computators`.
"""

from ..bp.computators import (
    BPComputator,
    MinSumComputator,
    MaxSumComputator,
    MaxProductComputator,
    SumProductComputator,
)

# Optional convenience registry
COMPUTATORS = {
    "min-sum": MinSumComputator,
    "max-sum": MaxSumComputator,
    "max-product": MaxProductComputator,
    "sum-product": SumProductComputator,
}

__all__ = [
    "BPComputator",
    "MinSumComputator",
    "MaxSumComputator",
    "MaxProductComputator",
    "SumProductComputator",
    "COMPUTATORS",
]

# Optional registry entry for the PyTorch soft-min computator
try:
    from ..nn.torch_computators import SoftMinTorchComputator  # type: ignore

    COMPUTATORS["softmin-torch"] = SoftMinTorchComputator
    __all__.append("SoftMinTorchComputator")
except Exception:
    pass
