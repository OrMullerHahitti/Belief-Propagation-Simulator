"""Constants and device/dtype helpers for the vendored DABP solver.

Ported from ``DABP-main/alg/constant.py`` with two changes:
- device selection also considers Apple MPS, not only CUDA
- dtype is chosen per device because MPS does not support float64
"""

import torch

# node-feature prefix one-hot identifiers (see DABP model)
VAR_ID = [1, 0, 0, 0]
FUN_ID = [0, 1, 0, 0]
V2F_ID = [0, 0, 1, 0]
F2V_ID = [0, 0, 0, 1]

MAX_COLOR_NUM = 100
# cost magnitude scaling used during training (does not affect reported cost,
# which we always evaluate on the original PropFlow cost tables)
SCALE = 50
# DABP splits every factor into two clones distributing the original cost
SPLIT_RATIO = 0.95


def select_device(prefer: str | None = None) -> torch.device:
    """pick a torch device: explicit override, else cuda > mps > cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for_device(device: torch.device) -> torch.dtype:
    """double everywhere for fidelity, except MPS which lacks float64 support."""
    return torch.float32 if device.type == "mps" else torch.float64
