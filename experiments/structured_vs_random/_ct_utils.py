"""Shared picklable cost-table factory used by generate_graphs.py and run_experiment.py."""

from __future__ import annotations

import numpy as np


class _FixedCostTable:
    """picklable cost-table factory — replaces the unpicklable lambda in create_from_cost_table."""

    def __init__(self, ct: np.ndarray) -> None:
        self.ct = ct.copy()

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.ct.copy()
