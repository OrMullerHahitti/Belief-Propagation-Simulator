"""Policies used by belief propagation bp."""

from .damping import damp, TD
from .cost_reduction import cost_reduction_all_factors_once, discount_attentive
from .splitting import split_all_factors
from .message_pruning import MessagePruningPolicy
from .normalize_cost import (
    normalize_inbox,
    normalize_soft_max,
    normalize_cost_table_sum,
    init_normalization,
)
from .convergance import ConvergenceConfig

__all__ = [
    "damp",
    "TD",
    "cost_reduction_all_factors_once",
    "discount_attentive",
    "split_all_factors",
    "MessagePruningPolicy",
    "normalize_inbox",
    "normalize_soft_max",
    "normalize_cost_table_sum",
    "init_normalization",
]
