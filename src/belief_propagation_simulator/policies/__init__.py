"""Policies used by belief propagation engines."""

from .damping import damp, TD
from .cost_reduction import cost_reduction_all_factors_once, discount_attentive
from .splitting import split_all_factors
from .message_pruning import MessagePruningPolicy

__all__ = [
    "damp",
    "TD",
    "cost_reduction_all_factors_once",
    "discount_attentive",
    "split_all_factors",
    "MessagePruningPolicy",
]
