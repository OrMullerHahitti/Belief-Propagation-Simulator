"""Policies for Normalizing Costs and Messages.

This module contains various functions used to normalize cost tables and
messages within the belief propagation framework. Normalization is often crucial
for preventing numerical instability and ensuring that costs or probabilities
remain in a manageable range.
"""
from typing import List
import numpy as np
from ..core.agents import VariableAgent, FactorAgent
from ..core.protocols import CostTable


def init_normalization(li: List[FactorAgent]) -> None:
    """Performs an initial normalization of cost tables for a list of factors.

    This function divides each factor's cost table by the total number of
    factors in the list.

    Args:
        li: A list of `FactorAgent` objects to be normalized.
    """
    x = len(li)
    for factor in li:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table / x


def normalize_soft_max(cost_table: np.ndarray) -> CostTable:
    """Normalizes a cost table using the softmax function.

    This ensures all values are positive and sum to 1, effectively converting
    costs into a probability distribution. A stability trick (subtracting the
    max value) is used to prevent overflow with large cost values.

    Args:
        cost_table: An n-dimensional numpy array representing the cost table.

    Returns:
        The normalized cost table.
    """
    exp_cost_table = np.exp(cost_table - np.max(cost_table))
    return exp_cost_table / np.sum(exp_cost_table)


def normalize_cost_table_sum(cost_table: np.ndarray) -> CostTable:
    """Normalizes a cost table so that the sum across all dimensions is equal.

    Note:
        The implementation of this function appears to be incorrect for its
        stated purpose and may not produce the intended normalization.

    Args:
        cost_table: An n-dimensional numpy array representing the cost table.

    Returns:
        The modified cost table.
    """
    total_sum = np.sum(cost_table)
    shape = cost_table.shape
    for dim in range(len(shape)):
        curr_sum = np.sum(cost_table, axis=dim)
        cost_table = cost_table / (curr_sum * total_sum)
    return cost_table


def normalize_inbox(variables: List[VariableAgent]) -> None:
    """Normalizes all messages in the inboxes of a list of variable agents.

    This function iterates through each variable's inbox and last iteration's
    messages, normalizing them by subtracting the minimum value. This centers
    the message values around zero without changing their relative differences.

    Args:
        variables: A list of `VariableAgent` objects whose messages will be
            normalized.
    """
    for var in variables:
        if var.last_iteration is not None:
            for message in var.last_iteration:
                if message.data is not None:
                    message.data = message.data - message.data.min()
        for message in var.mailer.inbox:
            if message.data is not None:
                message.data = message.data - message.data.min()
