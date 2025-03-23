# policies/VariableComputator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set
import numpy as np

from DCOP_base import VariableNode, Message, FactorNode

# Stub classes for demonstration; replace with actual imports:
# from your_project.agents import Message, VariableNode


class VariableComputator(ABC):
    """
    An abstract base class defining how variable nodes:
    1) Combine incoming messages to form outgoing messages.
    2) Compute their local belief.

    Different implementations (min-sum, max-sum, sum-product)
    will override these methods to define custom logic.
    """
    @abstractmethod
    def compute_outgoing_messages(
        self,
        var_node: VariableNode,
        incoming_messages: Set[Message]
    ) -> Set[Message]:
        """
        Using the variable node state plus the set of incoming messages,
        compute the set of outgoing messages for all neighbors.
        """
        pass

    @abstractmethod
    def compute_belief(
        self,
        var_node: VariableNode,
        incoming_messages: Set[Message]
    ) -> np.ndarray:
        """
        Compute the local belief or distribution for the variable node
        given its incoming messages.
        """
        pass


class MinSumVariableComputator(VariableComputator):
    """
    Example: Min-Sum logic.
    """
    def compute_outgoing_messages(
        self,
        var_node: VariableNode,
        incoming_messages: Set[Message]
    ) -> Set[Message]:
        message_set = set()

        # Suppose each incoming data is an array of shape (domain_size,).
        # Combine all messages: e.g., element-wise sum them.
        # Then to send a data back to each neighbor, we subtract that neighbor's contribution,
        # aligning with standard "min-sum" or "sum-product" style factor graphs.

        # 1) Gather incoming messages from each neighbor:
        sender_to_msg = {m.sender: m.data for m in incoming_messages}
        # 2) Sum them up:
        total = np.sum(list(sender_to_msg.values()), axis=0)
        # 3) Build new messages for each neighbor:
        for neighbor, msg_val in sender_to_msg.items():
            # "Remove" neighbor's contribution:
            new_msg_val = total - msg_val
            # Then, depending on min-sum vs. sum-product, we might do something else;
            # for min-sum, you might keep it just as is, or apply a transform.
            message_set.add(Message(new_msg_val, var_node, neighbor))

        return message_set

    def compute_belief(
        self,
        var_node: VariableNode,
        incoming_messages: Set[Message]
    ) -> np.ndarray:
        # For min-sum, you might do an element-wise sum and then take an argmin or similar.
        # As a trivial example:
        sum_all = np.sum([m.data for m in incoming_messages], axis=0)
        # Potentially apply further min-sum logic (like normalizing or picking the best domain index).
        return sum_all
class FactorComputator(ABC):
    """
    An abstract base class defining how factor nodes:
    1) Combine incoming messages to generate outgoing messages.
    2) Potentially compute factor-based local 'belief' if needed.
    """
    @abstractmethod
    def compute_outgoing_messages(
        self,
        factor_node: FactorNode,
        cost_table: np.ndarray,
        incoming_messages: Set[Message]
    ) -> Set[Message]:
        """
        Using the factor node's cost table and incoming messages from variable nodes,
        compute the set of outgoing messages to each connected variable.
        """
        pass


class MinSumFactorComputator(FactorComputator):
    """
    Example: Min-Sum logic for factor nodes.
    Typically, you'd marginalize over all but one variable
    to produce the data to that variable.
    """

    def compute_outgoing_messages(
        self,
        factor_node: FactorNode,
        cost_table: np.ndarray,
        incoming_messages: Set[Message]
    ) -> Set[Message]:
        message_set = set()

        # Example approach:
        # 1) Convert incoming messages into arrays that align with factor_node's cost_table indexing.
        # 2) Combine them with cost_table (e.g., add them for min-sum).
        # 3) Marginalize out the other variables to build a data for each variable node.

        # For demo, suppose each incoming data is an array we "add" to cost_table:
        for incoming_msg in incoming_messages:
            # Combine cost_table with the incoming data:
            combined = cost_table + incoming_msg.data
            # Then do a naive "min over one axis" to get a 1D array for the variable node:
            # (The actual axis depends on how you shape your cost table.)
            new_msg_values = np.min(combined, axis=0)

            # Construct the outgoing data, setting the factor_node as sender
            # and the original sender (a VariableNode) as recipient:
            msg = Message(new_msg_values, factor_node, incoming_msg.sender)
            message_set.add(msg)

        return message_set


class MaxSumFactorComputator(FactorComputator):
    """
    Example: Max-Sum logic for factor nodes.
    Similar structure to MinSumFactorComputator, but uses 'max' instead of 'min.'
    """

    def compute_outgoing_messages(
        self,
        factor_node: FactorNode,
        cost_table: np.ndarray,
        incoming_messages: Set[Message]
    ) -> Set[Message]:
        message_set = set()

        for incoming_msg in incoming_messages:
            combined = cost_table + incoming_msg.data
            # Use np.max along the appropriate axis to get the "max" version
            new_msg_values = np.max(combined, axis=0)

            msg = Message(new_msg_values, factor_node, incoming_msg.sender)
            message_set.add(msg)

        return message_set

class MaxSumVariableComputator(VariableComputator):
    """
    Example: Max-Sum logic.
    """
    def compute_outgoing_messages(
        self,
        var_node: VariableNode,
        incoming_messages: Set[Message]
    ) -> Set[Message]:
        message_set = set()

        sender_to_msg = {m.sender: m.data for m in incoming_messages}
        total = np.sum(list(sender_to_msg.values()), axis=0)
        for neighbor, msg_val in sender_to_msg.items():
            new_msg_val = total - msg_val
            message_set.add(Message(new_msg_val, var_node, neighbor))

        return message_set

    def compute_belief(
        self,
        var_node: VariableNode,
        incoming_messages: Set[Message]
    ) -> np.ndarray:
        # For max-sum, combine incoming messages in a "max-sum" sense, maybe element-wise sum
        # then interpret the result as a "utility" array, from which we pick argmax, etc.
        sum_all = np.sum([m.data for m in incoming_messages], axis=0)
        return sum_all
