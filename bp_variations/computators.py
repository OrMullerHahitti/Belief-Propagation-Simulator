from typing import List, Set
import numpy as np

from bp_base.agents import BPComputator, Message, BPAgent


class SumComputator(BPComputator):
    def __init__(self, aggregation_func=np.min):
        """
        Initialize with an aggregation function (min or max)
        """
        self.aggregation_func = aggregation_func

    def compute_Q(self, messages: List[Message]) -> np.ndarray:
        """Sum up message arrays from a set of messages."""
        if not messages:
            return np.array([])

        result = np.zeros_like(range(messages[0].message))

        for message in messages:
            result += message.message

        return result

    def compute_R(self, cost_table: np.ndarray, messages: List[Message]) -> List[Message]:
        """Compute factor-to-variable messages using the configured aggregation function."""
        outgoing_messages = []

        for idx, target_message in enumerate(messages):
            combined = cost_table.copy()

            for i, msg in enumerate(messages):
                if i != idx:  # Skip the message from the target variable
                    combined += msg.message

            # Use the configured aggregation function
            result = self.aggregation_func(combined, axis=tuple(range(1, combined.ndim)))

            outgoing_message = Message(result, target_message.recipient, target_message.sender)
            outgoing_messages.append(outgoing_message)

        return outgoing_messages

    def get_belief(self, node: BPAgent) -> np.ndarray:
        """Compute belief at a node by summing all incoming messages."""
        if not hasattr(node, 'messages') or not node.messages:
            return np.array([])

        result = np.zeros_like(next(iter(node.messages)).message)

        for message in node.messages:
            result += message.message

        return result




class ProductComputator(BPComputator):
    def __init__(self, aggregation_func=np.sum):
        """
        Initialize with an aggregation function for the factor update.
        """
        self.aggregation_func = aggregation_func

    def compute_Q(self, messages: Set[Message]) -> np.ndarray:
        """
        Multiply message arrays from a set of messages.
        For variable-to-factor messages.
        """
        if not messages:
            return np.array([])
        result = np.zeros_like(messages[0].message)
        for message in messages:
            result *= message.message
        return result

    def compute_R(self, cost_table: np.ndarray, messages: List[Message]) -> List[Message]:
        """
        Compute factor-to-variable messages.
        For each target message, multiply the cost table with all messages except that from the target.
        Then aggregate using the configured aggregation function over all dimensions except the target variable's.
        """
        outgoing_messages = []
        for idx, target_message in enumerate(messages):
            # Start from cost table copy.
            combined = cost_table.copy()
            # Multiply in all messages except the target.
            for i, msg in enumerate(messages):
                if i != idx:
                    combined *= msg.message
            # Aggregate over dimensions except the first (assumed target variable dimension)
            result = self.aggregation_func(combined, axis=tuple(range(1, combined.ndim)))
            outgoing_message = Message(result, target_message.recipient, target_message.sender)
            outgoing_messages.append(outgoing_message)
        return outgoing_messages

    def get_belief(self, node: BPAgent) -> np.ndarray:
        """
        Compute node belief by multiplying all incoming messages.
        """
        if not hasattr(node, 'messages') or not node.messages:
            return np.array([])
        result = np.ones_like(next(iter(node.messages)).message)
        for message in node.messages:
            result *= message.message
        return result

#### ------------------------------------------------------- ####
####                 Additional Computators                  ####
#### ------------------------------------------------------- ####

class MinSumComputator(SumComputator):
    def __init__(self):
        super().__init__(aggregation_func=np.min)


class MaxSumComputator(SumComputator):
    def __init__(self):
        super().__init__(aggregation_func=np.max)

class MaxProductComputator(ProductComputator):
    def __init__(self):
        super().__init__(aggregation_func=np.max)

class SumProductComputator(ProductComputator):
    def __init__(self):
        super().__init__(aggregation_func=np.sum)