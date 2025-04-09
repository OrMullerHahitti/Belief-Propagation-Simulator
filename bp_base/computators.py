from sys import implementation
from typing import List, Set, TypeAlias, Dict
import numpy as np
import logging

from bp_base.agents import BPAgent
from bp_base.components import Message, CostTable, BPMessage,Computator

# Configure logger
logger = logging.getLogger(__name__)

class BPComputator(Computator):
    """
    A demonstration class that performs min-sum or max-sum computations
    (depending on aggregation_func) for messages in a factor graph.
    """
    def __init__(self, reduce_func=np.min,combine_func = np.add):
        """
        :param aggregation_func: e.g., np.min for min-sum, or np.max for max-sum
        """
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        logger.info(f"Initialized Computator with reduce_func={reduce_func.__name__}, combine_func={combine_func.__name__}")

    def compute_Q(self, messages: List[BPMessage]) -> List[Message[BPMessage]]:
        """
        Compute variable->factor messages from a variable node's perspective.

        'messages' is a list of factor->variable messages (R_{F->X}),
        each with shape (d,). We want to produce Q_{X->F} for each factor neighbor F.
        """
        logger.info(f"Computing Q messages with {len(messages)} incoming messages")

        if not messages:
            logger.warning("No incoming messages, returning empty list")
            return []

        # 'me' is this variable node (the node that now sends Q_{X->F})
        me = messages[0].recipient  # since all these messages are to 'X'
        senders = [msg.sender for msg in messages]  # factor neighbors
        logger.debug(f"Variable node: {me}, Factor neighbors: {senders}")

        # We assume all messages have same shape 'd'
        d = messages[0].data.shape
        messages_to_send: List[Message[BPMessage]] = []

        # For each factor neighbor F, we compute Q_{X->F}
        for factor_node in senders:
            logger.debug(f"Computing message to factor node: {factor_node}")
            # Start with zero array (or ones if you're doing product)
            combined = np.zeros(d, dtype=float)

            # Sum up (or combine) all factor->variable msgs from the other factors
            for msg_i in messages:
                if msg_i.sender == factor_node:
                    # skip the factor 'F' that we're about to send back to
                    continue

                # e.g., combined += msg_i.data
                combined = self.combine_func(combined, msg_i.data)
                logger.debug(f"Combined with message from {msg_i.sender}, current result: {combined}")

            #normalize if you want:
            offset = np.min(combined)
            combined -= offset
            logger.debug(f"Normalized message by subtracting {offset}, result: {combined}")

            # 'me' is the variable node, 'factor_node' is the factor neighbor
            messages_to_send.append(Message(
                data=combined,
                sender=me,  # variable node
                recipient=factor_node  # factor node
            ))

        logger.info(f"Computed {len(messages_to_send)} outgoing Q messages")
        return messages_to_send


    def compute_R(self,
                  cost_table: CostTable,
                  incoming_messages: List[Message["BPAgent"]]) -> List[Message["BPAgent"]]:
        """
        Compute factor->variable messages. We assume:
          - 'cost_table' is an n-dimensional array (d, d, ..., d).
          - 'incoming_messages[i]' is Q_{i->f}, shape (d,), from variable i.

        :param cost_table: The factor's cost table
        :param incoming_messages: A list of n messages, one from each variable -> factor
        :return: A list of n messages, factor -> each variable
        """
        logger.info(f"Computing R messages with {len(incoming_messages)} incoming messages and cost table shape {cost_table.shape if hasattr(cost_table, 'shape') else 'unknown'}")
        
        n = len(incoming_messages)
        if n == 0:
            logger.warning("No incoming messages, returning empty list")
            return []

        d = cost_table.shape[0]  # assume each dimension is size d
        outgoing_messages = []

        # For each variable index i, compute R_{f->i}
        for  msg_i in incoming_messages:
            i= msg_i.recipient.[msg_i.recipient]
            logger.debug(f"Computing message to variable node: {msg_i.sender}")
            # 1) Copy the factor's cost table
            combined = cost_table  # shape (d, ..., d)

            # 2) Add incoming messages from all other variables j != i
            for  msg_j in incoming_messages:
                j= msg_j.sender.domains[msg_j.recipient]
                if j == i:
                    continue
                # Reshape Q_{j->f} for broadcasting across dimension j
                reshape_dims = [1] * n
                reshape_dims[j] = d
                msg_data_j = msg_j.data.reshape(reshape_dims)
                # Add to the combined cost table
                combined = self.combine_func(combined, msg_data_j)
                logger.debug(f"Combined with message from {msg_j.sender}")

            # 3) Aggregate (min or max) across all axes except i
            axes_to_reduce = tuple(ax for ax in range(n) if ax != i)
            logger.debug(f"Reducing across axes {axes_to_reduce}")
            result_1d = self.reduce_func(combined, axis=axes_to_reduce)  # shape (d,)
            logger.debug(f"Reduction result: {result_1d}")

            # 4) Wrap in a new Message, from factor -> variable i
            #    'msg_i.sender' is the variable node that sent Q_{i->f},
            #    so we now send from factor -> that same variable node.
            new_msg = Message(
                data=result_1d,
                sender=msg_i.recipient,          # or a factor node object if you have one
                recipient=msg_i.sender    # the original sender variable
            )
            outgoing_messages.append(new_msg)

        logger.info(f"Computed {len(outgoing_messages)} outgoing R messages")
        return outgoing_messages

    @property
    def assignemt(self,curr_belief:np.ndarray) -> Dict[str | int, float | int]:
        """
        Get the assignment of the variable based on the final belief.
        :return: index of the assignment
        """
        return {np.argmax(curr_belief).astype(int): curr_belief.max().astype(int)}

        pass




### -------------------------- ###
### ------ implementation ---- ###
### -------------------------- ###
class MinSumComputator(BPComputator):
    def __init__(self):
        super().__init__(combine_func=np.add, reduce_func=np.min)
        logger.info("Initialized MinSumComputator")

class MaxSumComputator(BPComputator):
    def __init__(self):
        super().__init__(combine_func=np.add, reduce_func=np.max)
        logger.info("Initialized MaxSumComputator")
