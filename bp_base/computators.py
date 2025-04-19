import numpy as np
import logging
from typing import List, Dict, Tuple, Callable, Any

from DCOP_base import Computator
from bp_base.typing import CostTable
from bp_base.components import Message

# Set up logging
logger = logging.getLogger(__name__)

class BPComputator(Computator):
    """
    Generic class for message computation in belief propagation.
    Can be configured for different BP variants (min-sum, max-sum, etc.)
    """

    def __init__(self, reduce_func, combine_func):
        """
        Initialize the computator with the appropriate combination and reduction operations.

        :param reduce_func: Function used to reduce over dimensions (e.g., min, max)
        :param combine_func: Function used to combine messages (e.g., add, multiply)
        """
        self.reduce_func = reduce_func
        self.combine_func = combine_func
        logger.info(f"Initialized Computator with reduce_func={reduce_func.__name__}, combine_func={combine_func.__name__}")
    
    def compute_Q(self, messages: List[Message]) -> List[Message]:
        """
        Compute variable->factor messages (Q messages).
        
        For each outgoing message to a factor f, the variable combines all 
        incoming messages from factors EXCEPT f.
        
        Args:
            messages: List of incoming messages from factors to the variable
            
        Returns:
            List of outgoing messages from variable to factors
        """
        logger.info(f"Computing Q messages with {len(messages)} incoming messages")
        
        if not messages:
            logger.warning("No incoming messages, returning empty list")
            return []
        
        # The recipient of all incoming messages is the same variable node
        variable = messages[0].recipient
        
        # Generate outgoing messages, one for each incoming message
        outgoing_messages = []
        
        for i, msg_i in enumerate(messages):
            factor = msg_i.sender
            
            # Combine all messages except the one from this factor
            other_messages = [msg_j.data for j, msg_j in enumerate(messages) if j != i]
            
            if other_messages:
                # Combine messages from other factors
                combined_data = other_messages[0].copy() #so the dim is right
                for msg_data in other_messages[1:]:
                    combined_data = self.combine_func(combined_data, msg_data)
                    

            else:
                # If no other messages, send uniform/uninformative message
                combined_data = np.zeros_like(msg_i.data)
            
            # Create outgoing message to this factor
            outgoing_message = Message(
                data=combined_data,
                sender=variable,
                recipient=factor
            )
            
            outgoing_messages.append(outgoing_message)
        
        logger.info(f"Computed {len(outgoing_messages)} outgoing Q messages")
        return outgoing_messages
    
    def compute_R(self, 
                  cost_table: CostTable, 
                  incoming_messages: List[Message]) -> List[Message]:
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
        
        factor = incoming_messages[0].recipient
        ###----------------------meant only for tests------------------------###
        if not hasattr(factor, 'connection_numbers'):
            # For mock nodes in tests, create a simulated connection_numbers dictionary
            # based on the order of the incoming messages
            factor.connection_numbers = {}
            for i, msg in enumerate(incoming_messages):
                factor.connection_numbers[msg.sender] = i
        ###----------------------meant only for tests------------------------###

        
        outgoing_messages = []
        
        # For each variable index i, compute R_{f->i}
        for i, msg_i in enumerate(incoming_messages):
            variable_node = msg_i.sender
            dim = factor.connection_numbers[variable_node]
            
            # Create a working copy of the cost table
            augmented_costs = cost_table.copy()
            
            # Add messages from all other variables to the costs
            for j, msg_j in enumerate(incoming_messages):
                if j != i:  # Skip the current variable
                    sender = msg_j.sender
                    sender_dim = factor.connection_numbers[sender]
                    
                    # Create the slicing needed to broadcast the message correctly
                    broadcast_shape = [1] * len(cost_table.shape)
                    broadcast_shape[sender_dim] = len(msg_j.data)
                    
                    # Reshape the message for proper broadcasting
                    reshaped_msg = msg_j.data.reshape(broadcast_shape)
                    
                    # Add the message to the cost table
                    augmented_costs = self.combine_func(augmented_costs, reshaped_msg) #if sum for example adding all messages
            
            # Reduce (min/max) over all dimensions except i
            axes = tuple(j for j in range(len(cost_table.shape)) if j != dim)
            reduced_msg = self.reduce_func(augmented_costs, axis=axes)
            

            
            # Create the outgoing message
            outgoing_message = Message(
                data=reduced_msg,
                sender=factor,
                recipient=variable_node
            )
            
            outgoing_messages.append(outgoing_message)
        
        return outgoing_messages


class MinSumComputator(BPComputator):
    """
    Min-sum algorithm for belief propagation.
    Used to find the assignment that minimizes the sum of costs.
    """
    
    def __init__(self):
        super().__init__(reduce_func=np.min, combine_func=np.add)
        logger.info(f"Initialized MinSumComputator")


class MaxSumComputator(BPComputator):
    """
    Max-sum algorithm for belief propagation.
    Used to find the assignment that maximizes the sum of utilities.
    """
    
    def __init__(self):
        super().__init__(reduce_func=np.max, combine_func=np.add)
        logger.info(f"Initialized MaxSumComputator")
