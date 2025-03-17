from ast import iter_fields
from typing import Set

from agents import BPComputator,BPAgent,Message
import numpy as np
class MinSumComputator(BPComputator):
    def compute_Q(self,messages:Set[Message]) -> np.ndarray :
        result = np.zeros_like(iter(messages))
        for message in messages:
            result += message.message

        return result
    #compute R as in min sum algorithm summing the cost table and the incoming messages and then computing the min
    def compute_R(self,cost_table:np.ndarray,message:f)->Message:
        cost_table = np.array(cost_table)
        result = np.zeros_like(cost_table)
        min_sum = np.min(cost_table + result, axis=0)
        return Message(min_sum, message.recipient, message.sender)
    def get_belief(self):
        pass