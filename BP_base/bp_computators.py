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

    def compute_R(self,cost_table:np.ndarray,messages:Message)->[Message]:
        pass
    def get_belief(self):
        pass