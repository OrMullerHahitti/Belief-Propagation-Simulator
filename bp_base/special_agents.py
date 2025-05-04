from typing import List

from agents import FactorAgent,VariableAgent

from components import Message
from computators import BPComputator
from policies.splitting import SplittingPolicy
class SplitFactorAgent(FactorAgent):
    """
    A specialized factor agent that splits its messages into two parts.
    """
    def __init__(self, name: str, domain: int, split :SplittingPolicy):
        super().__init__(name, domain, **kwargs)
        self.split_factor = 0.5  # Default split factor

    def compute_messages(self) -> List[Message]:
        """
        Compute messages to send to connected variable agents.
        This method splits the message into two parts based on the split factor.
        """
        messages = []



        return