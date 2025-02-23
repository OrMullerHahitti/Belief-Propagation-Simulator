'''class to compute the message for the factor node'''
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from DCOP_base.agent import Message


class Computator(ABC):
    @abstractmethod
    def compute_message(self, message:Message) -> Message:
        pass

#compute for a factor node the message to be sent to a variable node
class FactorCompute(Computator):
    def __init__(self, node_id: str, name: str, domain: List[Any]):
        self.node_id = node_id
        self.name = name
        self.type = "factor"
        self.domain = domain
        self.belief = {}
        self.messages = {}
    def compute_message(self, message:Message) -> Message:
        sender_id = message.sender
        message_data = message.message
        self.messages[sender_id] = message_data
        belief = np.ones(len(self.domain))
        for sender in self.messages:
            belief *= self.messages[sender]
        belief = belief / np.sum(belief)
        self.belief = belief
        return Message(self.node_id, sender_id, belief)
    def update_local_belief(self) -> None:
        pass
    def get_belief(self) -> Dict[str, float]:
        return self.belief
