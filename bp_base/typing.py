from typing import Any, Dict, List, Tuple,TypeAlias,Optional,Callable,Union,TypeVar, Protocol

class BPAgent(Protocol):
    """Base protocol type for Belief Propagation agents"""
    name: str
    domain: int
    mailbox: dict
#TODO : add more protocols!

import numpy as np

CostTable : TypeAlias = np.ndarray

class Message(Protocol):
    """Base protocol type for Message classes"""
    data: np.ndarray
    sender: Any
    recipient: Any
    def copy(self) -> 'Message':
        pass

class Computator(Protocol):
    """Base protocol type for Computator classes"""
    def compute_Q(self, messages: List[Message]) -> List[Message]:
        pass

    def compute_R(self, cost_table: CostTable, messages: Message) -> Message:
        pass
    


