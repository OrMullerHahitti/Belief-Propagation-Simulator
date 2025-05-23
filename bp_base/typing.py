from typing import (
    Any,
    Dict,
    List,
    Tuple,
    TypeAlias,
    Optional,
    Callable,
    Union,
    TypeVar,
    Protocol,
    Literal,
)
import numpy as np

PolicyType = Literal["message", "cost_table", "stopping_criteria", "assignment"]


class BPAgent(Protocol):
    """Base protocol type for Belief Propagation agents"""

    name: str
    domain: int
    mailbox: dict


CostTable: TypeAlias = np.ndarray


class Message(Protocol):
    """Base protocol type for Message classes"""

    data: np.ndarray
    sender: Any
    recipient: Any

    def copy(self) -> "Message":
        pass


class Computator(Protocol):
    """Base protocol type for Computator classes"""

    def compute_Q(self, messages: List[Message]) -> List[Message]:
        pass

    def compute_R(self, cost_table: CostTable, messages: Message) -> Message:
        pass


class Policy(Protocol):
    """Base protocol type for Policy classes"""

    type: PolicyType
    __call__: Callable
