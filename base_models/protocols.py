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
    runtime_checkable,
    Sequence,
    Mapping,
    Iterator,
)

import numpy as np

PolicyType = Literal["message", "cost_table", "stopping_criteria", "assignment"]

# --- Protocols for typing purposes ---


@runtime_checkable
class FGAgent(Protocol):
    name: str
    domain: int
    mailbox: dict
    mailer: "MailHandler"
    computator: Optional["Computator"]

    def receive_message(self, message: "Message") -> None: ...
    def send_message(self, message: "Message") -> None: ...
    def empty_mailbox(self) -> None: ...
    def empty_outgoing(self) -> None: ...
    @property
    def inbox(self) -> List["Message"]: ...
    @property
    def outbox(self) -> List["Message"]: ...
    def compute_messages(self) -> List["Message"]: ...
    @property
    def last_iteration(self) -> List["Message"]: ...
    def last_cycle(self, diameter: int = 1) -> List["Message"]: ...
    def append_last_iteration(self): ...
    @property
    def belief(self) -> np.ndarray: ...
    @property
    def curr_assignment(self) -> Union[int, float]: ...


CostTable: TypeAlias = np.ndarray


@runtime_checkable
class Message(Protocol):
    data: np.ndarray
    sender: Any
    recipient: Any

    def copy(self) -> "Message": ...


@runtime_checkable
class MailHandler(Protocol):
    inbox: List["Message"]
    outbox: List["Message"]

    def receive_messages(self, message: "Message") -> None: ...
    def send(self) -> None: ...
    def clear_inbox(self) -> None: ...
    def clear_outgoing(self) -> None: ...
    def prepare(self) -> None: ...
    def set_first_message(self, sender: Any, recipient: Any) -> None: ...
    def stage_sending(self, messages: List["Message"]) -> None: ...


@runtime_checkable
class Computator(Protocol):
    def compute_Q(self, messages: List["Message"]) -> List["Message"]: ...
    def compute_R(
        self, cost_table: CostTable, incoming_messages: List["Message"]
    ) -> List["Message"]: ...


@runtime_checkable
class Policy(Protocol):
    type: PolicyType

    def __call__(self, *args, **kwargs): ...


@runtime_checkable
class Step(Protocol):
    num: int
    messages: Dict[str, List["Message"]]

    def add(self, agent: Any, message: "Message"): ...


@runtime_checkable
class Cycle(Protocol):
    number: int
    steps: List[Step]

    def add(self, step: Step): ...
    def __eq__(self, other: Any): ...


@runtime_checkable
class HistoryProtocol(Protocol):
    config: dict
    cycles: Dict[int, Cycle]
    beliefs: Dict[int, Dict[str, np.ndarray]]
    assignments: Dict[int, Dict[str, Union[int, float]]]
    costs: List[Union[int, float]]
    engine_type: str

    def __setitem__(self, key: int, value: Cycle): ...
    def __getitem__(self, key: int): ...
    def initialize_cost(self, x: Union[int, float]) -> None: ...
    def compare_last_two_cycles(self): ...
    @property
    def name(self) -> str: ...
    def save_results(self, filename: str = None) -> str: ...
    def save_csv(self, config_name: Optional[str] = None) -> str: ...
