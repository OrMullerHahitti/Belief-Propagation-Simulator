from typing import TypeVar, Protocol

class BPAgent(Protocol):
    """Base protocol type for Belief Propagation agents"""
    name: str
    domain: int
    mailbox: dict
#TODO : add more protocols!