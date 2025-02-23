from typing import Sequence
from pydantic import Field

from DCOP_base.agent import Node
from DCOP_base.interfaces import NeighborAddingPolicy
from policies.node_add_policy import BPNeighborAddingPolicy


class VariableNode(Node):
    '''
    A class to represent a variable node in a factor graph.

    '''
    type :str = Field(default="Variable")
    # If you need a domain that is a sequence of ints/floats:
    # - Python 3.10+ supports int | float
    # - For older versions, use Union[int, float]
    domain: list[int | float] = Field(default_factory=list)

    def add_neighbor(self, neighbor, policy: NeighborAddingPolicy = BPNeighborAddingPolicy()):
        policy.add_neighbors(self, neighbor)


class FactorNode(Node):
    type: str = Field(default="Factor")
    domain: list[int | float] = Field(default_factory=list)
    var_names: list[str] = Field(default_factory=list)

    def add_neighbor(self, neighbor, policy: NeighborAddingPolicy = BPNeighborAddingPolicy()):
        policy.add_neighbors(self, neighbor)
