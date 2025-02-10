from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, TYPE_CHECKING

# Only necessary for forward references in older Python versions (<3.9)
# or if you're using "neighbors: List['Node']"
if TYPE_CHECKING:
    from abstract_base.node import Node  # We name it carefully to avoid circular import
    # or place this in a separate module if needed

from abstract_base.interfaces import NeighborAddingPolicy


class Node(ABC):
    """
    Base class for a generic node in a graph.

    Attributes
    ----------
    name : str
        The name of the node.
    type : str
        The type of the node (e.g., 'Variable', 'Factor', etc.)
    neighbors : list
        List of neighboring nodes in the graph. Neighbors are other Node objects.

    Methods
    -------
    add_neighbor(neighbor)
        Add a neighboring node to the list of neighbors.
    __eq__(other)
        Return True if the type and name of the other node are the same as this node.
    __hash__()
        Return a hash value for the node based on its type and name.
    """
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self.neighbors :List['Node'] = []


    def add_neighbor(self, neighbor, policy: NeighborAddingPolicy | None = None) ->None:
        """
        Add a neighboring node to the list of neighbors.

        Parameters
        ----------
        neighbor : Node
            The neighboring node to add to the list of neighbors.
            :param neighbor:
            :param policy:
        """
        pass

    __eq__ = lambda self, other: self.type == other.type and self.name == other.name
    __hash__ = lambda self: hash((self.type, self.name))

    class Config:
        arbitrary_types_allowed = True
    class FactorNode(Node):
        """
        A class to represent a factor node in a factor graph.

        Attributes
        ----------
        domain : list
            The domain of the factor node.
        var_names : list
            The names of the variable nodes connected to this factor node.

        Methods
        -------
        add_neighbor(neighbor)
            Add a neighboring node to the list of neighbors.
        """
        def __init__(self,name,type):
            super(name, type)

        def add_neighbor(self, neighbor, policy: NeighborAddingPolicy | None = None):
            """
            Add a neighboring node to the list of neighbors.

            Parameters
            ----------
            neighbor : Node
                The neighboring node to add to the list of neighbors.
            """
            if policy is not None:
                policy.add_neighbors(self, neighbor)
            else:
                if neighbor not in self.neighbors:
                    self.neighbors.append(neighbor)
                    neighbor.add_neighbor(self)
