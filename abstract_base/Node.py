from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, ForwardRef, Any

NodeRef = ForwardRef('Node')
class Node(ABC,BaseModel):
    ''' Base class for a generic node in a graph
    Attributes
    ----------
    name : str
        The name of the node.
    type : str
        The type of the node (e.g., 'Variable', 'Factor', etc.)
    neighbor
    s : list
        List of neighboring nodes in the graph. Neighbors are other Node objects.
    Methods
    -------
    add_neighbor(neighbor)
        Add a neighboring node to the list of neighbors.
    __eq__(other)
        Return True if the type and name of the other node are the same as this node.
    __hash__()
        Return a hash value for the node based on its type and name. '''
    name : str
    type : str
    # TODO fix the validation here
    neighbors: List = Field(default_factory=list)
    @abstractmethod
    def add_neighbor(self, neighbor,policy=neighbor_adding_policy|None = None):
        ''' Add a neighboring node to the list of neighbors.
        Parameters
        ----------
        neighbor : Node
            The neighboring node to add to the list of neighbors.
        '''
        pass


    def __eq__ (self, other):
        return self.type == other.type and self.name == other.name
    __hash__ = lambda self: hash((self.type, self.name))

    class config:
        arbitrary_types_allowed = True