from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Any
from pydantic import BaseModel, Field
class PotentialTable(ABC,BaseModel):
    ''' Abstract base class for a potential table.
    Attributes
    ----------
    table : dict
        The potential table, represented as a dictionary.
    Methods
    -------
    get_value(key)
        Return the value of the potential table at the given key.
    '''
    table : dict
    @abstractmethod
    def get_value(self, key):
        ''' Return the value of the potential table at the given key.
        Parameters
        ----------
        key : tuple
            The key for the potential table.
        '''
        pass