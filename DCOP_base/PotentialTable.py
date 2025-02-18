from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Any, Hashable, Union, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from DCOP_base.node import Node
from utils.conversions import to_ndarray
from utils.randomes import create_random_table, create_random_message


class CostTable():
    def __init__(self,domain_size:int,table:Union[np.ndarray,Dict[Hashable,Any],pd.DataFrame]|None=None):
        if table is None:
            self._data = create_random_table((domain_size,domain_size))
        else:
            self._data = to_ndarray(table)
        self.domain = self._data.shape

    def __array__(self)->np.ndarray:
        return self._data

    def __getitem__(self, item):
        return self._data[item]
    def __add__(self,other)->'CostTable':
        if isinstance(other,(int,float)):
            return CostTable(self.domain[0],self._data+other)
        return CostTable(self.domain[0],self._data+other._data)
    def __sub__(self, other)->'CostTable':
        if isinstance(other,(int,float)):
            return CostTable(self.domain[0],self._data-other)
        return CostTable(self.domain[0],self._data-other._data)
    def __mul__(self, other)->'CostTable':
        return CostTable(self.domain[0],self._data*other._data)
    @property
    def mean_cols(self)->np.ndarray:
        '''mean of each column'''
        return np.mean(self._data,axis=0)
    @property
    def mean_rows(self)->np.ndarray:
        '''mean of each row'''
        return np.mean(self._data,axis=1)
    @property
    def mean(self)->float:
        '''mean of the _data'''
        return np.mean(self._data)
    @property
    def total_std(self)->float:
        '''total_std of the _data'''
        return np.std(self._data)

class Message():
    def __init__(self,message:np.ndarray|None=None,domain:int|None=None,source:Node|None=None,target:Node|None=None):
        self.source = source
        self.target = target
        self.domain = domain if domain is not None else 3
        if message:
            self._data = create_random_message(domain if domain is not None else 3)

    def __str__(self)->str:
        return f'Message from {self.source} to {self.target}:\n{self.message}'

    def __getitem__(self, item):
        return self._data[item]
    def __add__(self,other)->Union['Message',CostTable]:
        if isinstance(other,(int,float)):
            return Message(self.domain,self._data+other)
        if isinstance(other,CostTable):
            return Message(self._data+other._data,self.domain)
    def __sub__(self, other)->'CostTable':
        if isinstance(other,(int,float)):
            return CostTable(self.domain[0],self._data-other)
        return CostTable(self.domain[0],self._data-other._data)
    def __mul__(self, other)->'CostTable':
        return CostTable(self.domain[0],self._data*other._data)