from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from bp_base.agents import BPAgent
from bp_base.components import Message
from bp_variations.factor_graph import FactorGraph

class BeliefPropagation(ABC):
    """
    Abstract engine for belief propagation.
    """
    def __init__(self, factor_graph: FactorGraph):
        self.factor_graph = factor_graph
        self.history = Dict[int, List[BPAgent]]  # Store history of beliefs

    @abstractmethod
    def run_inference(self, max_iters: int = 10):
        pass

    @abstractmethod
    def get_beliefs(self) -> Dict[str, np.ndarray]:
        ''' Return the beliefs of the factor graph.
        :return:
        :param: A dictionary mapping variable names to belief vectors.'''
        pass

    @abstractmethod
    def get_map_estimate(self) -> Dict[str, int]:
        pass