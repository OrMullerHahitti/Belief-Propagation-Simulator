from abc import ABC
from typing import Dict
import numpy as np

from bp_variations.factor_graph import FactorGraph


class BeliefPropagationEngine(ABC):
    """
    Abstract engine for belief propagation.
    """
    def __init__(self, factor_graph: FactorGraph):
        self.factor_graph = factor_graph
        self.Q: Dict[tuple, np.ndarray] = {}
        self.R: Dict[tuple, np.ndarray] = {}

    @abstractmethod
    def run_inference(self, max_iters: int = 10):
        pass

    @abstractmethod
    def get_beliefs(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_map_estimate(self) -> Dict[str, int]:
        pass