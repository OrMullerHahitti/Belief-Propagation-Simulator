# bp_engine.py
from abc import ABC, abstractmethod
import numpy as np

class BeliefPropagationEngine(ABC):
    """
    Abstract base class for a generic Belief Propagation engine.
    Provides skeleton methods: run_inference, get_beliefs, etc.
    The actual message update rules (computeQ, computeR) must be supplied
    by an injected MessageUpdateRule or by the subclass.
    """

    @abstractmethod
    def run_inference(self, max_iters=10):
        pass

    @abstractmethod
    def get_beliefs(self):
        pass

    @abstractmethod
    def get_map_estimate(self):
        pass
