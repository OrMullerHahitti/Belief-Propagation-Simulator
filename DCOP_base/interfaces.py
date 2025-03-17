# interfaces.py
from abc import ABC, abstractmethod
from pydantic import BaseModel,validate_arguments
import numpy as np
from typing import List
from typing import Annotated

class DampingPolicy(ABC):
    @abstractmethod
    def get_damping_factor(self, iteration: int) -> float:
        """
        Return the damping factor for the given iteration.
        """
        pass

class CostReductionPolicy(ABC):
    @abstractmethod
    @validate_arguments
    def should_apply(self, iteration: int) -> bool:
        """
        Return True if cost reduction should be applied at this iteration.
        """
        pass

    @abstractmethod
    @validate_arguments
    def get_K(self, iteration: int) -> float:
        """
        Return the cost-reduction multiplier (K) for the given iteration.
        """
        pass







class MessageUpdateRule(ABC):
    """
    Defines how to compute Q and R messages for a specific variant
    of Belief Propagation (e.g., Min-Sum, Max-Sum, Sum-Product, etc.).
    """

    @abstractmethod
    def compute_Q(
        self,
        v_name : str,
        f_name : str,
        factor_graph,
        Q_messages,
        R_messages,
        iteration
    ) -> np.ndarray:
        """
        Compute the Q_{v->f}(x_v) message vector.
        """
        pass

    @abstractmethod
    def compute_R(
        self,
        f_name,
        v_name,
        factor_graph,
        Q_messages,
        R_messages,
        iteration
    ) -> np.ndarray:
        """
        Compute the R_{f->v}(x_v) message vector.
        """
        pass

class NeighborAddingPolicy(ABC):
    """
    Abstract class for adding neighbours to nodes in the factor graph.
    """
    @abstractmethod
    def add_neighbors(self, node ,other) -> bool:
        """
        Given the factor graph, add neighbours to each node.
        """
        pass