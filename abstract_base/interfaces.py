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


class Updator(ABC):
    @abstractmethod
    @validate_arguments
    def schedule_updates(self, Q_keys, R_keys, iteration: int):
        """
        Given the lists of edges for Q and R, return the order in which they should be updated.
        In synchronous mode, might return something like [('Q', vName, fName), ...] for all Q
        then [('R', fName, vName), ...], or do them in random order, etc.

        Must produce an iterable of tuples of the form:
           ('Q', vName, fName) or ('R', fName, vName)
        indicating the update type and the node names.
        """
        pass


class StoppingCriterion(ABC):
    @abstractmethod
    def should_stop(self, iteration: int, max_iters: int, Q, R, Q_old, R_old) -> bool:
        """
        Return True if inference should stop. For example, check if iteration >= max_iters,
        or if messages have converged, etc.

        :param iteration: current iteration count (0-based).
        :param max_iters: user-specified maximum iteration limit.
        :param Q, R: current message dicts
        :param Q_old, R_old: previous iteration message dicts
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