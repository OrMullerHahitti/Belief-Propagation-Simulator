# interfaces.py
from abc import ABC, abstractmethod


class DampingPolicy(ABC):
    @abstractmethod
    def get_damping_factor(self, iteration: int) -> float:
        """
        Return the damping factor for the given iteration.
        """
        pass


class CostReductionPolicy(ABC):
    @abstractmethod
    def should_apply(self, iteration: int) -> bool:
        """
        Return True if cost reduction should be applied at this iteration.
        """
        pass

    @abstractmethod
    def get_K(self, iteration: int) -> float:
        """
        Return the cost-reduction multiplier (K) for the given iteration.
        """
        pass


class Updator(ABC):
    @abstractmethod
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

class Node(ABC):
    def __init__(self, name,type):
        self.name = name
        self.neighbors = []
        self.type = type

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    __eq__ = lambda self, other: self.type == other.type and self.name == other.name
