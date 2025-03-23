# stopping.py
from abc import ABC, abstractmethod

import numpy as np

class StoppingCriterion(ABC):
    @abstractmethod
    def should_stop(self, iteration: int, max_iters: int, Q, R, Q_old, R_old) -> bool:
        """
        Return True if inference should stop. For example, check if iteration >= max_iters,
        or if messages have converged, etc.

        :param iteration: current iteration count (0-based).
        :param max_iters: user-specified maximum iteration limit.
        :param Q, R: current data dicts
        :param Q_old, R_old: previous iteration data dicts
        """
        pass
class MaxIterationsStopping(StoppingCriterion):
    """
    Stop after max_iters unconditionally.
    """

    def should_stop(self, iteration: int, max_iters: int, Q, R, Q_old, R_old) -> bool:
        return (iteration >= max_iters)


class DeltaConvergenceStopping(StoppingCriterion):
    """
    Stop if the maximum difference between Q (and R) messages
    from the previous iteration is below some threshold.
    Also stop if iteration >= max_iters.
    """

    def __init__(self, delta_threshold: float):
        self.delta_threshold = delta_threshold

    def should_stop(self, iteration: int, max_iters: int, Q, R, Q_old, R_old) -> bool:
        if iteration >= max_iters:
            return True

        # Compute max difference
        max_diff = 0.0
        for key in Q:
            max_diff = max(max_diff, np.max(np.abs(Q[key] - Q_old[key])))
        for key in R:
            max_diff = max(max_diff, np.max(np.abs(R[key] - R_old[key])))

        return (max_diff < self.delta_threshold)
