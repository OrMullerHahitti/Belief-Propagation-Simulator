# stopping.py
import numpy as np
from abstract_base.interfaces import StoppingCriterion


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
