# policies.py
import math
from interfaces import DampingPolicy, CostReductionPolicy

class ConstantDampingPolicy(DampingPolicy):
    def __init__(self, damping_value: float):
        self.damping_value = damping_value

    def get_damping(self, iteration: int) -> float:
        return self.damping_value


class DynamicDampingPolicy(DampingPolicy):
    """
    Example: damping(t) = min(0.9, 0.1 * t)
    just for illustration.
    """
    def get_damping(self, iteration: int) -> float:
        return min(0.9, 0.1 * iteration)


class ConstantCostReductionPolicy(CostReductionPolicy):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def should_apply(self, iteration: int) -> bool:
        return True

    def get_alpha(self, iteration: int) -> float:
        return self.alpha


class FirstIterationCostReductionPolicy(CostReductionPolicy):
    """
    Apply cost reduction only at iteration 0.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha

    def should_apply(self, iteration: int) -> bool:
        return (iteration == 0)

    def get_alpha(self, iteration: int) -> float:
        return self.alpha
