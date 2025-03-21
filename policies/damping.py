
from abc import ABC, abstractmethod

from policies.abstract import FactorPolicy, MessagePolicy


class DampingPolicy(MessagePolicy):
    pass
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
