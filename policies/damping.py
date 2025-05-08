from abc import ABC, abstractmethod

from policies.bp_policies import DampingPolicy, PolicyType


class ConstantDampingPolicy(DampingPolicy):
    """
    Constant damping policy.
    """

    def __init__(self, damping_value: float):
        super().__init__(PolicyType.VARIABLE)
        self.damping_value = damping_value

    def _get_damping(self, *args, **kwargs) -> float:
        return self.damping_value
