# policies.py
from abstract_base.interfaces import DampingPolicy, CostReductionPolicy



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
