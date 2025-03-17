#cost reduction policy for the factor node
from abc import ABC, abstractmethod
from typing import Callable

from DCOP_base.agent import Agent


class CostReductionPolicy(ABC):
    def __init__(self, stopping_critiria:Callable[]):
    @abstractmethod
    def should_apply(self, iteration: int) -> bool:
        """
        Return True if cost reduction should be applied at this iteration.
        """
        pass

    @abstractmethod
    def get_K(self, agent:Agent|None=None) -> float:
        """
        Return the cost-reduction multiplier (K) for the given iteration.
    """
        pass

class EveryTimeCostReduction(CostReductionPolicy):
    def should_apply(self, iteration: int) -> bool:
        return True
    def get_K(self) -> float:
        return 0.5

class MinMaxEnvelopeCostReduction(CostReductionPolicy):
    def should_apply(self, iteration: int) -> bool:
        return iteration % 2 == 0
