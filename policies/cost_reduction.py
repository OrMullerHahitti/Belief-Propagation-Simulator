#cost reduction policy for the factor node
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, Any

from DCOP_base import Agent
Iteration :TypeAlias = Any #for now only a place holder
'''
this module is mostly for config!!
'''
class CostReductionPolicy():
    def __init__(self, stopping_critiria:Callable|None=None,applying_critiria:Callable|None=None):
        self.calc_k = stopping_critiria
        self.applying_critiria = applying_critiria


    def should_apply(self, iteration: Iteration) -> bool:
        if not self.applying_critiria:
            return False
        return self.applying_critiria(Iteration)




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
