from typing import List

from bp_base.agents import FactorAgent


def init_normalization(li: List[FactorAgent]):
    x = len(li)
    for factor in li:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table / x
