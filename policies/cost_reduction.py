from bp_base.agents import FactorAgent, VariableAgent
from bp_base.components import Message
from typing import Tuple, List, Iterable


def cost_reduction_all_factors_once(fac_a: Iterable[FactorAgent], x: float):
    for factor in fac_a:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table * x


def discount(fac_a: Iterable[FactorAgent], x: float):
    for factor in fac_a:
        if factor.cost_table is not None:
            factor.cost_table = factor.cost_table * x
