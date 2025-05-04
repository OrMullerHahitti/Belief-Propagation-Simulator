from bp_base.factor_graph import FactorGraph
from bp_policies import SplittingPolicy

class ConstantSplittingPolicy(SplittingPolicy):
    """
    Constant splitting policy.
    """
    def __init__(self, splitting_value: float,factor_graph: FactorGraph):
        super().__init__(factor_graph)
        self.splitting_value = splitting_value

    def _get_splitting(self) -> float:
        for factor in self.factor:
            factor.update_cost_table = factor.cost_table * self.splitting_value