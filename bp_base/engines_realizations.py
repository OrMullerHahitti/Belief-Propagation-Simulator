from bp_base.agents import VariableAgent
from bp_base.bp_engine_base import BPEngine
from policies.cost_reduction import cost_reduction_all_factors_once, discount

from policies.splitting import split_all_factors
from policies.damping import TD, damp


class SplitEngine(BPEngine):
    def __init__(self, *args, p: float = 0.5, **kwargs):
        self.p = p
        super().__init__(*args, **kwargs)

    def post_init(self) -> None:
        split_all_factors(self.graph, self.p)


class TDEngine(BPEngine):
    def __init__(self, *args, damping_factor: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.damping_factor = damping_factor

    def post_var_cycle(self):
        TD(self.var_nodes, self.damping_factor)


class CostReductionOnceEngine(BPEngine):
    def __init__(self, *args, p: float = 0.5, **kwargs):
        self.cr = p
        super().__init__(*args, **kwargs)

    def post_two_cycles(self):
        cost_reduction_all_factors_once(self.factor_nodes, self.cr)


# cost reduction and damping
class CostReductionAndTD(CostReductionOnceEngine, TDEngine):
    def __init__(self, *args, p: float = 0.5, damping_factor: float = 0.9, **kwargs):
        self.cr = p
        self.damping_factor = damping_factor
        super().__init__(*args, **kwargs)


class TDAndSplitting(SplitEngine, TDEngine):
    def __init__(self, *args, p: float = 0.3, damping_factor: float = 0.9, **kwargs):
        kwargs.setdefault("discount_factor", 0.995)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)


class DiscountEngine(BPEngine):
    def __init__(self, *args, discount_factor: float = 0.9, **kwargs):
        self.discount_factor = discount_factor
        super().__init__(*args, **kwargs)

    def post_factor_cycle(self):
        discount(self.factor_nodes, self.discount_factor)


class TDAndDiscountBPEngine(TDEngine, DiscountEngine):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("discount_factor", 0.99)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)


class DampingEngine(BPEngine):
    def __init__(self, *args, damping_factor: float = 0.1, **kwargs):
        self.damping_factor = damping_factor
        super().__init__(*args, **kwargs)

    def post_var_compute(self, var: VariableAgent):
        damp(var, self.damping_factor)
        var.append_last_iteration()
