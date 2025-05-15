from bp_base.bp_engine import BPEngine
from policies.cost_reduction import cost_reduction_all_factors_one_one

from policies.splitting import split_all_factors
from policies.damping import damp


class SplitEngine(BPEngine):
    def __init__(self, *args, p: float = 0.5, **kwargs):
        self.p = p
        super().__init__(*args, **kwargs)

    def post_init(self) -> None:
        split_all_factors(self.graph, self.p)


class DampingEngine(BPEngine):
    def __init__(self, *args, damping_factor: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.damping_factor = damping_factor

    def post_var_cycle(self):
        damp(self.var_nodes, self.damping_factor)


class CostReductionOnceEngine(BPEngine):
    def __init__(self, *args, p: float = 0.5, **kwargs):
        self.cr = p
        super().__init__(*args, **kwargs)

    def post_two_cycles(self):
        cost_reduction_all_factors_one(self.factor_nodes, self.cr)


# cost reduction and damping
class CostReductionAndDamping(CostReductionOnceEngine, DampingEngine):
    def __init__(self, *args, p: float = 0.5, damping_factor: float = 0.9, **kwargs):
        self.cr = p
        self.damping_factor = damping_factor
        super().__init__(*args, **kwargs)


class DampingAndSplitting(SplitEngine, DampingEngine):
    def __init__(self, *args, p: float = 0.5, damping_factor: float = 0.9, **kwargs):
        self.damping_factor = damping_factor
        self.p = p
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    SplitEngine()
