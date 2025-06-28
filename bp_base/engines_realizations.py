from statsmodels.multivariate.factor import Factor

from base_all.agents import VariableAgent, FactorAgent
from bp_base.bp_engine_base import BPEngine
from policies.cost_reduction import (
    cost_reduction_all_factors_once,
    discount_attentive,
)

from policies.splitting import split_all_factors
from policies.damping import TD, damp
from utils.inbox_utils import double_messages


class SplitEngine(BPEngine):
    def __init__(self, *args, split_factor: float = 0.6, **kwargs):
        self.split_factor = split_factor
        super().__init__(*args, **kwargs)
        self._name = "SPFGEngine"
        self._set_name({"split-": f"{str(self.split_factor)}-{str(self.split_factor)}"})

    def post_init(self) -> None:
        split_all_factors(self.graph, self.split_factor)


class TDEngine(BPEngine):
    def __init__(self, *args, damping_factor: float = 0.9, **kwargs):
        self.damping_factor = damping_factor
        super().__init__(*args, **kwargs)

    def post_var_cycle(self):
        TD(self.var_nodes, self.damping_factor)


class CostReductionOnceEngine(BPEngine):
    def __init__(self, *args, reduction_factor: float = 0.5, **kwargs):
        self.reduction_factor = reduction_factor
        super().__init__(*args, **kwargs)

    def post_init(self):
        cost_reduction_all_factors_once(self.graph, self.reduction_factor)

    def pre_factor_compute(self, factor: FactorAgent):
        double_messages(factor.inbox)


class DampingEngine(BPEngine):
    def __init__(self, *args, damping_factor: float = 0.9, **kwargs):
        self.damping_factor = damping_factor
        super().__init__(*args, **kwargs)
        self._name = "DampingEngine"
        self._set_name({"damping": str(self.damping_factor)})

    def post_var_compute(self, var: VariableAgent):
        damp(var, self.damping_factor)
        var.append_last_iteration()


class DampingSCFGEngine(DampingEngine, SplitEngine):
    """BP Engine with damping and splitting."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("split_factor", 0.6)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)
        self.split_factor = kwargs.get("split_factor", 0.6)  # Ensure attribute exists
        self._name = "DampingSCFG"
        self._set_name(
            {
                "split": f"{str(self.split_factor)}-{str(1-self.split_factor)}",
                "damping": "0.9",
            }
        )


class DampingCROnceEngine(DampingEngine, CostReductionOnceEngine):
    """BP Engine with damping and discounting."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reduction_factor", 0.5)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)
        self.reduction_factor = kwargs.get(
            "reduction_factor", 0.5
        )  # Ensure attribute exists
        self._name = "DampingCROnceEngine"
        self._set_name(
            {
                "split": f"{str(self.reduction_factor)}-{str(1-self.reduction_factor)}",
                "damping": "0.9",
            }
        )


class MessagePruningEngine(BPEngine):
    """BP Engine with message pruning to reduce memory usage."""

    def __init__(
        self,
        *args,
        prune_threshold: float = 1e-4,
        min_iterations: int = 5,
        adaptive_threshold: bool = True,
        **kwargs,
    ):
        self.prune_threshold = prune_threshold
        self.min_iterations = min_iterations
        self.adaptive_threshold = adaptive_threshold
        super().__init__(*args, **kwargs)

    def post_init(self) -> None:
        """Initialize message pruning policy."""
        from policies.message_pruning import MessagePruningPolicy

        self.pruning_policy = MessagePruningPolicy(
            prune_threshold=self.prune_threshold,
            min_iterations=self.min_iterations,
            adaptive_threshold=self.adaptive_threshold,
        )


class TDAndPruningEngine(TDEngine, MessagePruningEngine):
    """Combined TD damping and message pruning engine."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("prune_threshold", 1e-4)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)


class DiscountEngine(BPEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_factor_cycle(self):
        discount_attentive(self.graph)
