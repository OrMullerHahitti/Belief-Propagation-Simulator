from ..core.agents import VariableAgent, FactorAgent
from .engine_base import BPEngine
from ..policies.cost_reduction import (
    cost_reduction_all_factors_once,
    discount_attentive,
)

from ..policies.splitting import split_all_factors
from ..policies import damp
from ..utils.inbox_utils import multiply_messages_attentive


class Engine(BPEngine):
    """A basic belief propagation engine.

    This is a direct alias for `BPEngine` and provides the standard,
    unmodified belief propagation behavior.
    """
    ...


class SplitEngine(BPEngine):
    """A BP engine that applies the factor splitting policy.

    This engine modifies the factor graph by splitting each factor into two,
    distributing the original cost between them. This can sometimes help with
    convergence.
    """
    def __init__(self, *args, split_factor: float = 0.6, **kwargs):
        """Initializes the SplitEngine.

        Args:
            *args: Positional arguments for the base `BPEngine`.
            split_factor: The proportion of the cost to allocate to the first
                of the two new factors. Defaults to 0.6.
            **kwargs: Keyword arguments for the base `BPEngine`.
        """
        self.split_factor = split_factor
        super().__init__(*args, **kwargs)
        self._name = "SPFGEngine"
        self._set_name({"split-": f"{str(self.split_factor)}-{str(self.split_factor)}"})

    def post_init(self) -> None:
        """Applies the factor splitting policy after initialization."""
        split_all_factors(self.graph, self.split_factor)


class CostReductionOnceEngine(BPEngine):
    """A BP engine that applies a one-time cost reduction policy.

    This engine reduces the costs in the factor tables at the beginning of the
    simulation and then applies a discount to outgoing messages from factors.
    """
    def __init__(self, *args, reduction_factor: float = 0.5, **kwargs):
        """Initializes the CostReductionOnceEngine.

        Args:
            *args: Positional arguments for the base `BPEngine`.
            reduction_factor: The factor by which to reduce costs.
                Defaults to 0.5.
            **kwargs: Keyword arguments for the base `BPEngine`.
        """
        self.reduction_factor = reduction_factor
        super().__init__(*args, **kwargs)

    def post_init(self):
        """Applies the one-time cost reduction after initialization."""
        cost_reduction_all_factors_once(self.graph, self.reduction_factor)

    def post_factor_compute(self, factor: FactorAgent, iteration: int):
        """Applies a discount to outgoing messages from factors."""
        multiply_messages_attentive(factor.outbox, 0.5, iteration)


class DampingEngine(BPEngine):
    """A BP engine that applies message damping.

    Damping averages the message from the previous iteration with the newly
    computed message. This can help prevent oscillations and improve convergence.
    """
    def __init__(self, *args, damping_factor: float = 0.9, **kwargs):
        """Initializes the DampingEngine.

        Args:
            *args: Positional arguments for the base `BPEngine`.
            damping_factor: The weight given to the previous message.
                Defaults to 0.9.
            **kwargs: Keyword arguments for the base `BPEngine`.
        """
        self.damping_factor = damping_factor
        super().__init__(*args, **kwargs)
        self._name = "DampingEngine"
        self._set_name({"damping": str(self.damping_factor)})

    def post_var_compute(self, var: VariableAgent):
        """Applies damping after a variable node computes its messages."""
        damp(var, self.damping_factor)
        var.append_last_iteration()


class DampingSCFGEngine(DampingEngine, SplitEngine):
    """A BP engine that combines message damping and factor splitting."""

    def __init__(self, *args, **kwargs):
        """Initializes the DampingSCFGEngine.

        This engine inherits parameters from both `DampingEngine` and `SplitEngine`.

        Args:
            *args: Positional arguments for the base engines.
            **kwargs: Keyword arguments for the base engines (e.g.,
                `damping_factor`, `split_factor`).
        """
        kwargs.setdefault("split_factor", 0.6)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)
        self.split_factor = kwargs.get("split_factor", 0.6)
        self._name = "DampingSCFG"
        self._set_name(
            {
                "split": f"{str(self.split_factor)}-{str(1-self.split_factor)}",
                "damping": "0.9",
            }
        )


class DampingCROnceEngine(DampingEngine, CostReductionOnceEngine):
    """A BP engine that combines message damping and one-time cost reduction."""

    def __init__(self, *args, **kwargs):
        """Initializes the DampingCROnceEngine.

        This engine inherits parameters from `DampingEngine` and
        `CostReductionOnceEngine`.

        Args:
            *args: Positional arguments for the base engines.
            **kwargs: Keyword arguments for the base engines (e.g.,
                `damping_factor`, `reduction_factor`).
        """
        kwargs.setdefault("reduction_factor", 0.5)
        kwargs.setdefault("damping_factor", 0.9)
        super().__init__(*args, **kwargs)
        self.reduction_factor = kwargs.get("reduction_factor", 0.5)
        self._name = "DampingCROnceEngine"
        self._set_name(
            {
                "split": f"{str(self.reduction_factor)}-{str(1-self.reduction_factor)}",
                "damping": "0.9",
            }
        )


class MessagePruningEngine(BPEngine):
    """A BP engine that applies a message pruning policy to reduce memory usage."""

    def __init__(
        self,
        *args,
        prune_threshold: float = 1e-4,
        min_iterations: int = 5,
        adaptive_threshold: bool = True,
        **kwargs,
    ):
        """Initializes the MessagePruningEngine.

        Args:
            *args: Positional arguments for the base `BPEngine`.
            prune_threshold: The threshold below which messages are pruned.
            min_iterations: The number of iterations to wait before pruning.
            adaptive_threshold: Whether to adapt the threshold dynamically.
            **kwargs: Keyword arguments for the base `BPEngine`.
        """
        self.prune_threshold = prune_threshold
        self.min_iterations = min_iterations
        self.adaptive_threshold = adaptive_threshold
        super().__init__(*args, **kwargs)

    def post_init(self) -> None:
        """Initializes and sets the message pruning policy on agent mailers."""
        from ..policies.message_pruning import MessagePruningPolicy

        self.pruning_policy = MessagePruningPolicy(
            prune_threshold=self.prune_threshold,
            min_iterations=self.min_iterations,
            adaptive_threshold=self.adaptive_threshold,
        )


class DiscountEngine(BPEngine):
    """A BP engine that applies a discount factor to messages over time."""
    def __init__(self, *args, **kwargs):
        """Initializes the DiscountEngine.

        Args:
            *args: Positional arguments for the base `BPEngine`.
            **kwargs: Keyword arguments for the base `BPEngine`.
        """
        super().__init__(*args, **kwargs)

    def post_factor_cycle(self):
        """Applies the discount policy after each message passing cycle."""
        discount_attentive(self.graph)
