import random
from typing import Dict, Optional

import networkx as nx
import numpy as np

from ..core.agents import VariableAgent, FactorAgent
from .engine_base import BPEngine
from ..policies.cost_reduction import (
    cost_reduction_all_factors_once,
    discount_attentive,
)
from ..policies.splitting import split_all_factors
from ..policies import damp
from ..utils.inbox_utils import multiply_messages_attentive
from propflow.bp.engine_base import BPEngine
from propflow.core.components import Message


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
        self._set_name({"split-": f"{self.split_factor}-{self.split_factor}"})

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


class DiffusionEngine(BPEngine):
    """A BP engine that applies spatial message diffusion.

    Unlike damping which blends messages across time (current vs previous),
    diffusion blends messages across space (local vs neighbors) at each iteration.
    This can help smooth the optimization landscape and improve convergence on
    densely connected graphs.
    """

    def __init__(self, *args, alpha: float = 0.3, **kwargs):
        """Initializes the DiffusionEngine.

        Args:
            *args: Positional arguments for the base `BPEngine`.
            alpha: Diffusion coefficient in [0, 1]. Higher values = more smoothing.
                - alpha=0: no diffusion (pure BP)
                - alpha=0.1-0.3: recommended range for most problems
                - alpha=1: complete averaging (may lose local information)
                Defaults to 0.3.
            **kwargs: Keyword arguments for the base `BPEngine`.
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        super().__init__(*args, **kwargs)
        self._name = "DiffusionEngine"
        self._set_name({"alpha": str(self.alpha)})

    def post_var_compute(self, var: VariableAgent) -> None:
        """Apply spatial diffusion to Q-messages (variable → factor).

        For each message this variable sends to a factor, blend it with
        messages from other variables connected to the same factor.
        """
        if self.alpha == 0:
            return  # No diffusion needed

        # For each message in this variable's outbox
        for msg in var.outbox:
            target_factor = msg.recipient

            # Collect Q-messages from OTHER variables connected to same factor
            neighbor_msgs = []
            for neighbor in self.graph.G.neighbors(target_factor):
                # Neighbors of a factor are variables
                if neighbor != var:  # Skip self
                    # Find if this neighbor variable has a message to the same factor
                    for neighbor_msg in neighbor.outbox:
                        if neighbor_msg.recipient == target_factor:
                            neighbor_msgs.append(neighbor_msg.data)
                            break

            # Apply diffusion if neighbors exist
            if neighbor_msgs:
                neighbor_avg = np.mean(neighbor_msgs, axis=0)
                # Blend: (1-α) × local + α × neighbor_average
                msg.data = (1 - self.alpha) * msg.data + self.alpha * neighbor_avg

    def post_factor_compute(self, factor: FactorAgent, iteration: int) -> None:
        """Apply spatial diffusion to R-messages (factor → variable).

        For each message this factor sends to a variable, blend it with
        messages from other factors connected to the same variable.
        """
        if self.alpha == 0:
            return  # No diffusion needed

        # For each message in this factor's outbox
        for msg in factor.outbox:
            target_var = msg.recipient

            # Collect R-messages from OTHER factors connected to same variable
            neighbor_msgs = []
            for neighbor in self.graph.G.neighbors(target_var):
                # Neighbors of a variable are factors
                if neighbor != factor:  # Skip self
                    # Find if this neighbor factor has a message to the same variable
                    for neighbor_msg in neighbor.outbox:
                        if neighbor_msg.recipient == target_var:
                            neighbor_msgs.append(neighbor_msg.data)
                            break

            # Apply diffusion if neighbors exist
            if neighbor_msgs:
                neighbor_avg = np.mean(neighbor_msgs, axis=0)
                # Blend: (1-α) × local + α × neighbor_average
                msg.data = (1 - self.alpha) * msg.data + self.alpha * neighbor_avg


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


class TRWEngine(BPEngine):
    """
    Tree-Reweighted Belief Propagation engine (Min-Sum variant).

    The engine keeps the standard Min-Sum computator but automatically:

        1. Samples spanning trees over the variable-only (primal) graph to
           estimate per-factor appearance probabilities ``rho_f``.
        2. Scales each factor's energy table by ``1 / rho_f`` before message
           computation so local costs match the TRW objective.
        3. Re-weights outgoing R-messages from factors by ``rho_f`` so that
           variable updates/beliefs operate on appropriately weighted costs.

    Rho sampling and scaling can be overridden by providing explicit
    ``factor_rhos`` (all > 0). Otherwise the engine performs end-to-end
    TRW reweighting using the current factor graph structure.
    """

    DEFAULT_MIN_RHO = 1e-6

    def __init__(
        self,
        *args,
        factor_rhos: Optional[Dict[str, float]] = None,
        tree_sample_count: int = 64,
        tree_sampler_seed: Optional[int] = None,
        min_rho: float = DEFAULT_MIN_RHO,
        **kwargs,
    ) -> None:
        """
        Args:
            factor_rhos:
                Optional explicit mapping from factor name to rho_f > 0. When
                omitted, the engine estimates rhos via spanning-tree sampling.
            tree_sample_count:
                Number of spanning trees to sample when estimating rhos.
            tree_sampler_seed:
                Seed forwarded to the tree sampler for reproducibility.
            min_rho:
                Lower bound applied to sampled rhos to keep them strictly
                positive (important for stable scaling).
        """
        self.tree_sample_count = max(1, int(tree_sample_count))
        self.tree_sampler_seed = tree_sampler_seed
        self.min_rho = max(float(min_rho), self.DEFAULT_MIN_RHO)
        self._user_defined_rhos = bool(factor_rhos)
        self.factor_rhos: Dict[str, float] = dict(factor_rhos or {})

        super().__init__(*args, **kwargs)
        self._name = "TRWEngine"
        suffix = "custom" if self._user_defined_rhos else f"trees-{self.tree_sample_count}"
        self._set_name({"trw": suffix})

    def post_init(self) -> None:
        """
        Validate rho configuration, sample if needed, and scale costs.

        Called from BPEngine.__init__ after `self.graph` is set but before
        messages are initialized.
        """
        factors = getattr(self.graph, "factors", [])
        if not factors:
            return

        if not self.factor_rhos:
            self.factor_rhos = self._estimate_rhos_via_spanning_trees(factors)
        else:
            for factor in factors:
                self.factor_rhos.setdefault(factor.name, 1.0)

        for factor in factors:
            rho = self.factor_rhos.get(factor.name, 1.0)
            if rho <= 0:
                raise ValueError(
                    f"TRWEngine: rho for factor '{factor.name}' must be > 0, got {rho}"
                )
            self._scale_factor_cost_table(factor, rho)

    def post_factor_compute(self, factor: FactorAgent, iteration: int) -> None:
        """Scale outgoing R-messages by rho_f before they are sent."""
        rho = self.factor_rhos.get(factor.name, 1.0)
        if rho == 1.0 or not factor.mailer.outbox:
            return

        for msg in factor.mailer.outbox:
            msg.data = rho * msg.data

    # --- Internal helpers -------------------------------------------------

    def _scale_factor_cost_table(self, factor: FactorAgent, rho: float) -> None:
        """Reset to the original cost table (if saved) and divide by rho."""
        factor.save_original()
        base = factor.original_cost_table if factor.original_cost_table is not None else factor.cost_table
        if base is None:
            return
        factor.cost_table = base / rho

    def _estimate_rhos_via_spanning_trees(
        self, factors: list[FactorAgent]
    ) -> Dict[str, float]:
        """Compute rho_f by sampling spanning trees on the primal graph."""
        primal_graph, edge_to_factors = self._build_primal_graph()
        if (
            primal_graph.number_of_edges() == 0
            or primal_graph.number_of_nodes() == 0
            or not nx.is_connected(primal_graph)
        ):
            # Fallback: no usable topology information, keep uniform rhos.
            return {factor.name: 1.0 for factor in factors}

        counts = {factor.name: 0 for factor in factors}
        rng = random.Random(self.tree_sampler_seed)
        samples = max(1, self.tree_sample_count)

        for _ in range(samples):
            seed = rng.randint(0, 2**32 - 1)
            tree = nx.random_spanning_tree(primal_graph, seed=seed)
            for node_u, node_v in tree.edges():
                key = tuple(sorted((node_u, node_v)))
                for factor in edge_to_factors.get(key, []):
                    counts[factor.name] += 1

        rhos: Dict[str, float] = {}
        for factor in factors:
            count = counts.get(factor.name, 0)
            rho = count / samples if count > 0 else 0.0
            if rho <= 0:
                rho = self.min_rho
            rhos[factor.name] = rho

        return rhos

    def _build_primal_graph(self) -> tuple[nx.Graph, Dict[tuple[str, str], list[FactorAgent]]]:
        """Construct the variable-only graph used for tree sampling."""
        graph = nx.Graph()
        variables = getattr(self.graph, "variables", [])
        graph.add_nodes_from(var.name for var in variables)

        edge_to_factors: Dict[tuple[str, str], list[FactorAgent]] = {}
        for factor in getattr(self.graph, "factors", []):
            var_names = sorted(factor.connection_number.keys())
            if len(var_names) != 2:
                # Hyper-edges are left unweighted (rho defaults to 1).
                continue
            edge_key = (var_names[0], var_names[1])
            graph.add_edge(*edge_key)
            edge_to_factors.setdefault(edge_key, []).append(factor)

        # Guard against multiple factors per variable pair by treating them
        # as parallel edges (each receives credit whenever the pair is picked).
        return graph, edge_to_factors

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
