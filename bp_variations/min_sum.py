from typing import Dict

import numpy as np

from DCOP_base.bp_engine import BeliefPropagationEngine
from DCOP_base.interfaces import DampingPolicy, CostReductionPolicy
from bp_variations.factor_graph import FactorGraph


class MinSumBP(BeliefPropagationEngine):
    def __init__(
        self,
        factor_graph: FactorGraph,
        damping_policy: DampingPolicy | None = None,
        cost_reduction_policy: CostReductionPolicy|None = None,
    ):
        super().__init__(factor_graph)
        self.damping_policy = damping_policy
        self.cost_reduction_policy = cost_reduction_policy

    def run_inference(self, max_iters: int = 10):
        for iteration in range(max_iters):
            # Apply cost reduction
            if self.cost_reduction_policy.should_apply(iteration):
                alpha = self.cost_reduction_policy.get_alpha(iteration)
                for factor in self.factor_graph.all_factors():
                    factor.potential_table *= alpha

            # Update messages (simplified; actual logic goes here)
            for var_node in self.factor_graph.all_variables():
                for neighbor in var_node.neighbors:
                    # Update Q and R messages here
                    pass

    def get_beliefs(self) -> Dict[str, np.ndarray]:
        # Compute beliefs from Q and R messages
        beliefs = {}
        for var_node in self.factor_graph.all_variables():
            beliefs[var_node.name] = np.sum(
                [self.R[(f.name, var_node.name)] for f in var_node.neighbors], axis=0
            )
        return beliefs

    def get_map_estimate(self) -> Dict[str, int]:
        # Return the MAP estimate based on beliefs
        beliefs = self.get_beliefs()
        return {var: np.argmin(cost) for var, cost in beliefs.items()}