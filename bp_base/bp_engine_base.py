import logging
import typing
from typing import Dict, List, Optional
import numpy as np
import networkx as nx
from policies.normalize_cost import init_normalization, normalize_after_cycle
import json
import os
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.computators import MinSumComputator, MaxSumComputator
from bp_base.engine_components import History, Cycle, Step
from bp_base.factor_graph import FactorGraph
from bp_base.DCOP_base import Computator, Agent
from bp_base.typing import Policy, PolicyType
from policies.convergance import ConvergenceMonitor, ConvergenceConfig
from utils.performance import PerformanceMonitor
from dataclasses import dataclass, field

from configs.loggers import Logger
from utils.fg_utils import generate_random_cost

T = typing.TypeVar("T")

logger = Logger(__name__, file=True)
logger.setLevel(100)


class BPEngine:
    """
    Abstract engine for belief propagation with fixed synchronization.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: Computator = MinSumComputator(),
        policies: Dict[PolicyType, List[Policy]] | None = None,
        name: str = "BPEngine",
        normalize: bool = False,
        convergence_config: ConvergenceConfig | None = None,
        monitor_performance: bool = False,
    ):
        """
        Initialize the belief propagation engine.
        """
        self.name = name
        self.graph = factor_graph
        self.var_nodes, self.factor_nodes = nx.bipartite.sets(self.graph.G)

        # Initialize components
        self.post_init()
        self.graph.set_computator(computator)

        # Setup history
        engine_type = self.__class__.__name__
        self.history = History(
            engine_type=engine_type,
            computator=computator,
            policies=policies,
            factor_graph=factor_graph,
        )

        self.graph_diameter = nx.diameter(self.graph.G)

        # Normalization
        if normalize:
            init_normalization(list(self.factor_nodes))


        self.convergence_monitor = ConvergenceMonitor(convergence_config)
        self.performance_monitor = PerformanceMonitor() if monitor_performance else None

    def step(self, i: int = 0) -> Step:
        """Run one step of belief propagation with proper synchronization."""
        if self.performance_monitor:
            start_time = self.performance_monitor.start_step()

        step = Step(i)

        # Phase 1: All variables compute messages (but don't send yet)
        for var in self.var_nodes:
            var.compute_messages()
            self.post_var_compute(var)

        # Phase 2: All variables send messages at once
        for var in self.var_nodes:
            var.mailer.send()

        # Phase 3: Clear variable mailboxes and prepare for next round
        for var in self.var_nodes:
            var.empty_mailbox()
            var.mailer.prepare()

        # Phase 4: All factors compute messages
        for factor in self.factor_nodes:
            factor.compute_messages()
            self.post_factor_step()

        # Phase 5: All factors send messages at once
        for factor in self.factor_nodes:
            factor.mailer.send()
            # Record messages in step
            for message in factor.mailer.outbox:
                step.add(message.recipient, message)

        # Phase 6: Clear factor mailboxes and prepare
        for factor in self.factor_nodes:
            factor.empty_mailbox()
            factor.mailer.prepare()

        # Calculate global cost after all messages are sent
        global_cost = self.calculate_global_cost()
        self.history.costs.append(global_cost)

        # Performance monitoring
        if self.performance_monitor:
            all_messages = []
            for node in self.graph.G.nodes():
                all_messages.extend(node.mailer.outbox)
            self.performance_monitor.end_step(start_time, i, all_messages)

        return step

    def cycle(self, j) -> Cycle:
        """Run one complete cycle of belief propagation."""
        cy = Cycle(j)

        # Run diameter + 1 steps
        for i in range(self.graph_diameter):
            step_result = self.step(i)
            cy.add(step_result)
        # Post-cycle operations
        if j == 2:
            self.post_two_cycles()
        self.post_var_cycle()
        self.post_factor_cycle()
        normalize_after_cycle(self.graph)
        # Update beliefs and assignments
        self.history.beliefs[j] = self.get_beliefs()
        self.history.assignments[j] = self.assignments

        return cy

    def run(
        self,
        max_iter: int = 1000,
        save_json: bool = False,
        save_csv: bool = True,
        filename: str = None,
        config_name: str = None,
    ) -> Optional[str]:
        """
        Run the factor graph algorithm for a maximum number of iterations.
        """
        if config_name is None:
            config_name = self._generate_config_name()

        # Reset convergence monitor
        self.convergence_monitor.reset()

        for i in range(max_iter):
            self.history[i] = self.cycle(i)

            if self._is_converged():
                logger.info(f"Converged after {i + 1} cycles")
                break

        # Save results
        if save_json:
            self.history.save_results(filename or "results.json")
        if save_csv:
            self.history.save_csv(config_name)

        # Log performance summary if monitoring
        if self.performance_monitor:
            summary = self.performance_monitor.get_summary()
            logger.info(f"Performance summary: {summary}")

        return None

    def _generate_config_name(self) -> str:
        """Generate a configuration name based on the engine parameters."""
        config_name = self.name

        if hasattr(self, "p"):
            config_name += f"_p{self.p}"
        if hasattr(self, "cr"):
            config_name += f"_cr{self.cr}"
        if hasattr(self, "damping_factor"):
            config_name += f"_df{self.damping_factor}"

        return config_name

    def get_beliefs(self) -> Dict[str, np.ndarray]:
        """Return the beliefs of the factor graph."""
        beliefs = {}
        for node in self.var_nodes:
            if isinstance(node, VariableAgent):
                beliefs[node.name] = getattr(node, "belief", None)
        return beliefs

    def _is_converged(self) -> bool:
        """Check convergence using the monitor."""
        if not self.history.beliefs or not self.history.assignments:
            return False

        latest_cycle = max(self.history.beliefs.keys())
        beliefs = self.history.beliefs[latest_cycle]
        assignments = self.history.assignments[latest_cycle]

        return self.convergence_monitor.check_convergence(beliefs, assignments)

    @property
    def assignments(self) -> Dict[str, int | float]:
        """Get the assignments of the factor graph."""
        return {
            node.name: node.curr_assignment
            for node in self.var_nodes
            if isinstance(node, VariableAgent)
        }

    def calculate_global_cost(self) -> float:
        """Calculate the global cost based on current assignments."""
        var_assignments = {node.name: node.curr_assignment for node in self.var_nodes}

        total_cost = 0.0
        for factor in self.factor_nodes:
            if factor.cost_table is not None:
                indices = []
                for var_name, dim in factor.connection_number.items():
                    if var_name in var_assignments:
                        # Ensure indices list is the right size
                        while len(indices) <= dim:
                            indices.append(None)
                        indices[dim] = var_assignments[var_name]

                # Check if we have all indices
                if None not in indices and len(indices) == len(
                    factor.connection_number
                ):
                    if factor.original_cost_table is not None:
                        total_cost += factor.original_cost_table[tuple(indices)]
                    else:
                        total_cost += factor.cost_table[tuple(indices)]

        return total_cost

    def __str__(self):
        return f"{self.name}"

    # Hook methods for subclasses
    def post_init(self) -> None:
        pass

    def post_var_cycle(self) -> None:
        pass

    def post_factor_step(self) -> None:
        pass

    def post_factor_cycle(self):
        pass

    def post_two_cycles(self):
        pass

    def post_var_compute(self, var: VariableAgent):
        pass
