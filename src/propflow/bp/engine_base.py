import typing
from typing import Dict, Optional, Callable
import numpy as np
import networkx as nx
from ..policies.normalize_cost import normalize_inbox
from ..core.agents import VariableAgent, FactorAgent
from .computators import MinSumComputator
from .engine_components import History, Step
from .factor_graph import FactorGraph
from ..core.dcop_base import Computator
from ..policies.convergance import ConvergenceMonitor, ConvergenceConfig
from ..utils.tools.performance import PerformanceMonitor

from ..configs.loggers import Logger
from ..configs.global_config_mapping import ENGINE_DEFAULTS
from ..utils import dummy_func

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
        init_normalization: Callable = dummy_func,
        name: str = "BPEngine",
        convergence_config: ConvergenceConfig | None = None,
        monitor_performance: bool = None,
        normalize_messages: bool = None,
        anytime: bool = None,
        use_bct_history: bool = None,
    ):
        """
        Initialize the belief propagation engine.
        """
        # Apply defaults from global config with override capability
        self.computator = computator
        self.anytime = anytime if anytime is not None else ENGINE_DEFAULTS["anytime"]
        self.normalize_messages = (
            normalize_messages
            if normalize_messages is not None
            else ENGINE_DEFAULTS["normalize_messages"]
        )
        self.graph = factor_graph
        self.post_init()
        self._initialize_messages()
        self.graph.set_computator(self.computator)
        self.var_nodes, self.factor_nodes = nx.bipartite.sets(self.graph.G)

        # Setup history
        engine_type = self.__class__.__name__
        use_bct = (
            use_bct_history
            if use_bct_history is not None
            else ENGINE_DEFAULTS["use_bct_history"]
        )
        self.history = History(
            engine_type=engine_type,
            computator=computator,
            factor_graph=factor_graph,
            use_bct_history=use_bct,
        )

        self.graph_diameter = nx.diameter(self.graph.G)
        self.convergence_monitor = ConvergenceMonitor(convergence_config)
        monitor_perf = (
            monitor_performance
            if monitor_performance is not None
            else ENGINE_DEFAULTS["monitor_performance"]
        )
        self.performance_monitor = PerformanceMonitor() if monitor_perf else None
        self._name = name
        init_normalization(self.factor_nodes)

    def step(self, i: int = 0) -> Step:
        """Run one step with message pruning support."""
        if self.performance_monitor:
            start_time = self.performance_monitor.start_step()

        step = Step(i)

        # Phase 1: All variables compute messages
        for var in self.var_nodes:
            var.compute_messages()
            self.post_var_compute(var)

        # Phase 2: All variables send messages
        for var in self.var_nodes:
            var.mailer.send()

        # Phase 3: Clear and prepare variables
        for var in self.var_nodes:
            var.empty_mailbox()
            var.mailer.prepare()

        # Phase 4: All factors compute messages
        for factor in self.factor_nodes:
            self.pre_factor_compute(factor, i)
            factor.compute_messages()
            self.post_factor_compute(factor, i)

        # Phase 5: All factors send messages
        for factor in self.factor_nodes:
            factor.mailer.send()
            for message in factor.mailer.outbox:
                step.add(message.recipient, message)

        # Phase 6: Clear and prepare factors
        for factor in self.factor_nodes:
            factor.empty_mailbox()
            factor.mailer.prepare()

        self.update_global_cost()
        self.history.track_step_data(i, step, self)

        if self.performance_monitor:
            step_matric = self.performance_monitor.end_step(start_time, i)
        return step

    def run(
        self,
        max_iter: int = None,
        save_json: bool = False,
        save_csv: bool = True,
        filename: str = None,
        config_name: str = None,
    ) -> Optional[str]:
        """
        Run the factor graph algorithm for a maximum number of iterations.
        """
        max_iterations = (
            max_iter if max_iter is not None else ENGINE_DEFAULTS["max_iterations"]
        )
        self.convergence_monitor.reset()
        for i in range(max_iterations):
            self.step(i)
            try:
                self._handle_cycle_events(i)
            except StopIteration:
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

    def _set_name(self, kwargs=Optional[Dict[str, str]]) -> None:
        """Generate a configuration name based on the engine parameters."""
        config_name = self._name
        for k, v in kwargs.items():
            config_name += f"_{str(k)}-{str(v)}"

        self._name = config_name

    @property
    def name(self) -> str:
        """Get the name of the engine."""
        return self._name

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
        for factor in self.graph._original_factors:
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

    def _initialize_messages(self) -> None:
        """
        Initialize mailboxes for all nodes with zero messages.
        Each node creates outgoing messages to all its neighbors.
        """
        # For each node, create outgoing messages to all its neighbors
        for node in self.graph.G.nodes():
            neighbors = list(self.graph.G.neighbors(node))
            if isinstance(node, VariableAgent):
                for neighbor in neighbors:
                    # Check if neighbor has a domain attribute
                    logger.info("Initializing mailbox for node: %s", node)

                    node.mailer.set_first_message(node, neighbor)
                    # Initialize messages to send

    def __str__(self):
        return f"{self.name}"

    def post_init(self) -> None:
        pass

    def post_factor_cycle(self):
        pass

    def post_two_cycles(self):
        pass

    def pre_factor_compute(self, factor: FactorAgent, iteration: int = 0):
        pass

    def post_factor_compute(self, factor: FactorAgent, iteration: int):
        pass

    def pre_var_compute(self, var: VariableAgent):
        pass

    def post_var_compute(self, var: VariableAgent):
        pass

    def init_normalize(self) -> None:
        pass

    def update_global_cost(self) -> None:
        """

        Placeholder for any time-specific logic.
        This method can be overridden in subclasses if needed.
        """
        cost = self.calculate_global_cost()
        # Only compare with last cost if it exists
        if self.anytime and self.history.costs and self.history.costs[-1] > cost:
            self.history.costs.append(self.history.costs[-1])
            return
        self.history.costs.append(cost)

    def normalize_messages(self) -> None:
        """
        Normalize messages in the factor graph.
        This is a placeholder for normalization logic.
        """
        pass

    def _handle_cycle_events(self, i: int):
        if i == 2 * self.graph_diameter:
            self._handle_two_cycle_event()
        if i % self.graph_diameter == 0:
            self._handle_regular_cycle_event(i)

    def _handle_two_cycle_event(self):
        self.post_two_cycles()

    def _handle_regular_cycle_event(self, i: int):
        if self.normalize_messages:
            normalize_inbox(self.var_nodes)
        self.history.beliefs[i] = self.get_beliefs()
        self.history.assignments[i] = self.assignments
        if self._is_converged():
            logger.debug(f"Converged after {i + 1} steps")
            raise StopIteration
