from typing import Callable, Dict, Optional, List
import numpy as np
import networkx as nx
from ..policies.normalize_cost import normalize_inbox
from ..core.agents import VariableAgent, FactorAgent
from .computators import BPComputator, MinSumComputator
from .engine_components import Step, SnapshotHistoryView
from .factor_graph import FactorGraph
from ..core.dcop_base import Computator
from ..policies.convergance import ConvergenceMonitor, ConvergenceConfig
from ..snapshots import SnapshotManager, EngineSnapshot
from ..utils.tools.performance import PerformanceMonitor

from ..configs.loggers import Logger
from ..configs.global_config_mapping import EngineDefaults
from ..utils import dummy_func

logger = Logger(__name__, file=True)
logger.setLevel(100)


class BPEngine:
    """The core engine for running belief propagation simulations.

    This class orchestrates the belief propagation process on a factor graph.
    It manages the simulation loop, message passing schedule, history tracking,
    convergence checking, and performance monitoring. It is designed to be
    extended by other engine classes that implement specific BP variants or policies.

    Attributes:
        computator (Computator): The computator instance for message calculation.
        anytime (bool): If True, the engine tracks the best cost found so far.
        normalize_messages (bool): If True, messages are normalized each cycle.
        graph (FactorGraph): The factor graph on which the simulation runs.
        var_nodes (list): A list of variable agent nodes in the graph.
        factor_nodes (list): A list of factor agent nodes in the graph.
        history (History): An object for tracking the simulation's history.
        graph_diameter (int): The diameter of the factor graph.
        convergence_monitor (ConvergenceMonitor): The monitor for checking convergence.
        performance_monitor (PerformanceMonitor): An optional monitor for performance.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: Computator | BPComputator = MinSumComputator(),
        init_normalization: Callable = dummy_func,
        name: str = "BPEngine",
        convergence_config: ConvergenceConfig | None = None,
        monitor_performance: bool | None = None,
        normalize_messages: bool | None = None,
        anytime: bool | None = None,
        use_bct_history: bool | None = None,
        snapshot_manager: SnapshotManager | None = None,
    ):
        """Initializes the belief propagation engine.

        Args:
            factor_graph: The factor graph for the simulation.
            computator: The message computation logic. Defaults to MinSumComputator().
            init_normalization: A function to normalize initial cost tables.
            name: The name of the engine instance.
            convergence_config: Configuration for the convergence monitor.
            monitor_performance: Whether to monitor performance.
            normalize_messages: Whether to normalize messages during execution.
            anytime: Whether to operate in "anytime" mode, tracking best cost.
            use_bct_history: Retained for backwards compatibility (no-op).
            snapshot_manager: Optional custom snapshot manager to use.
        """
        # Apply defaults from global config with override capability
        self.computator = computator
        self.anytime = anytime if anytime is not None else EngineDefaults().anytime
        self.normalize_messages = (
            normalize_messages
            if normalize_messages is not None
            else EngineDefaults().normalize_messages
        ) # type: ignore
        self.graph = factor_graph
        self.post_init()
        self._initialize_messages()
        self.graph.set_computator(self.computator) # type: ignore
        var_set, factor_set = nx.bipartite.sets(self.graph.G)
        self.var_nodes = sorted(var_set, key=lambda node: node.name)
        self.factor_nodes = sorted(factor_set, key=lambda node: node.name)

        engine_type = self.__class__.__name__
        use_bct = (
            use_bct_history
            if use_bct_history is not None
            else EngineDefaults().use_bct_history
        )
        self._use_bct_history = bool(use_bct)
        self._last_cost: float | None = None
        self.snapshot_manager = snapshot_manager or SnapshotManager()
        self._snapshots: dict[int, EngineSnapshot] = {}

        self.graph_diameter = nx.diameter(self.graph.G)
        self.convergence_monitor = ConvergenceMonitor(convergence_config)
        monitor_perf = (
            monitor_performance
            if monitor_performance is not None
            else EngineDefaults().monitor_performance
        )
        self.performance_monitor = PerformanceMonitor() if monitor_perf else None
        self._name = name
        init_normalization(self.factor_nodes)
        self.history = SnapshotHistoryView(
            self,
            engine_type=engine_type,
            config={"computator": computator, "factor_graph": factor_graph},
            use_bct_history=self._use_bct_history,
        )

    def step(self, i: int = 0) -> Step:
        """Runs one full step of the synchronous belief propagation algorithm.

        A step consists of two main phases:
        1. Variable nodes compute and send messages to factor nodes.
        2. Factor nodes compute and send messages to variable nodes.

        Args:
            i: The current step number.

        Returns:
            A `Step` object containing information about the messages exchanged.
        """
        if self.performance_monitor:
            start_time = self.performance_monitor.start_step()

        step = Step(i)

        # Phase 1: All variables compute and send messages
        for var in self.var_nodes:
            var.compute_messages()
            self.post_var_compute(var)
            if var.mailer.outbox:
                outbox_copy = list(var.mailer.outbox)
                step.add_q(var.name, outbox_copy)
                for message in outbox_copy:
                    step.add(message.recipient, message)
        for var in self.var_nodes:
            var.mailer.send()
        for var in self.var_nodes:
            var.empty_mailbox()
            var.mailer.prepare()

        # Phase 2: All factors compute and send messages
        for factor in self.factor_nodes:
            self.pre_factor_compute(factor, i)
            factor.compute_messages()
            self.post_factor_compute(factor, i)
            if factor.mailer.outbox:
                outbox_copy = list(factor.mailer.outbox)
                step.add_r(factor.name, outbox_copy)
                for message in outbox_copy:
                    step.add(message.recipient, message)
        for factor in self.factor_nodes:
            factor.mailer.send()
        for factor in self.factor_nodes:
            factor.empty_mailbox()
            factor.mailer.prepare()

        cost = self.update_global_cost()
        snapshot = self.snapshot_manager.capture_step(i, step, self)
        if snapshot.global_cost is None:
            snapshot.global_cost = cost
        if self._use_bct_history:
            self.snapshot_manager.capture_bct_data(snapshot, self)
        self._snapshots[i] = snapshot

        if self.performance_monitor:
            self.performance_monitor.end_step(start_time, i)

        return step

    def run(
        self,
        max_iter: int | None = None,
        save_json: bool = False,
        save_csv: bool = False,
        filename: str = None,
        config_name: str | None = None,
    ) -> Optional[str]:
        """Runs the simulation until convergence or max iterations is reached.

        Args:
            max_iter: The maximum number of iterations to run.
            save_json: Whether to save the full history as a JSON file.
            save_csv: Whether to save the cost history as a CSV file.
            filename: The base name for the output files.
            config_name: The name of the configuration to use for saving.

        Returns:
            An optional string, typically for results or status.
        """
        max_iterations = (
            max_iter if max_iter is not None else EngineDefaults().max_iterations
        )
        self.convergence_monitor.reset()
        for i in range(max_iterations):
            self.step(i)
            try:
                self._handle_cycle_events(i)
            except StopIteration:
                break

        if save_json or save_csv:
            logger.warning(
                "Snapshot persistence is not implemented in the simplified snapshot manager."
            )

        if self.performance_monitor:
            summary = self.performance_monitor.get_summary()
            logger.info(f"Performance summary: {summary}")

        return None

    def _set_name(self, kwargs: Optional[Dict[str, str]] = None) -> None:
        """Generates a configuration name based on engine parameters."""
        config_name = self._name
        if kwargs:
            for k, v in kwargs.items():
                config_name += f"_{str(k)}-{str(v)}"
        self._name = config_name

    @property
    def name(self) -> str:
        """str: The name of the engine instance."""
        return self._name

    @property
    def iteration_count(self) -> int:
        """int: The number of iterations completed so far."""
        return len(self._snapshots)

    @property
    def use_bct_history(self) -> bool:
        return self._use_bct_history

    def get_beliefs(self) -> Dict[str, np.ndarray]:
        """Retrieves the current beliefs of all variable nodes.

        Returns:
            A dictionary mapping variable names to their belief vectors.
        """
        beliefs = {}
        for node in self.var_nodes:
            if isinstance(node, VariableAgent):
                beliefs[node.name] = getattr(node, "belief", None)
        return beliefs

    def _is_converged(self) -> bool:
        """Checks if the simulation has converged."""
        beliefs_dict = {
            name: np.asarray(value, dtype=float)
            for name, value in self.get_beliefs().items()
            if value is not None
        }
        assignments_dict = self.assignments
        if not beliefs_dict or not assignments_dict:
            return False

        return self.convergence_monitor.check_convergence(
            beliefs_dict, assignments_dict
        )

    @property
    def assignments(self) -> Dict[str, int | float]:
        """dict: The current assignments of all variable nodes."""
        return {
            node.name: node.curr_assignment
            for node in self.var_nodes
            if isinstance(node, VariableAgent)
        }

    def calculate_global_cost(self) -> float:
        """Calculates the global cost based on the current variable assignments.

        This method uses the original, unmodified factor cost tables to ensure
        the true cost is computed, independent of any runtime cost modifications.

        Returns:
            The total cost of the current assignments.
        """
        var_assignments = {node.name: node.curr_assignment for node in self.var_nodes}
        total_cost = 0.0
        for factor in self.graph._original_factors:
            if factor.cost_table is not None:
                indices = []
                for var_name, dim in factor.connection_number.items():
                    if var_name in var_assignments:
                        while len(indices) <= dim:
                            indices.append(None)
                        indices[dim] = var_assignments[var_name]

                if None not in indices and len(indices) == len(
                    factor.connection_number
                ):
                    cost_table = (
                        factor.original_cost_table
                        if factor.original_cost_table is not None
                        else factor.cost_table
                    )
                    total_cost += cost_table[tuple(indices)]
        return total_cost

    def _initialize_messages(self) -> None:
        """Initializes mailboxes for all nodes with zero-messages.

        This ensures that every node has a message from each neighbor before
        the first computation step, preventing errors with missing messages.
        """
        for node in self.graph.G.nodes():
            neighbors = list(self.graph.G.neighbors(node))
            if isinstance(node, VariableAgent):
                for neighbor in neighbors:
                    logger.info("Initializing mailbox for node: %s", node)
                    node.mailer.set_first_message(node, neighbor)

    def __str__(self) -> str:
        """Returns the name of the engine."""
        return f"{self.name}"

    def post_init(self) -> None:
        """Hook for logic to be executed after engine initialization."""
        pass

    def post_factor_cycle(self) -> None:
        """Hook for logic after a full message passing cycle."""
        pass

    def post_two_cycles(self) -> None:
        """Hook for logic after the first two message passing cycles."""
        pass

    def pre_factor_compute(self, factor: FactorAgent, iteration: int = 0) -> None:
        """Hook for logic before a factor computes its messages.

        Args:
            factor: The factor agent about to compute messages.
            iteration: The current simulation iteration.
        """
        pass

    def post_factor_compute(self, factor: FactorAgent, iteration: int) -> None:
        """Hook for logic after a factor computes its messages.

        Args:
            factor: The factor agent that just computed messages.
            iteration: The current simulation iteration.
        """
        pass

    def pre_var_compute(self, var: VariableAgent) -> None:
        """Hook for logic before a variable computes its messages.

        Args:
            var: The variable agent about to compute messages.
        """
        pass

    def post_var_compute(self, var: VariableAgent) -> None:
        """Hook for logic after a variable computes its messages.

        Args:
            var: The variable agent that just computed messages.
        """
        pass

    def init_normalize(self) -> None:
        """Hook for initial normalization logic."""
        pass

    def update_global_cost(self) -> float:
        """Calculates and records the global cost for the current step.

        If in "anytime" mode, it ensures the cost recorded does not increase.
        """
        cost = float(self.calculate_global_cost())
        if self.anytime and self._last_cost is not None and self._last_cost < cost:
            cost = float(self._last_cost)
        self._last_cost = cost
        return cost

    def normalize_messages(self) -> None:
        """Placeholder hook for message normalization logic."""
        pass

    def _handle_cycle_events(self, i: int) -> None:
        """Handles events that occur at specific cycle intervals.

        This includes checking for convergence and calling cycle-based hooks.

        Args:
            i: The current iteration number.

        Raises:
            StopIteration: If convergence is detected.
        """
        if i == 2 * self.graph_diameter:
            self._handle_two_cycle_event()
        if i % self.graph_diameter == 0:
            self._handle_regular_cycle_event(i)

    def _handle_two_cycle_event(self) -> None:
        """Calls the hook for the two-cycle event."""
        self.post_two_cycles()

    def _handle_regular_cycle_event(self, i: int) -> None:
        """Handles events for regular message passing cycles.

        This involves normalizing messages (if enabled), tracking beliefs and
        assignments, and checking for convergence.

        Args:
            i: The current iteration number.
        """
        if self.normalize_messages:
            normalize_inbox(self.var_nodes)
        if self._is_converged():
            logger.debug(f"Converged after {i + 1} steps")
            raise StopIteration

    # --- Snapshots convenience API ---
    def latest_snapshot(self) -> Optional[EngineSnapshot]:
        if not self._snapshots:
            return None
        latest_step = max(self._snapshots)
        return self._snapshots[latest_step]

    def get_snapshot(self, step_index: int) -> Optional[EngineSnapshot]:
        return self._snapshots.get(step_index)

    @property
    def snapshots(self) -> List[EngineSnapshot]:
        return [self._snapshots[idx] for idx in sorted(self._snapshots)]

    @property
    def snapshot_map(self) -> Dict[int, EngineSnapshot]:
        return self._snapshots
