import typing
from typing import Dict, List
import numpy as np
import networkx as nx
import json
import os
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.computators import MinSumComputator, MaxSumComputator
from bp_base.factor_graph import FactorGraph
from bp_base.DCOP_base import Computator, Agent
from functools import reduce
from bp_base.typing import Policy, PolicyType
from dataclasses import dataclass, field

from configs.loggers import Logger

T = typing.TypeVar("T")
""" in this module we will implement the belief propagation with various policies with factor graph configs
most of which are implemented in the factor graph module and will be max-sum with different policies and different structures
we will start with the usual 3-cycle and then move to more complex structures"""

logger = Logger(__name__, file=True)


@dataclass
class Step:
    """
    A class to represent a step in the factor graph.
    """

    num: int = 0
    messages: Dict[str, List[Message]] = field(default_factory=dict)

    def add(self, agent: Agent, message: Message):
        """
        Add a List of messages for each agent per step.
        :param agent: Agent who will send the messages next step
        :param message: the messages to be sent
        :return:
        """
        if agent.name not in self.messages:
            self.messages[agent.name] = []
        # Ensure messages is a list, even if None
        self.messages[agent.name].append(message)


@dataclass
class Cycle:
    """
    A class to represent a cycle in the factor graph.
    """

    number: int
    steps: List[Step] = field(default_factory=list)

    def add(self, step: Step):
        """
        Add a step to the cycle.
        """
        self.steps.append(step)

    def __eq__(self, other: "Cycle"):
        """
        Check if two cycles are equal.
        """
        if len(self.steps) != len(other.steps):
            return False
        for step1, step2 in zip(self.steps, other.steps):
            if step1.messages != step2.messages:
                return False
        return True


class History:
    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        self.cycles: Dict[int, Cycle] = {}
        self.beliefs: Dict[int, Dict[str, np.ndarray]] = {}
        self.assignments: Dict[int, Dict[str, int | float]] = {}
        self.costs: Dict[int, float] = {}  # Add dictionary to store costs per cycle

    def __setitem__(self, key: int, value: Cycle):
        self.cycles[key] = value

    def __getitem__(self, key: int):
        return self.cycles[key]

    def compare_last_two_cycles(self):
        if len(self.cycles) < 2:
            return False
        last_iteration = list(self.cycles)[-1]
        last_cycle = list(self.assignments[last_iteration].values())
        second_last_cycle = list(self.assignments[last_iteration - 1].values())
        return last_cycle == second_last_cycle

    @property
    def name(self):
        # TODO add something that is not a test
        return f"test_1"

    def save_results(self, filename: str = None) -> str:
        """
        Save cycles, assignments and beliefs as pure-Python JSON.
        """
        if filename is None:
            filename = f"{self.name}.json"

        # build the raw data dict
        raw = {
            "name": self.name,
            # "cycles":     self.cycles,
            "assignments": self.assignments,
            "beliefs": self.beliefs,
            "costs": self.costs,  # Include costs in the saved data
        }

        #  normalize everything
        def normalize(obj):
            # NumPy scalars
            if isinstance(obj, np.generic):
                return obj.item()
            # NumPy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # custom objects
            if hasattr(obj, "__dict__"):
                return normalize(vars(obj))
            # dicts
            if isinstance(obj, dict):
                return {k: normalize(v) for k, v in obj.items()}
            # lists (or tuples)
            if isinstance(obj, (list, tuple)):
                return [normalize(v) for v in obj]
            # otherwise assume it's JSON-friendly already
            return obj

        data = normalize(raw)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        return filename

    def save_csv(self) -> str:
        """
        save only the global costs as csv ready for plotting
        """
        os.makedirs("results", exist_ok=True)
        # 2) write the data to a csv file
        with open(f"results/costs_{self.name}.csv", "w") as f:
            for cycle, cost in self.costs.items():
                f.write(f"{cycle},{cost}\n")
        return f"results/costs_{self.name}.csv"


### begining running the algorithm
class BPEngine:
    """
    Abstract engine for belief propagation.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: Computator = MaxSumComputator(),
        policies: Dict[PolicyType, List[Policy]] | None = None,
        name: str = "test",
    ):
        """
        Initialize the belief propagation engine.
        :param factor_graph:
        :param computator:
        :param policies:
        """
        self.name = name
        self.graph = factor_graph
        self.var_nodes, self.factor_nodes = nx.bipartite.sets(self.graph.G)
        self.post_init()
        self.graph.set_computator(computator)  # Store history of beliefs
        self.policies = policies  # Store policies - with all different kinds - message , cost table, stopping critiria, etc.
        self.history = History(
            computator=computator, policies=policies, factor_graph=factor_graph
        )  # Store history of beliefs

        # Pre-calculate graph diameter once - with fallback if it fails
        self.graph_diameter = nx.diameter(self.graph.G)

    def step(self, i: int = 0) -> Step:
        """Run the factor graph algorithm."""
        step = Step(i)
        # compute messages to send and put them in the mailbox
        for var in self.var_nodes:
            var.compute_messages()
            self.post_var_step()
            var.empty_mailbox()
            var.mailer.send()
            var.mailer.prepare()

        for factor in self.graph.G.nodes():
            factor.compute_messages()
            self.post_factor_step()
            factor.empty_mailbox()
            factor.mailer.send()
            for message in factor.mailer.outbox:
                step.add(message.recipient, message)
            factor.mailer.prepare()

        return step

    def cycle(self, j) -> Cycle:
        cy = Cycle(j)
        self.graph.normalize_messages()
        # Use pre-computed diameter instead of calculating it each time
        for i in range(self.graph_diameter + 1):
            logger.debug(f"Starting step {i} of cycle {j}")
            step_result = self.step(i)
            cy.add(step_result)
            logger.debug(f"Completed step {i}")

        logger.info(f"Updating beliefs and assignments for cycle {j}")
        self.history.beliefs[j] = self.get_beliefs()
        self.history.assignments[j] = self.assignments

        # Calculate and store the global cost at the end of the cycle
        logger.info(f"Calculating global cost for cycle {j}")
        global_cost = self.calculate_global_cost()
        self.history.costs[j] = global_cost
        logger.info(f"Global cost after cycle {j}: {global_cost}")

        return cy

    def run(
        self,
        max_iter: int = 1000,
        save_json: bool = False,
        save_csv: bool = True,
        filename: str = None,
    ) -> None:
        """
        Run the factor graph algorithm for a maximum number of iterations.

        Args:
            max_iter (int): Maximum number of iterations to run.
            save_json (bool): Whether to save the results as a JSON file.
            filename (str, optional): The name of the file to save. If None, a default name will be used.

        Returns:
            str or None: The path to the saved file if save_json is True, None otherwise.
        """
        for i in range(max_iter):
            self.history[i] = self.cycle(i)
            if self._is_converged():
                break

        # Save results as JSON if requested
        if save_json:
            self.history.save_results(
                "results.json"
            )  # TODO ; needs to be changed to a more general name
        if save_csv:
            self.history.save_csv()

        return None

    ### -------------------------------------------------------------------####
    # from here we will implement the getters/properties for the beliefs and the assignments
    def get_beliefs(self) -> Dict[str, np.ndarray]:
        """Return the beliefs of the factor graph.
        :return:
        :param: A dictionary mapping variable names to belief vectors."""
        beliefs = {}
        for node in self.graph.G.nodes():
            if isinstance(node, VariableAgent):
                beliefs[node.name] = getattr(node, "belief", None)
        return beliefs

    # TODO : make it modular with the policies (i.e. policy stopping criteria)
    def _is_converged(self) -> bool:
        return self.history.compare_last_two_cycles()

    @property
    def assignments(self) -> Dict[str, int | float]:
        """
        Get the assignments of the factor graph.
        Get the assignments of the factor graph.
        :return: A dictionary mapping variable names to their assignments.
        """
        return {
            node.name: node.curr_assignment
            for node in self.graph.G.nodes()
            if isinstance(node, VariableAgent)
        }

    def calculate_global_cost(self) -> float:
        """
        Calculate the global cost based on the current assignments of variables and the cost tables of factors.
        :return: The global cost as a float.
        """
        # PERFORMANCE IMPROVEMENT: Get variables and factors once using bipartite sets

        var_assignments = {node.name: node.curr_assignment for node in self.var_nodes}

        total_cost = 0.0
        # Only iterate through factor nodes
        for node in self.factor_nodes:
            if node.cost_table is not None:
                indices = []
                for var, dim in node.connection_number.items():
                    if var in var_assignments:
                        indices.append(var_assignments[var])

                # If we have assignments for all connected variables, add the cost
                if len(indices) == len(node.connection_number):
                    total_cost += node.cost_table[tuple(indices)]

        return total_cost

    # abstract methods to try splitting damping and cost reduction
    def post_init(self) -> None:
        return

    def post_var_step(self) -> None:
        return

    def post_factor_step(self) -> None:
        return
