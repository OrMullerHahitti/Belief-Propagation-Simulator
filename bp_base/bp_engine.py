import typing
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Tuple, Any
import numpy as np
import networkx as nx
import json
import os
from bp_base.agents import BPAgent, VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.computators import MaxSumComputator
from bp_base.factor_graph import FactorGraph
from DCOP_base import Computator, Agent
from functools import reduce
from bp_base.typing import Policy, PolicyType
from dataclasses import dataclass, field

T = typing.TypeVar("T")
""" in this module we will implement the belief propagation with various policies with factor graph configs
most of which are implemented in the factor graph module and will be max-sum with different policies and different structures
we will start with the usual 3-cycle and then move to more complex structures"""


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

    def __setitem__(self, key: int, value: Cycle):
        self.cycles[key] = value

    def __getitem__(self, key: int):
        return self.cycles[key]

    def compare_last_two_cycles(self):
        if len(self.cycles) < 2:
            return False
        last_cycle = list(self.assignments.values())[-1]
        second_last_cycle = list(self.assignments.values())[-2]
        return list(last_cycle.values()) == list(second_last_cycle.values())

    @property
    def name(self):
        return f"{self.config['factor_graph'].name}_{self.config['computator']}_{self.config['policies'] if self.config['policies'] else ''}"

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

        # 3) write it out
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        return filename


### TODO: create a wrapper to config everything beforehand
### TODO: add a class to handle the policies and the history of the beliefs
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
    ):
        """
        Initialize the belief propagation engine.
        :param factor_graph:
        :param computator:
        :param policies:
        """
        self.graph = factor_graph
        self.graph.set_computator(computator)  # Store history of beliefs
        self.policies = policies  # Store policies - with all different kinds - message , cost table, stopping critiria, etc.
        self.history = History(
            computator=computator, policies=policies, factor_graph=factor_graph
        )  # Store history of beliefs

    # TODO:maybe apply here cost table policies too? or after a cycle? should think this over will open issue
    def step(self, i: int = 0) -> Step:
        """Run the factor graph algorithm."""
        step = Step(i)
        # compute messages to send and put them in the mailbox
        # TODO: save the messages into the _history of variable nodes
        # TODO change it to work on bipartite graph running on both sides one after another - best practice
        for agent in self.graph.G.nodes():
            self.graph.normalize_messages()
            agent.compute_messages()
            agent.empty_mailbox()
            # clear the mailbox
            agent.mailer.send()
            if isinstance(agent, FactorAgent):
                for message in agent.mailer.outbox:
                    step.add(message.recipient, message)
            agent.mailer.prepare()
            # both sending and receiving

            # apply message policies
            # TODO next work
            """ 
            if self.policies and "message" in self.policies and self.policies["message"]:
                for message in agent.messages_to_send:
                    if isinstance(message, Message):
                        message.data = self._apply_policies(self.policies["message"], message)
                        """
            # add the message to the mailbox

        # send the messages to the right nodes

        return step

    def cycle(self, j) -> Cycle:
        cy = Cycle(j)
        # TODO add diameter in the factorgraph class
        for i in range(nx.diameter(self.graph.G)):
            cy.add(self.step(i))
        self.history.beliefs[j] = self.get_beliefs()
        self.history.assignments[j] = self.assignments
        return cy

    def run(
        self, max_iter: int = 1000, save_json: bool = True, filename: str = None
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
        # Get current assignments
        var_assignments = {
            node: node.curr_assignment
            for node in self.graph.G.nodes()
            if isinstance(node, VariableAgent)
        }

        total_cost = 0.0
        # For each factor, calculate the cost based on the assignments of its connected variables
        for node in self.graph.G.nodes():
            if isinstance(node, FactorAgent) and node.cost_table is not None:
                # Get the indices for the cost table based on the assignments of connected variables
                indices = []
                for var, dim in node.connection_number.items():
                    if var in var_assignments:
                        indices.append(var_assignments[var])

                # If we have assignments for all connected variables, add the cost
                if len(indices) == len(node.connection_number):
                    total_cost += node.cost_table[tuple(indices)]

        return total_cost

    @staticmethod
    def _apply_policies(policies: List[Policy], data: T) -> T:
        """
        Apply the policies to the factor graph.
        """
        return reduce(lambda acc, policy: policy(acc), policies, data)
