import json
import os
from dataclasses import field, dataclass
from typing import Dict, List

import numpy as np

from bp_base.DCOP_base import Agent
from bp_base.components import Message


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

    def initialize_cost(self, x: int | float) -> None:
        self.costs[0] = x

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
